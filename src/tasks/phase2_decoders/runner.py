from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import mse_and_corr, regression_metrics
from tasks.phase2_decoders.data import identity_hash
from tasks.phase2_decoders.models import DecoderModel, make_decoder


@dataclass(frozen=True)
class DecoderRunConfig:
    task: str
    decoder_id: str
    context_length: int = 8
    epoch_budgets: tuple[int, ...] = (15, 50, 100)
    seed: int = 0
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.task not in {"price_prediction", "volatility_prediction"}:
            raise ValueError("Stage 1 supports price_prediction and volatility_prediction")
        if not self.epoch_budgets or tuple(sorted(set(self.epoch_budgets))) != self.epoch_budgets:
            raise ValueError("epoch budgets must be unique and sorted")
        if any(v <= 0 for v in self.epoch_budgets) or self.context_length < 1 or self.batch_size < 1:
            raise ValueError("invalid positive configuration value")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self); result["epoch_budgets"] = list(self.epoch_budgets); return result


class IndexedDecoderDataset(Dataset):
    def __init__(self, features: np.ndarray, contexts: np.ndarray, targets: np.ndarray, temporal: bool) -> None:
        if len(contexts) != len(targets): raise ValueError("context and target counts differ")
        self.features = torch.as_tensor(np.asarray(features, np.float32))
        self.contexts = np.asarray(contexts, np.int64)
        self.targets = torch.as_tensor(np.asarray(targets, np.float32)).reshape(-1,1)
        self.temporal = temporal

    def __len__(self) -> int: return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.contexts[index]
        x = self.features[rows] if self.temporal else self.features[rows[-1]]
        return x, self.targets[index]


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def prepare_run_root(path: str | Path, overwrite: bool) -> Path:
    out=Path(path)
    if out.exists() and any(out.iterdir()):
        if not overwrite: raise FileExistsError(f"run directory is not empty: {out}; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True); return out


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload,indent=2,allow_nan=True),encoding="utf-8")


def _loader(features: np.ndarray, contexts: np.ndarray, targets: np.ndarray, config: DecoderRunConfig, shuffle: bool) -> DataLoader:
    generator=torch.Generator().manual_seed(config.seed)
    return DataLoader(IndexedDecoderDataset(features,contexts,targets,config.decoder_id in {"D3","D4"}),
                      batch_size=config.batch_size,shuffle=shuffle,generator=generator)


@torch.no_grad()
def predict(model: DecoderModel, loader: DataLoader, device: torch.device) -> tuple[np.ndarray,np.ndarray,np.ndarray|None]:
    model.eval(); predictions=[]; targets=[]; auxiliaries=[]
    for x,y in loader:
        output,aux=model.forward_with_aux(x.to(device)); predictions.append(output.cpu()); targets.append(y)
        if aux is not None: auxiliaries.append(aux.cpu())
    pred=torch.cat(predictions).reshape(-1).numpy().astype(np.float32)
    target=torch.cat(targets).reshape(-1).numpy().astype(np.float32)
    if not np.all(np.isfinite(pred)): raise FloatingPointError("non-finite predictions")
    return pred,target,(torch.cat(auxiliaries).numpy().astype(np.float32) if auxiliaries else None)


def _per_contract(pred: np.ndarray,target: np.ndarray,contracts: np.ndarray) -> dict[str,object]:
    result={}
    for cid in np.unique(contracts):
        mask=contracts==cid; p=torch.from_numpy(pred[mask]); y=torch.from_numpy(target[mask])
        metrics=regression_metrics(p,y); metrics.update(mse_and_corr(p,y)); result[str(int(cid))]={**metrics,"count":int(mask.sum())}
    return result


def run_decoder_experiment(
    *, train_features: np.ndarray, test_features: np.ndarray,
    train_contexts: np.ndarray, test_contexts: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray, identities: Mapping[str,np.ndarray],
    branch_dims: Mapping[str,int], scaler: Mapping[str,np.ndarray], run_root: str|Path,
    config: DecoderRunConfig, dataset_manifest: Mapping[str,object], overwrite: bool=False,
) -> list[dict[str,object]]:
    out=prepare_run_root(run_root,overwrite); set_seed(config.seed); device=resolve_device(config.device)
    for split,y,contexts in (("train",y_train,train_contexts),("test",y_test,test_contexts)):
        if len(y)!=len(contexts): raise ValueError(f"{split} target/context counts differ")
        for field in ("row_indices","contract_ids","window_starts","timestamps_ns"):
            if len(identities[f"{split}_{field}"])!=len(y): raise ValueError(f"{split} {field} count differs")
    model=make_decoder(config.decoder_id,branch_dims,config.task,config.context_length).to(device)
    parameter_count=count_parameters(model)
    architecture={"decoder_id":config.decoder_id,"branch_dims":dict(branch_dims),"parameter_count":parameter_count}
    _write_json(out/"config.json",config.to_dict()); _write_json(out/"architecture.json",architecture)
    np.savez(out/"feature_standardizer.npz",**scaler)
    manifest={**dict(dataset_manifest),**architecture,"train_sample_count":len(y_train),"test_sample_count":len(y_test),
              "train_identity_hash":identity_hash(identities["train_row_indices"],identities["train_contract_ids"],identities["train_window_starts"]),
              "test_identity_hash":identity_hash(identities["test_row_indices"],identities["test_contract_ids"],identities["test_window_starts"]),
              "test_evaluated_after_training":True}
    _write_json(out/"dataset_manifest.json",manifest)
    train_loader=_loader(train_features,train_contexts,y_train,config,True)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    criterion=nn.MSELoss(); history=[]; checkpoints=[]; checkpoint_seconds={}; started=time.perf_counter()
    diagnostics=[]
    for epoch in range(1,max(config.epoch_budgets)+1):
        model.train(); total=count=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True); prediction=model(x); loss=criterion(prediction,y)
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            if any(p.grad is not None and not torch.all(torch.isfinite(p.grad)) for p in model.parameters()):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            optimizer.step(); total+=float(loss.item())*len(y); count+=len(y)
        history.append(total/max(count,1)); diagnostics.append({"epoch":epoch,"train_loss":history[-1]})
        print(f"phase2 decoder={config.decoder_id} task={config.task} epoch={epoch} loss={history[-1]:.8f}",flush=True)
        if epoch in config.epoch_budgets:
            budget=out/f"e{epoch}"; budget.mkdir(parents=True,exist_ok=True); checkpoint=budget/"checkpoint.pth"
            torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                        "completed_epoch":epoch,"config":config.to_dict(),**architecture},checkpoint)
            np.savez(budget/"history.npz",train_loss=np.asarray(history,np.float32),epochs=np.arange(1,epoch+1),seed=np.asarray(config.seed))
            checkpoints.append(checkpoint); checkpoint_seconds[epoch]=time.perf_counter()-started
    training_seconds=time.perf_counter()-started; _write_json(out/"training_diagnostics.json",diagnostics)
    test_loader=_loader(test_features,test_contexts,y_test,config,False); sweep=[]
    for checkpoint in checkpoints:
        saved=torch.load(checkpoint,map_location=device); evaluated=make_decoder(config.decoder_id,branch_dims,config.task,config.context_length).to(device)
        evaluated.load_state_dict(saved["model_state_dict"])
        if device.type=="cuda": torch.cuda.synchronize()
        infer_start=time.perf_counter(); pred,target,aux=predict(evaluated,test_loader,device)
        if device.type=="cuda": torch.cuda.synchronize()
        inference_seconds=time.perf_counter()-infer_start
        metrics=regression_metrics(torch.from_numpy(pred),torch.from_numpy(target)); metrics.update(mse_and_corr(torch.from_numpy(pred),torch.from_numpy(target)))
        metrics["negative_prediction_fraction"]=float(np.mean(pred<0)); epoch=int(saved["completed_epoch"]); budget=checkpoint.parent
        payload={"predictions":pred,"targets":target,"row_indices":identities["test_row_indices"],
                 "contract_ids":identities["test_contract_ids"],"window_starts":identities["test_window_starts"],
                 "timestamps_ns":identities["test_timestamps_ns"]}
        if aux is not None: payload["gate_weights"]=aux
        np.savez_compressed(budget/"predictions.npz",**payload)
        row={**metrics,"epoch":epoch,"seed":config.seed,"task":config.task,"decoder_id":config.decoder_id,
             "train_loss":float(history[epoch-1]),"parameter_count":parameter_count,
             "training_seconds_to_checkpoint":checkpoint_seconds[epoch],"training_seconds_total":training_seconds,
             "inference_seconds":inference_seconds,"inference_rows_per_second":len(target)/max(inference_seconds,1e-12)}
        _write_json(budget/"metrics.json",row); _write_json(budget/"per_contract_metrics.json",_per_contract(pred,target,identities["test_contract_ids"]))
        with np.load(budget/"predictions.npz",allow_pickle=False) as replay:
            verified=all(np.array_equal(replay[k],payload[k]) for k in ("targets","row_indices","contract_ids","window_starts","timestamps_ns"))
        if not verified: raise RuntimeError("prediction replay failed")
        _write_json(budget/"replay.json",{"verified":True}); sweep.append(row)
    _write_json(out/"sweep_metrics.json",sweep); _write_json(out/"timing.json",{"training_seconds_total":training_seconds})
    lines=[f"# Phase 2 decoder {config.decoder_id}: {config.task}","","All snapshots are reported; none is selected from test performance.","", "| epoch | MAE | RMSE | MSE | correlation | train loss |","|---:|---:|---:|---:|---:|---:|"]
    for row in sweep: lines.append(f"| {row['epoch']} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['mse']:.6f} | {row['corr']:.6f} | {row['train_loss']:.6f} |")
    (out/"summary.md").write_text("\n".join(lines)+"\n"); return sweep
