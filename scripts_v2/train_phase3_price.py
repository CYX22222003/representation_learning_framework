"""Train one Phase 3 framework price configuration on contract-safe labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(Path(__file__).resolve().parent))
from tasks.price_prediction import PriceRegressor  # noqa: E402
from phase3_encoder_common import (DEFAULT_FEATURE_NPZ,DEFAULT_PRICE_LABELS,DEFAULT_PRICE_ROOT,DEFAULT_PROCESSED_NPZ,EPOCH_BUDGETS,PHASE3_EXPERIMENT_ROOT,environment_manifest,refuse_occupied,require_phase3_path,resolve_device,set_seed,sha256_file,write_json)  # noqa: E402
from phase3_price_common import PRICE_CONFIGS,run_root,source_branch  # noqa: E402
from extract_phase3_features import validate_features  # noqa: E402
from prepare_phase3_price_labels import validate_labels  # noqa: E402


def _load(features_path,labels_path,config_id):
    branches=PRICE_CONFIGS[config_id]
    with np.load(features_path,allow_pickle=False) as f, np.load(Path(f"{features_path}.index.npz"),allow_pickle=False) as idx, np.load(labels_path,allow_pickle=False) as lab:
        ntrain=int(idx["train_size"]); selected={}
        for name in branches:
            source=source_branch(name); values=np.asarray(f[source],np.float32); selected[name]=(values[:ntrain],values[ntrain:])
        train_i=np.asarray(lab["train_row_indices"],np.int64); test_i=np.asarray(lab["test_row_indices"],np.int64)
        ytrain=np.asarray(lab["train_labels"],np.float32); ytest=np.asarray(lab["test_labels"],np.float32)
        identities={name:np.asarray(lab[f"test_{name}"]) for name in ("contract_ids","window_starts","timestamps_ns")}
        identities["row_indices"]=test_i
    train={name:pair[0][train_i] for name,pair in selected.items()}; test={name:pair[1][test_i] for name,pair in selected.items()}
    return train,ytrain,test,ytest,identities


def _standardize(train,test):
    train_out={}; test_out={}; payload={}
    for name in train:
        mean=train[name].mean(0,dtype=np.float64).astype(np.float32); std=train[name].std(0,dtype=np.float64).astype(np.float32); std=np.where(std<1e-6,1.0,std).astype(np.float32)
        train_out[name]=np.clip((train[name]-mean)/std,-10,10).astype(np.float32); test_out[name]=np.clip((test[name]-mean)/std,-10,10).astype(np.float32)
        payload[f"{name}__mean"]=mean; payload[f"{name}__std"]=std
    return train_out,test_out,payload


def _concat(branches): return np.concatenate([branches[name] for name in branches],axis=1).astype(np.float32)


def _metrics(pred,target):
    error=pred-target; mse=float(np.mean(error**2));
    corr=float(np.corrcoef(pred,target)[0,1]) if len(pred)>1 and np.std(pred)>0 and np.std(target)>0 else float("nan")
    return {"mae":float(np.mean(np.abs(error))),"rmse":float(np.sqrt(mse)),"mse":mse,"corr":corr,
        "prediction_below_zero_fraction":float(np.mean(pred<0)),"prediction_above_one_fraction":float(np.mean(pred>1))}


def run(config_id,processed_npz,features_path,labels_path,out,device_name):
    if config_id not in PRICE_CONFIGS: raise ValueError("unknown price configuration")
    require_phase3_path(out,DEFAULT_PRICE_ROOT); refuse_occupied((out,)); validate_features(features_path,processed_npz); label_check=validate_labels(labels_path,processed_npz)
    train,ytrain,test,ytest,identities=_load(features_path,labels_path,config_id); train,test,scaler=_standardize(train,test)
    xtrain,xtest=_concat(train),_concat(test)
    if len(ytrain)!=len(xtrain) or len(ytest)!=len(xtest): raise ValueError("feature/label row mismatch")
    device=resolve_device(device_name); set_seed(0); model=PriceRegressor(xtrain.shape[1],128).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=1e-4); criterion=torch.nn.MSELoss()
    generator=torch.Generator().manual_seed(0); loader=DataLoader(TensorDataset(torch.from_numpy(xtrain),torch.from_numpy(ytrain[:,None])),batch_size=512,shuffle=True,generator=generator)
    out.mkdir(parents=True,exist_ok=False); np.savez_compressed(out/"feature_standardizer.npz",**scaler)
    write_json(out/"config.json",{"phase":3,"task":"price_prediction","config_id":config_id,"branches":list(PRICE_CONFIGS[config_id]),"fusion":"concat","head":"PriceRegressor(128,64,1; dropout=0.1; linear output)","loss":"MSE","epoch_budgets":list(EPOCH_BUDGETS),"seed":0,"batch_size":512,"learning_rate":1e-4,"standardize":"selected train rows only; clip +/-10"})
    write_json(out/"provenance.json",{"processed_npz":str(processed_npz.resolve()),"processed_sha256":sha256_file(processed_npz),"features_npz":str(features_path.resolve()),"features_sha256":sha256_file(features_path),"labels_npz":str(labels_path.resolve()),"labels_sha256":sha256_file(labels_path),**label_check,**environment_manifest(device)})
    history=[]; snapshots=[]; started=time.perf_counter(); parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    for epoch in range(1,51):
        model.train(); total=0.; count=0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad(set_to_none=True); pred=model(xb); loss=criterion(pred,yb)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite price loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError("non-finite price gradient")
            optimizer.step(); total+=float(loss.item())*len(xb); count+=len(xb)
        history.append(total/count); print(f"price {config_id} epoch={epoch} train_loss={history[-1]:.8f}",flush=True)
        if epoch not in EPOCH_BUDGETS: continue
        budget=out/f"e{epoch}"; budget.mkdir(); checkpoint=budget/"checkpoint.pth"; torch.save({"phase":3,"task":"price_prediction","config_id":config_id,"completed_epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"input_dim":xtrain.shape[1],"branches":list(PRICE_CONFIGS[config_id]),"processed_sha256":sha256_file(processed_npz),"features_sha256":sha256_file(features_path),"labels_sha256":sha256_file(labels_path)},checkpoint)
        np.savez_compressed(budget/"history.npz",epochs=np.arange(1,epoch+1,dtype=np.int32),train_loss=np.asarray(history,np.float64))
        model.eval(); preds=[]
        with torch.no_grad():
            for start in range(0,len(xtest),2048): preds.append(model(torch.from_numpy(xtest[start:start+2048]).to(device)).cpu().numpy().reshape(-1))
        prediction=np.concatenate(preds).astype(np.float32); metrics={**_metrics(prediction,ytest),"epoch":epoch,"train_loss":history[-1],"train_rows":len(ytrain),"test_rows":len(ytest),"parameter_count":parameters,"elapsed_seconds":time.perf_counter()-started}
        if not all(math.isfinite(v) for k,v in metrics.items() if k!="corr") or not (math.isfinite(metrics["corr"]) or math.isnan(metrics["corr"])): raise ValueError("non-finite metric")
        np.savez_compressed(budget/"predictions.npz",predictions=prediction,targets=ytest,**identities); write_json(budget/"metrics.json",metrics); snapshots.append(metrics)
    write_json(out/"sweep_metrics.json",{"snapshots":snapshots}); write_json(out/"training_complete.json",{"complete":True,"config_id":config_id,"budgets":list(EPOCH_BUDGETS),"selection_rule":"report all; no test-selected checkpoint"}); return snapshots


def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config-id",required=True,choices=PRICE_CONFIGS); p.add_argument("--processed-npz",type=Path,default=DEFAULT_PROCESSED_NPZ); p.add_argument("--features-npz",type=Path,default=DEFAULT_FEATURE_NPZ); p.add_argument("--labels-npz",type=Path,default=DEFAULT_PRICE_LABELS); p.add_argument("--run-root",type=Path); p.add_argument("--device",default="auto"); a=p.parse_args(argv)
    try: run(a.config_id,a.processed_npz,a.features_npz,a.labels_npz,a.run_root or run_root(a.config_id),a.device); print(f"Phase 3 price run complete: {a.config_id}"); return 0
    except FileExistsError as exc: print(exc,file=sys.stderr); return 2
    except Exception as exc: print(f"Phase 3 price run failed: {exc}",file=sys.stderr); return 1


if __name__=="__main__": raise SystemExit(main())
