from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tasks.phase2_classification.labels import CLASS_NAMES, identity_hash
from tasks.phase2_classification.metrics import probabilistic_classification_metrics
from tasks.phase2_classification.protocols import PROTOCOLS, class_priors, make_loss, protocol_sample_indices


@dataclass(frozen=True)
class RunConfig:
    model_id: str
    protocol_id: str = "P2"
    epoch_budgets: tuple[int, ...] = (15, 50, 100)
    seed: int = 0
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    logit_adjustment_strength: float = 1.0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.protocol_id not in PROTOCOLS:
            raise ValueError(f"protocol_id must be one of {sorted(PROTOCOLS)}")
        if not self.epoch_budgets or tuple(sorted(set(self.epoch_budgets))) != self.epoch_budgets:
            raise ValueError("epoch_budgets must be positive, unique, and sorted")
        if any(value <= 0 for value in self.epoch_budgets):
            raise ValueError("epoch_budgets must be positive")
        if self.batch_size <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer or batch-size setting")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["epoch_budgets"] = list(self.epoch_budgets)
        result["protocol"] = PROTOCOLS[self.protocol_id].to_dict()
        return result


def parse_budgets(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("expected at least one epoch budget")
    return result


def resolve_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_run_directory(path: str | Path, overwrite: bool) -> Path:
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"run directory is not empty: {out}; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


@torch.no_grad()
def _predict(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size)
    outputs = [model(batch[0].to(device)).cpu().numpy() for batch in loader]
    result = np.concatenate(outputs).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("model produced non-finite logits")
    return result


def _training_diagnostics(logits: np.ndarray, labels: np.ndarray, epoch: int) -> dict[str, object]:
    raw = np.asarray(logits, dtype=np.float64)
    shifted = raw - raw.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(probabilities)):
        raise FloatingPointError("training probabilities are non-finite")
    probability_variance = probabilities.var(axis=0)
    if not np.any(probability_variance > 1e-12):
        raise FloatingPointError("constant-probability numerical failure on training rows")
    predictions = probabilities.argmax(axis=1)
    n_classes = len(CLASS_NAMES)
    predicted_counts = np.bincount(predictions, minlength=n_classes)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(labels, dtype=np.int64), predictions), 1)
    recalls: list[float] = []
    f1s: list[float] = []
    for class_id in range(n_classes):
        tp = int(cm[class_id, class_id])
        fp = int(cm[:, class_id].sum() - tp)
        fn = int(cm[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        recalls.append(recall)
        f1s.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "epoch": int(epoch),
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class_recall": recalls,
        "predicted_class_counts": predicted_counts.tolist(),
        "predicted_class_concentration": float(predicted_counts.max() / len(predictions)),
        "mean_probabilities": probabilities.mean(axis=0).tolist(),
        "mean_max_probability": float(probabilities.max(axis=1).mean()),
        "logit_variance": raw.var(axis=0).tolist(),
        "probability_variance": probability_variance.tolist(),
    }


def _per_contract_metrics(logits: np.ndarray, labels: np.ndarray, contract_ids: np.ndarray) -> dict[str, object]:
    result = {}
    for contract_id in np.unique(contract_ids):
        mask = contract_ids == contract_id
        metrics, _ = probabilistic_classification_metrics(logits[mask], labels[mask], CLASS_NAMES.tolist())
        result[str(int(contract_id))] = metrics
    return result


def _summary(run_root: Path, sweep: list[dict[str, object]], config: RunConfig) -> None:
    lines = [
        f"# Phase 2 classification: {config.model_id} / {config.protocol_id}", "",
        "Every predeclared epoch snapshot is reported; no test-selected checkpoint is designated.", "",
        "| epoch | accuracy | macro-F1 | balanced accuracy | ROC-AUC | PR-AUC | train loss |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sweep:
        lines.append(
            f"| {row['epoch']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['balanced_accuracy']:.6f} | {row['macro_roc_auc']:.6f} | "
            f"{row['macro_average_precision']:.6f} | {row['train_loss']:.6f} |"
        )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_classification_experiment(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    identities: Mapping[str, np.ndarray],
    model_factory: Callable[[], nn.Module],
    model_spec: Mapping[str, object],
    run_root: str | Path,
    config: RunConfig,
    dataset_manifest: Mapping[str, object],
    scaler: Mapping[str, np.ndarray] | None = None,
) -> list[dict[str, object]]:
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("feature/label row counts differ")
    for split, y in (("train", y_train), ("test", y_test)):
        for name in ("indices", "contract_ids", "window_starts", "timestamps_ns"):
            if len(identities[f"{split}_{name}"]) != len(y):
                raise ValueError(f"{split}_{name} length differs from labels")
    out = Path(run_root)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = resolve_device(config.device)
    priors = class_priors(y_train)
    sample_positions, sampler_audit = protocol_sample_indices(y_train, config.protocol_id, config.seed)
    np.savez_compressed(
        out / "sampling_indices.npz", positions=sample_positions,
        source_row_indices=np.asarray(identities["train_indices"])[sample_positions],
    )
    _write_json(out / "sampling_manifest.json", {**sampler_audit, "class_priors": priors.tolist()})
    _write_json(out / "config.json", {**config.to_dict(), "model_spec": dict(model_spec)})
    manifest = {
        **dict(dataset_manifest),
        "model_id": config.model_id,
        "protocol_id": config.protocol_id,
        "train_sample_count_before_sampling": int(len(y_train)),
        "train_draw_count": int(len(sample_positions)),
        "test_sample_count": int(len(y_test)),
        "train_identity_hash": identity_hash(
            identities["train_indices"], identities["train_contract_ids"], identities["train_window_starts"]
        ),
        "test_identity_hash": identity_hash(
            identities["test_indices"], identities["test_contract_ids"], identities["test_window_starts"]
        ),
        "test_distribution_untouched": True,
    }
    _write_json(out / "dataset_manifest.json", manifest)
    if scaler:
        np.savez(out / "feature_standardizer.npz", **scaler)

    train_dataset = TensorDataset(
        torch.tensor(X_train[sample_positions], dtype=torch.float32),
        torch.tensor(y_train[sample_positions], dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = make_loss(config.protocol_id, priors, config.logit_adjustment_strength).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    history: list[float] = []
    training_diagnostics: list[dict[str, object]] = []
    checkpoints: list[Path] = []
    checkpoint_training_seconds: dict[int, float] = {}
    started = time.perf_counter()
    for epoch in range(1, max(config.epoch_budgets) + 1):
        model.train()
        loss_total = 0.0
        count = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad))
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            optimizer.step()
            loss_total += float(loss.item()) * len(y_batch)
            count += len(y_batch)
        history.append(loss_total / max(count, 1))
        train_logits = _predict(model, X_train, config.batch_size, device)
        training_diagnostics.append(_training_diagnostics(train_logits, y_train, epoch))
        print(f"phase2 classification epoch={epoch} loss={history[-1]:.8f}", flush=True)
        if epoch in config.epoch_budgets:
            budget_dir = out / f"e{epoch}"
            budget_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = budget_dir / "checkpoint.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "completed_epoch": epoch, "config": config.to_dict(), "model_spec": dict(model_spec),
                    "parameter_count": parameter_count,
                }, checkpoint,
            )
            np.savez(
                budget_dir / "history.npz", train_loss=np.asarray(history, dtype=np.float32),
                epochs=np.arange(1, epoch + 1, dtype=np.int32), seed=np.asarray(config.seed, dtype=np.int32),
            )
            checkpoints.append(checkpoint)
            checkpoint_training_seconds[epoch] = time.perf_counter() - started

    training_seconds = time.perf_counter() - started
    _write_json(out / "training_diagnostics.json", training_diagnostics)
    sweep: list[dict[str, object]] = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        evaluated_model = model_factory().to(device)
        evaluated_model.load_state_dict(checkpoint["model_state_dict"])
        inference_started = time.perf_counter()
        logits = _predict(evaluated_model, X_test, config.batch_size, device)
        inference_seconds = time.perf_counter() - inference_started
        metrics, arrays = probabilistic_classification_metrics(logits, y_test, CLASS_NAMES.tolist())
        budget_dir = checkpoint_path.parent
        np.savez_compressed(
            budget_dir / "predictions.npz", **arrays,
            row_indices=np.asarray(identities["test_indices"], dtype=np.int64),
            contract_ids=np.asarray(identities["test_contract_ids"], dtype=np.int32),
            window_starts=np.asarray(identities["test_window_starts"], dtype=np.int64),
            timestamps_ns=np.asarray(identities["test_timestamps_ns"], dtype=np.int64),
        )
        np.savez_compressed(budget_dir / "curves.npz", **{k: v for k, v in arrays.items() if "_roc_" in k or "_pr_" in k})
        with np.load(budget_dir / "history.npz") as history_file:
            final_loss = float(history_file["train_loss"][-1])
        row = {
            **metrics, "epoch": int(checkpoint["completed_epoch"]), "seed": config.seed,
            "model_id": config.model_id, "protocol_id": config.protocol_id,
            "candidate_protocol": PROTOCOLS[config.protocol_id].candidate,
            "train_loss": final_loss, "parameter_count": parameter_count,
            "training_seconds_to_checkpoint": checkpoint_training_seconds[int(checkpoint["completed_epoch"])],
            "training_seconds_total": training_seconds,
            "inference_seconds": inference_seconds,
            "inference_rows_per_second": float(len(y_test) / max(inference_seconds, 1e-12)),
        }
        _write_json(budget_dir / "metrics.json", row)
        _write_json(
            budget_dir / "per_contract_metrics.json",
            _per_contract_metrics(logits, y_test, np.asarray(identities["test_contract_ids"])),
        )
        with np.load(budget_dir / "predictions.npz", allow_pickle=False) as replay:
            replay_ok = (
                np.array_equal(replay["targets"], y_test)
                and np.array_equal(replay["row_indices"], identities["test_indices"])
                and np.array_equal(replay["contract_ids"], identities["test_contract_ids"])
                and np.array_equal(replay["window_starts"], identities["test_window_starts"])
                and np.array_equal(replay["timestamps_ns"], identities["test_timestamps_ns"])
            )
        if not replay_ok:
            raise RuntimeError(f"saved prediction replay verification failed: {budget_dir}")
        _write_json(budget_dir / "replay.json", {"verified": True})
        sweep.append(row)
    _write_json(out / "sweep_metrics.json", sweep)
    _summary(out, sweep, config)
    return sweep
