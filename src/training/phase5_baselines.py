"""Matched raw-sequence baselines for the frozen Phase 5 task matrix."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baselines.mlp_baseline.mlp_model import RawOHLCVMLP
from data_processing.phase5_walks import CLASS_NAMES, validate_phase5_bundle_files
from features.phase5_features import IDENTITY_FIELDS
from tasks.phase2_classification.metrics import probabilistic_classification_metrics
from tasks.phase2_classification.protocols import LogitAdjustedCrossEntropy, class_priors
from tasks.price_prediction import PriceRegressor
from tasks.trend_classification import TrendClassifier
from training.phase5_absolute_price import price_breakdowns, relative_skill
from training.phase5_downstream import (
    _atomic_savez,
    environment_manifest,
    predict,
    regression_breakdowns,
    regression_metrics,
    resolve_device,
    set_seed,
)
from training.phase5_encoder import sha256_file, write_json


BASELINES = ("raw_ohlcv_mlp", "raw_ohlcv_lstm")
TASKS = ("regression_h2", "classification_h2", "absolute_price_h8")
SNAPSHOT_EPOCHS = (5, 15, 50)
CPU_REPLAY_TOLERANCE = {
    "raw_ohlcv_mlp": 2e-5,
    # cuDNN and the CPU LSTM backend use different floating-point reduction
    # paths whose small per-step differences accumulate through three recurrent
    # layers.  The trained checkpoint still replays bit-for-bit on CUDA; this
    # wider tolerance is only for the independent CPU replay.  For the scaled
    # movement task, 2e-3 here is 2e-5 in raw probability-change units.
    "raw_ohlcv_lstm": 2e-3,
}


@dataclass(frozen=True)
class Phase5BaselineConfig:
    baseline: str
    task: str
    walk: int
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    head_hidden_dim: int = 128
    regression_target_multiplier: float = 100.0
    logit_adjustment_strength: float = 1.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("Phase 5 baselines require 50 epochs and 5/15/50 snapshots")
        if self.seed != 0 or self.batch_size != 512 or self.learning_rate != 1e-4:
            raise ValueError("Phase 5 baselines are frozen to seed0/batch512/lr1e-4")
        if self.weight_decay != 0.0 or self.head_hidden_dim != 128:
            raise ValueError("Phase 5 baselines are frozen to weight_decay=0 and head_hidden_dim=128")
        if self.regression_target_multiplier != 100.0:
            raise ValueError("movement regression requires the fixed 100 * delta unit")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot_epochs"] = list(self.snapshot_epochs)
        return payload


class RawOHLCVLSTM(nn.Module):
    """Three-layer multivariate adaptation of the declared stacked-LSTM benchmark."""

    output_dim = 20

    def __init__(self, n_features: int = 5) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(50, 30, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.lstm3 = nn.LSTM(30, self.output_dim, batch_first=True)
        self.dropout3 = nn.Dropout(0.05)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values, _ = self.lstm1(values)
        values = self.dropout1(values)
        values, _ = self.lstm2(values)
        values = self.dropout2(values)
        values, _ = self.lstm3(values)
        return self.dropout3(values[:, -1, :])


class Phase5RawBaseline(nn.Module):
    def __init__(self, config: Phase5BaselineConfig, seq_len: int = 64, n_features: int = 5) -> None:
        super().__init__()
        if config.baseline == "raw_ohlcv_mlp":
            self.encoder = RawOHLCVMLP(
                seq_len=seq_len,
                n_features=n_features,
                hidden_dims=[512, 512, 256, 256, 128],
                output_dim=128,
                dropout=0.1,
            )
            output_dim = self.encoder.output_dim
        else:
            self.encoder = RawOHLCVLSTM(n_features=n_features)
            output_dim = self.encoder.output_dim
        if config.task == "classification_h2":
            self.head = TrendClassifier(output_dim, config.head_hidden_dim, n_classes=3)
            self.output_transform: nn.Module = nn.Identity()
        else:
            self.head = PriceRegressor(output_dim, config.head_hidden_dim)
            self.output_transform = nn.Sigmoid() if config.task == "absolute_price_h8" else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output_transform(self.head(self.encoder(values)))


def model_spec(config: Phase5BaselineConfig) -> dict[str, Any]:
    encoder = (
        {
            "type": "flattened_raw_ohlcv_mlp",
            "hidden_layers": [512, 512, 256, 256, 128],
            "embedding_dim": 128,
            "activation": "ReLU",
            "dropout": 0.1,
        }
        if config.baseline == "raw_ohlcv_mlp"
        else {
            "type": "three_layer_raw_ohlcv_lstm",
            "hidden_layers": [50, 30, 20],
            "embedding_dim": 20,
            "dropout": [0.2, 0.1, 0.05],
        }
    )
    return {
        "input": "stored train/test sequences [N,64,5]",
        "encoder": encoder,
        "head": {
            "hidden_layers": [128, 64],
            "activation": "GELU",
            "dropout": 0.2 if config.task == "classification_h2" else 0.1,
            "output_dim": 3 if config.task == "classification_h2" else 1,
            "output_transform": "sigmoid" if config.task == "absolute_price_h8" else "identity",
        },
        "explicit_current_price_skip": False,
    }


def _metadata(source: Mapping[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    fields = (
        *IDENTITY_FIELDS,
        "target_date_ns",
        "current_close",
        "target_close",
        "context_imputed_rows",
        "lifecycle_fraction",
        "lifecycle_stage",
        "categories",
        "event_families",
        "selection_ranks",
    )
    return {field: np.asarray(source[f"{split}_{field}"]) for field in fields}


def load_baseline_data(dataset_path: Path, task: str) -> dict[str, Any]:
    validate_phase5_bundle_files(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as source:
        data: dict[str, Any] = {
            "X_train": np.asarray(source["train_sequences"], dtype=np.float32),
            "X_test": np.asarray(source["test_sequences"], dtype=np.float32),
            "train_metadata": _metadata(source, "train"),
            "test_metadata": _metadata(source, "test"),
        }
        if task == "regression_h2":
            data["y_train"] = np.asarray(source["train_regression_labels"], dtype=np.float64)
            data["y_test"] = np.asarray(source["test_regression_labels"], dtype=np.float64)
        elif task == "classification_h2":
            data["y_train"] = np.asarray(source["train_classification_labels"], dtype=np.int64)
            data["y_test"] = np.asarray(source["test_classification_labels"], dtype=np.int64)
        elif task == "absolute_price_h8":
            data["y_train"] = np.asarray(source["train_target_close"], dtype=np.float64)
            data["y_test"] = np.asarray(source["test_target_close"], dtype=np.float64)
        else:
            raise ValueError(f"unknown task: {task}")
    for split in ("train", "test"):
        x = data[f"X_{split}"]
        y = data[f"y_{split}"]
        if x.ndim != 3 or x.shape[1:] != (64, 5) or len(x) != len(y):
            raise ValueError(f"unaligned Phase 5 baseline {split} data")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"non-finite Phase 5 baseline {split} data")
    if task == "absolute_price_h8" and (
        np.any((data["y_train"] < 0.0) | (data["y_train"] > 1.0))
        or np.any((data["y_test"] < 0.0) | (data["y_test"] > 1.0))
    ):
        raise ValueError("absolute-price targets must lie in [0,1]")
    return data


def smoke_test_baselines() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for baseline in BASELINES:
        for task in TASKS:
            config = Phase5BaselineConfig(baseline=baseline, task=task, walk=1, device="cpu")
            model = Phase5RawBaseline(config)
            x = torch.randn(4, 64, 5)
            output = model(x)
            expected = (4, 3) if task == "classification_h2" else (4, 1)
            if tuple(output.shape) != expected or not torch.isfinite(output).all():
                raise RuntimeError(f"baseline smoke test failed for {baseline}/{task}")
            if task == "absolute_price_h8" and not torch.all((output >= 0.0) & (output <= 1.0)):
                raise RuntimeError("absolute-price output is outside [0,1]")
            results[f"{baseline}/{task}"] = {"output_shape": list(output.shape)}
    return {"valid": True, "models": results}


def _save_history(path: Path, losses: list[float], seconds: list[float]) -> None:
    _atomic_savez(path, {
        "epochs": np.arange(1, len(losses) + 1, dtype=np.int32),
        "train_loss": np.asarray(losses, dtype=np.float64),
        "epoch_seconds": np.asarray(seconds, dtype=np.float64),
    })


def _evaluate_snapshot(
    model: nn.Module,
    data: Mapping[str, Any],
    config: Phase5BaselineConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    output = predict(model, data["X_test"], config.batch_size, device)
    target = np.asarray(data["y_test"])
    metadata = data["test_metadata"]
    if config.task == "regression_h2":
        prediction_pp = output.reshape(-1).astype(np.float64)
        prediction = prediction_pp / config.regression_target_multiplier
        metrics, per_contract = regression_breakdowns(prediction, target, metadata)
        zero, zero_per_contract = regression_breakdowns(np.zeros_like(target), target, metadata)
        arrays = {
            "prediction_probability_points": prediction_pp,
            "prediction_delta": prediction,
            "target_probability_points": target * config.regression_target_multiplier,
            "target_delta": target,
            **metadata,
        }
        payload = {
            "baseline": metrics,
            "exact_zero_reference": zero,
            "per_contract": per_contract,
            "zero_reference_per_contract": zero_per_contract,
        }
        summary = {key: metrics["overall"][key] for key in ("mae", "rmse", "mse", "pearson", "spearman", "sign_agreement")}
    elif config.task == "classification_h2":
        metrics, arrays = probabilistic_classification_metrics(output, target, CLASS_NAMES)
        arrays = {**arrays, **metadata}
        stable_logits = np.full((len(target), 3), -30.0, dtype=np.float32)
        stable_logits[:, 1] = 0.0
        stable_metrics, _ = probabilistic_classification_metrics(
            stable_logits, target, CLASS_NAMES
        )
        priors = class_priors(np.asarray(data["y_train"]), n_classes=3)
        prior_logits = np.broadcast_to(
            np.log(np.asarray(priors, dtype=np.float64)), (len(target), 3)
        ).copy()
        prior_metrics, _ = probabilistic_classification_metrics(
            prior_logits, target, CLASS_NAMES
        )
        payload = {
            "baseline": metrics,
            "always_stable_reference": stable_metrics,
            "repeated_training_prior_reference": prior_metrics,
            "class_names": list(CLASS_NAMES),
        }
        summary = {key: metrics[key] for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "macro_roc_auc", "macro_average_precision", "nll", "multiclass_brier")}
    else:
        prediction = output.reshape(-1).astype(np.float64)
        price, per_contract = price_breakdowns(prediction, target, metadata)
        current = np.asarray(metadata["current_close"], dtype=np.float64)
        persistence, persistence_per_contract = price_breakdowns(current, target, metadata)
        predicted_delta = prediction - current
        target_delta = target - current
        raw_sequences = np.asarray(data["X_test"], dtype=np.float64)
        last_hour_reversal = -(raw_sequences[:, -1, 3] - raw_sequences[:, -2, 3])
        movement = regression_metrics(predicted_delta, target_delta)
        arrays = {
            "prediction_future_price": prediction,
            "target_future_price": target,
            "current_close": current,
            "prediction_delta_h8": predicted_delta,
            "target_delta_h8": target_delta,
            "last_hour_reversal": last_hour_reversal,
            **metadata,
        }
        payload = {
            "price": price,
            "persistence_reference": persistence,
            "persistence_relative_skill": relative_skill(price["overall"], persistence["overall"]),
            "implied_movement": movement,
            "last_hour_reversal": regression_metrics(last_hour_reversal, target_delta),
            "per_contract": per_contract,
            "persistence_per_contract": persistence_per_contract,
        }
        summary = {
            "mae": price["overall"]["mae"],
            "rmse": price["overall"]["rmse"],
            "mse": price["overall"]["mse"],
            "pearson": price["overall"]["pearson"],
            "spearman": price["overall"]["spearman"],
            "implied_delta_pearson": movement["pearson"],
            "implied_delta_spearman": movement["spearman"],
        }
    return payload, arrays, summary


def run_baseline_training(
    dataset_path: Path,
    run_root: Path,
    config: Phase5BaselineConfig,
) -> list[dict[str, Any]]:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite baseline run: {run_root}")
    data = load_baseline_data(dataset_path, config.task)
    device = resolve_device(config.device)
    set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    y_train = np.asarray(data["y_train"])
    if config.task == "regression_h2":
        target = torch.from_numpy((y_train * config.regression_target_multiplier).astype(np.float32).reshape(-1, 1))
        criterion: nn.Module = nn.MSELoss()
        priors = None
    elif config.task == "classification_h2":
        target = torch.from_numpy(y_train.astype(np.int64))
        priors = class_priors(y_train, n_classes=3)
        criterion = LogitAdjustedCrossEntropy(priors, strength=config.logit_adjustment_strength)
    else:
        target = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
        criterion = nn.MSELoss()
        priors = None
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data["X_train"]), target),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = Phase5RawBaseline(config).to(device)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "config.json", config.to_dict())
    write_json(run_root / "environment.json", environment_manifest(device))
    write_json(run_root / "architecture_manifest.json", {**model_spec(config), "parameter_count": parameter_count})
    source_manifest = json.loads(Path(f"{dataset_path}.manifest.json").read_text(encoding="utf-8"))
    write_json(run_root / "dataset_manifest.json", {
        "phase": 5,
        "purpose": "matched_raw_sequence_baseline",
        "baseline": config.baseline,
        "task": config.task,
        "walk": config.walk,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "train_identity_hash": source_manifest["identity_hashes"]["train"],
        "test_identity_hash": source_manifest["identity_hashes"]["test"],
        "input_array": "train_sequences/test_sequences",
        "input_preprocessing": source_manifest["preprocessing"],
        "train_rows": len(data["X_train"]),
        "test_rows": len(data["X_test"]),
        "evaluation_used_for_fitting_or_selection": False,
        "checkpoint_selection": "epoch 50 predeclared",
    })
    if priors is not None:
        write_json(run_root / "imbalance_manifest.json", {
            "sampling": "natural rows",
            "loss": "logit-adjusted cross-entropy",
            "strength": config.logit_adjustment_strength,
            "training_class_counts": np.bincount(y_train, minlength=3).tolist(),
            "training_class_priors": priors.tolist(),
            "evaluation_distribution_untouched": True,
        })
    losses: list[float] = []
    seconds: list[float] = []
    checkpoints: list[Path] = []
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite baseline loss at epoch {epoch}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"non-finite baseline gradient at epoch {epoch}")
            optimizer.step()
            total += float(loss.detach()) * len(x_batch)
            count += len(x_batch)
        losses.append(total / count)
        seconds.append(time.perf_counter() - started)
        print(f"baseline={config.baseline} task={config.task} walk={config.walk} epoch={epoch} loss={losses[-1]:.8f}", flush=True)
        if epoch in config.snapshot_epochs:
            snapshot = run_root / f"e{epoch}"
            snapshot.mkdir()
            checkpoint = snapshot / "checkpoint.pth"
            torch.save({
                "phase": 5,
                "purpose": "matched_raw_sequence_baseline",
                "baseline": config.baseline,
                "task": config.task,
                "walk": config.walk,
                "completed_epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.to_dict(),
                "model_spec": model_spec(config),
                "parameter_count": parameter_count,
                "dataset_sha256": sha256_file(dataset_path),
            }, checkpoint)
            _save_history(snapshot / "history.npz", losses, seconds)
            checkpoints.append(checkpoint)
    snapshots: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        evaluated = Phase5RawBaseline(config).to(device)
        evaluated.load_state_dict(checkpoint["model_state_dict"], strict=True)
        inference_started = time.perf_counter()
        payload, arrays, summary = _evaluate_snapshot(evaluated, data, config, device)
        inference_seconds = time.perf_counter() - inference_started
        snapshot = checkpoint_path.parent
        _atomic_savez(snapshot / "predictions.npz", arrays)
        row = {
            "phase": 5,
            "baseline": config.baseline,
            "task": config.task,
            "walk": config.walk,
            "epoch": int(checkpoint["completed_epoch"]),
            "seed": config.seed,
            "train_loss": losses[int(checkpoint["completed_epoch"]) - 1],
            "parameter_count": parameter_count,
            "elapsed_training_seconds": float(sum(seconds[: int(checkpoint["completed_epoch"])])),
            "inference_seconds": inference_seconds,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(snapshot / "predictions.npz"),
            **summary,
        }
        write_json(snapshot / "metrics.json", {**row, **payload})
        snapshots.append(row)
    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    write_json(run_root / "training_complete.json", {
        "complete": True,
        "phase": 5,
        "baseline": config.baseline,
        "task": config.task,
        "walk": config.walk,
        "snapshots": list(config.snapshot_epochs),
        "principal_checkpoint": str(run_root / "e50" / "checkpoint.pth"),
        "selection_rule": "epoch 50 predeclared; evaluation metrics did not select a checkpoint",
    })
    write_baseline_summary(run_root)
    return snapshots


def write_baseline_summary(run_root: Path) -> None:
    config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    rows = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))["snapshots"]
    lines = [
        f"# Phase 5 {config['baseline']} {config['task']} Walk {config['walk']}",
        "",
        "One uninterrupted seed-0 trajectory. Epoch 50 is predeclared; evaluation did not select a checkpoint.",
        "",
        "| Epoch | Train loss | Principal metrics |",
        "|---:|---:|---|",
    ]
    excluded = {"phase", "baseline", "task", "walk", "epoch", "seed", "train_loss", "parameter_count", "elapsed_training_seconds", "inference_seconds", "peak_cuda_memory_bytes", "checkpoint_path", "checkpoint_sha256", "predictions_sha256"}
    for row in rows:
        metrics = ", ".join(f"{key}={value:.6f}" for key, value in row.items() if key not in excluded and isinstance(value, (int, float)))
        lines.append(f"| {row['epoch']} | {row['train_loss']:.8f} | {metrics} |")
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_baseline_run(
    dataset_path: Path,
    run_root: Path,
    *,
    replay_tolerance: float | None = None,
) -> dict[str, Any]:
    required = {"config.json", "environment.json", "architecture_manifest.json", "dataset_manifest.json", "sweep_metrics.json", "training_complete.json", "summary.md"}
    missing = sorted(name for name in required if not (run_root / name).is_file())
    if missing:
        raise ValueError(f"baseline run is missing artifacts: {missing}")
    payload = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
    config = Phase5BaselineConfig(**payload)
    effective_replay_tolerance = (
        CPU_REPLAY_TOLERANCE[config.baseline]
        if replay_tolerance is None
        else replay_tolerance
    )
    data = load_baseline_data(dataset_path, config.task)
    manifest = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("baseline source dataset hash mismatch")
    if manifest.get("evaluation_used_for_fitting_or_selection") is not False:
        raise ValueError("baseline evaluation isolation invariant is missing")
    rows = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))["snapshots"]
    if [int(row["epoch"]) for row in rows] != list(SNAPSHOT_EPOCHS):
        raise ValueError("baseline snapshot matrix is incomplete")
    metadata = data["test_metadata"]
    for row in rows:
        epoch = int(row["epoch"])
        snapshot = run_root / f"e{epoch}"
        paths = [snapshot / name for name in ("checkpoint.pth", "history.npz", "predictions.npz", "metrics.json")]
        if not all(path.is_file() for path in paths):
            raise ValueError(f"baseline e{epoch} artifacts are incomplete")
        metrics = json.loads(paths[3].read_text(encoding="utf-8"))
        if sha256_file(paths[0]) != metrics.get("checkpoint_sha256") or sha256_file(paths[2]) != metrics.get("predictions_sha256"):
            raise ValueError(f"baseline e{epoch} artifact hash mismatch")
        checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
        expected = {"phase": 5, "purpose": "matched_raw_sequence_baseline", "baseline": config.baseline, "task": config.task, "walk": config.walk, "completed_epoch": epoch, "dataset_sha256": sha256_file(dataset_path)}
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError(f"baseline e{epoch} checkpoint metadata mismatch")
        model = Phase5RawBaseline(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        replay = predict(model, data["X_test"], config.batch_size, torch.device("cpu"))
        with np.load(paths[2], allow_pickle=False) as saved:
            for field, expected_values in metadata.items():
                if not np.array_equal(saved[field], expected_values):
                    raise ValueError(f"baseline e{epoch} identity mismatch: {field}")
            if config.task == "regression_h2":
                stored = saved["prediction_probability_points"].reshape(-1, 1)
            elif config.task == "classification_h2":
                stored = saved["logits"]
            else:
                stored = saved["prediction_future_price"].reshape(-1, 1)
            if not np.allclose(
                replay,
                stored,
                rtol=effective_replay_tolerance,
                atol=effective_replay_tolerance,
            ):
                raise ValueError(f"baseline e{epoch} CPU prediction replay mismatch")
        with np.load(paths[1], allow_pickle=False) as history:
            if len(history["epochs"]) != epoch or int(history["epochs"][-1]) != epoch:
                raise ValueError(f"baseline e{epoch} history mismatch")
    complete = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("snapshots") != list(SNAPSHOT_EPOCHS):
        raise ValueError("baseline completion marker is invalid")
    principal = rows[-1]
    return {
        "valid": True,
        "baseline": config.baseline,
        "task": config.task,
        "walk": config.walk,
        "snapshots": list(SNAPSHOT_EPOCHS),
        "cpu_replay_tolerance": effective_replay_tolerance,
        "epoch50_checkpoint_sha256": principal["checkpoint_sha256"],
        "epoch50_predictions_sha256": principal["predictions_sha256"],
    }
