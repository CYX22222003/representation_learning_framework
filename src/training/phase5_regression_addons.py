"""Phase 5 exploratory eight-hour and log-return additional regression tasks."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_processing.phase5_walks import validate_phase5_bundle_files
from features.phase5_features import IDENTITY_FIELDS, load_concat_features, validate_phase5_feature_bundle
from training.phase5_downstream import (
    _atomic_savez,
    apply_feature_standardizer,
    build_head,
    environment_manifest,
    predict,
    regression_metrics,
    resolve_device,
    set_seed,
    validate_feature_standardizer,
)
from training.phase5_encoder import sha256_file, write_json


TASKS = ("raw_delta_h8", "log_return_h2")
SNAPSHOT_EPOCHS = (5, 15, 50)


@dataclass(frozen=True)
class RegressionAddonConfig:
    task: str
    walk: int
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("additional regression tasks require 50 epochs and 5/15/50 snapshots")
        if self.seed != 0 or self.batch_size != 512 or self.learning_rate != 1e-4:
            raise ValueError("additional regression tasks are frozen to seed0/batch512/lr1e-4")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["snapshot_epochs"] = list(self.snapshot_epochs)
        return result


def derive_scientific_target(
    task: str, current_close: np.ndarray, target_close: np.ndarray
) -> np.ndarray:
    current = np.asarray(current_close, dtype=np.float64)
    future = np.asarray(target_close, dtype=np.float64)
    if current.shape != future.shape or not np.isfinite(current).all() or not np.isfinite(future).all():
        raise ValueError("current/target prices must be aligned and finite")
    if task == "raw_delta_h8":
        return future - current
    if task == "log_return_h2":
        if np.any(current <= 0.0) or np.any(future <= 0.0):
            raise ValueError("ordinary log return requires strictly positive current and target prices")
        return np.log(future / current)
    raise ValueError(f"unknown regression add-on task: {task}")


def fit_target_transform(task: str, train_target: np.ndarray) -> dict[str, np.ndarray]:
    target = np.asarray(train_target, dtype=np.float64)
    if not np.isfinite(target).all() or len(target) == 0:
        raise ValueError("target transform requires non-empty finite training targets")
    if task == "raw_delta_h8":
        center, scale, method = 0.0, 0.01, "fixed_probability_points"
    elif task == "log_return_h2":
        center = float(target.mean())
        raw_std = float(target.std(ddof=0))
        scale = raw_std if raw_std >= 1e-8 else 1.0
        method = "training_mean_std"
    else:
        raise ValueError(f"unknown regression add-on task: {task}")
    return {
        "center": np.asarray(center, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
        "method": np.asarray(method),
        "uses_evaluation": np.asarray(False),
    }


def apply_target_transform(target: np.ndarray, transform: Mapping[str, np.ndarray]) -> np.ndarray:
    return (
        (np.asarray(target, dtype=np.float64) - float(transform["center"]))
        / float(transform["scale"])
    ).astype(np.float32)


def invert_target_transform(values: np.ndarray, transform: Mapping[str, np.ndarray]) -> np.ndarray:
    return (
        np.asarray(values, dtype=np.float64) * float(transform["scale"])
        + float(transform["center"])
    )


def reconstruct_probability(
    task: str, current_close: np.ndarray, predicted_target: np.ndarray
) -> np.ndarray:
    current = np.asarray(current_close, dtype=np.float64)
    prediction = np.asarray(predicted_target, dtype=np.float64)
    if task == "raw_delta_h8":
        return current + prediction
    if task == "log_return_h2":
        return current * np.exp(prediction)
    raise ValueError(f"unknown regression add-on task: {task}")


def load_regression_addon_data(
    dataset_path: Path, feature_path: Path, scaler_path: Path, task: str
) -> dict[str, Any]:
    validation = validate_phase5_bundle_files(dataset_path)
    validate_phase5_feature_bundle(feature_path, dataset_path=dataset_path)
    validate_feature_standardizer(feature_path, scaler_path, dataset_path=dataset_path)
    train_features, test_features = load_concat_features(feature_path)
    with np.load(scaler_path, allow_pickle=False) as scaler:
        X_train = apply_feature_standardizer(train_features, scaler)
        X_test = apply_feature_standardizer(test_features, scaler)
    with np.load(dataset_path, allow_pickle=False) as source:
        result: dict[str, Any] = {"X_train": X_train, "X_test": X_test}
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
        for split in ("train", "test"):
            for field in fields:
                result[f"{split}_{field}"] = np.asarray(source[f"{split}_{field}"])
            result[f"y_{split}"] = derive_scientific_target(
                task, source[f"{split}_current_close"], source[f"{split}_target_close"]
            )
    expected_horizon = 8 if task == "raw_delta_h8" else 2
    manifest = json.loads(Path(f"{dataset_path}.manifest.json").read_text(encoding="utf-8"))
    if int(manifest["horizon_hours"]) != expected_horizon:
        raise ValueError("regression add-on task and data horizon disagree")
    if len(X_train) != len(result["y_train"]) or len(X_test) != len(result["y_test"]):
        raise ValueError("regression add-on features and targets are not aligned")
    result["walk"] = int(validation["walk"])
    return result


def _masked_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any] | None:
    selected = np.asarray(mask, dtype=bool)
    return regression_metrics(prediction[selected], target[selected]) if np.any(selected) else None


def regression_addon_breakdowns(
    task: str,
    prediction: np.ndarray,
    target: np.ndarray,
    metadata: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pred = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    current = np.asarray(metadata["current_close"], dtype=np.float64)
    groups: dict[str, Any] = {
        "overall": regression_metrics(pred, y),
        "exact_zero": _masked_metrics(pred, y, y == 0.0),
        "non_zero": _masked_metrics(pred, y, y != 0.0),
        "context_observed_only": _masked_metrics(
            pred, y, np.asarray(metadata["context_imputed_rows"]) == 0
        ),
        "context_has_imputation": _masked_metrics(
            pred, y, np.asarray(metadata["context_imputed_rows"]) > 0
        ),
    }
    if task == "raw_delta_h8":
        groups["within_0.001"] = _masked_metrics(pred, y, np.abs(y) <= 0.001)
        groups["beyond_0.001"] = _masked_metrics(pred, y, np.abs(y) > 0.001)
    price_bands = {
        "p_le_0.01": current <= 0.01,
        "p_0.01_to_0.10": (current > 0.01) & (current <= 0.10),
        "p_0.10_to_0.90": (current > 0.10) & (current < 0.90),
        "p_ge_0.90": current >= 0.90,
    }
    groups["starting_price_bands"] = {
        name: _masked_metrics(pred, y, mask) for name, mask in price_bands.items()
    }
    stage = np.asarray(metadata["lifecycle_stage"])
    groups["lifecycle"] = {
        name: _masked_metrics(pred, y, stage == code)
        for code, name in ((-1, "unknown"), (0, "early"), (1, "middle"), (2, "late"))
    }
    per_contract: dict[str, Any] = {}
    condition_ids = np.asarray(metadata["condition_ids"])
    for condition_id in np.unique(condition_ids):
        mask = condition_ids == condition_id
        per_contract[str(condition_id)] = regression_metrics(pred[mask], y[mask])
    metric_names = ("mae", "mse", "rmse", "pearson", "spearman", "sign_agreement")
    groups["contract_macro"] = {
        metric: float(np.mean([row[metric] for row in per_contract.values() if math.isfinite(float(row[metric]))]))
        for metric in metric_names
        if any(math.isfinite(float(row[metric])) for row in per_contract.values())
    }
    groups["contract_count"] = len(per_contract)
    return groups, per_contract


def _metadata(data: Mapping[str, Any], split: str = "test") -> dict[str, np.ndarray]:
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
    return {field: np.asarray(data[f"{split}_{field}"]) for field in fields}


def run_regression_addon(
    dataset_path: Path,
    feature_path: Path,
    scaler_path: Path,
    run_root: Path,
    config: RegressionAddonConfig,
) -> list[dict[str, Any]]:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite regression add-on run: {run_root}")
    data = load_regression_addon_data(dataset_path, feature_path, scaler_path, config.task)
    transform = fit_target_transform(config.task, data["y_train"])
    y_train_optimization = apply_target_transform(data["y_train"], transform)
    device = resolve_device(config.device)
    set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data["X_train"]),
            torch.from_numpy(y_train_optimization.reshape(-1, 1)),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = build_head("regression").to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "config.json", config.to_dict())
    write_json(run_root / "environment.json", environment_manifest(device))
    _atomic_savez(run_root / "target_transform.npz", transform)
    write_json(
        run_root / "dataset_manifest.json",
        {
            "phase": 5,
            "purpose": "additional_regression_task",
            "exploratory_status": "additional post-primary regression task",
            "task": config.task,
            "walk": config.walk,
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "feature_path": str(feature_path),
            "feature_sha256": sha256_file(feature_path),
            "feature_scaler_path": str(scaler_path),
            "feature_scaler_sha256": sha256_file(scaler_path),
            "target_transform_sha256": sha256_file(run_root / "target_transform.npz"),
            "train_rows": int(len(data["X_train"])),
            "test_rows": int(len(data["X_test"])),
            "evaluation_used_for_fitting_or_selection": False,
            "principal_epoch": 50,
        },
    )
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
                raise FloatingPointError(f"non-finite regression add-on loss at epoch {epoch}")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise FloatingPointError(f"non-finite regression add-on gradient at epoch {epoch}")
            optimizer.step()
            total += float(loss.detach()) * len(x_batch)
            count += len(x_batch)
        losses.append(total / count)
        seconds.append(time.perf_counter() - started)
        print(
            f"walk={config.walk} task={config.task} epoch={epoch} "
            f"loss={losses[-1]:.8f} seconds={seconds[-1]:.2f}",
            flush=True,
        )
        if epoch in config.snapshot_epochs:
            snapshot = run_root / f"e{epoch}"
            snapshot.mkdir()
            checkpoint_path = snapshot / "checkpoint.pth"
            torch.save(
                {
                    "phase": 5,
                    "purpose": "additional_regression_task",
                    "task": config.task,
                    "walk": config.walk,
                    "completed_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "target_center": float(transform["center"]),
                    "target_scale": float(transform["scale"]),
                    "dataset_sha256": sha256_file(dataset_path),
                    "feature_sha256": sha256_file(feature_path),
                    "feature_scaler_sha256": sha256_file(scaler_path),
                },
                checkpoint_path,
            )
            _atomic_savez(
                snapshot / "history.npz",
                {
                    "epochs": np.arange(1, epoch + 1, dtype=np.int32),
                    "train_loss": np.asarray(losses, dtype=np.float64),
                    "epoch_seconds": np.asarray(seconds, dtype=np.float64),
                },
            )
            checkpoints.append(checkpoint_path)

    metadata = _metadata(data)
    snapshots: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        evaluated = build_head("regression").to(device)
        evaluated.load_state_dict(checkpoint["model_state_dict"], strict=True)
        inference_started = time.perf_counter()
        optimized_prediction = predict(
            evaluated, data["X_test"], config.batch_size, device
        ).reshape(-1)
        inference_seconds = time.perf_counter() - inference_started
        scientific_prediction = invert_target_transform(optimized_prediction, transform)
        scientific_target = np.asarray(data["y_test"], dtype=np.float64)
        metrics, per_contract = regression_addon_breakdowns(
            config.task, scientific_prediction, scientific_target, metadata
        )
        reference, reference_per_contract = regression_addon_breakdowns(
            config.task, np.zeros_like(scientific_target), scientific_target, metadata
        )
        reconstructed = reconstruct_probability(
            config.task, metadata["current_close"], scientific_prediction
        )
        reconstruction_metrics = regression_metrics(reconstructed, metadata["target_close"])
        reconstruction_metrics.update(
            {
                "below_zero_count": int(np.count_nonzero(reconstructed < 0.0)),
                "above_one_count": int(np.count_nonzero(reconstructed > 1.0)),
                "out_of_bounds_fraction": float(
                    np.mean((reconstructed < 0.0) | (reconstructed > 1.0))
                ),
            }
        )
        snapshot = checkpoint_path.parent
        arrays = {
            "prediction_optimization_unit": optimized_prediction.astype(np.float64),
            "prediction_scientific_target": scientific_prediction,
            "target_scientific": scientific_target,
            "prediction_future_probability": reconstructed,
            **metadata,
        }
        _atomic_savez(snapshot / "predictions.npz", arrays)
        write_json(snapshot / "per_contract_metrics.json", per_contract)
        write_json(snapshot / "zero_reference_per_contract_metrics.json", reference_per_contract)
        overall = metrics["overall"]
        row = {
            "phase": 5,
            "task": config.task,
            "walk": config.walk,
            "epoch": int(checkpoint["completed_epoch"]),
            "seed": config.seed,
            "train_loss": losses[int(checkpoint["completed_epoch"]) - 1],
            "mae": overall["mae"],
            "rmse": overall["rmse"],
            "mse": overall["mse"],
            "pearson": overall["pearson"],
            "spearman": overall["spearman"],
            "sign_agreement": overall["sign_agreement"],
            "reconstructed_probability_mae": reconstruction_metrics["mae"],
            "reconstructed_probability_rmse": reconstruction_metrics["rmse"],
            "inference_seconds": inference_seconds,
            "parameter_count": parameter_count,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(snapshot / "predictions.npz"),
        }
        write_json(
            snapshot / "metrics.json",
            {
                **row,
                "framework": metrics,
                "zero_change_reference": reference,
                "reconstructed_probability": reconstruction_metrics,
            },
        )
        snapshots.append(row)
    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    write_json(
        run_root / "training_complete.json",
        {
            "complete": True,
            "phase": 5,
            "task": config.task,
            "walk": config.walk,
            "snapshots": list(config.snapshot_epochs),
            "principal_epoch": 50,
            "selection_rule": "epoch 50 predeclared; no evaluation-driven selection",
        },
    )
    return snapshots


def validate_regression_addon_run(
    dataset_path: Path,
    feature_path: Path,
    scaler_path: Path,
    run_root: Path,
    *,
    tolerance: float = 2e-5,
) -> dict[str, Any]:
    required = {
        "config.json",
        "environment.json",
        "dataset_manifest.json",
        "target_transform.npz",
        "sweep_metrics.json",
        "training_complete.json",
    }
    missing = sorted(name for name in required if not (run_root / name).is_file())
    if missing:
        raise ValueError(f"regression add-on run is missing artifacts: {missing}")
    payload = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
    config = RegressionAddonConfig(**payload)
    data = load_regression_addon_data(dataset_path, feature_path, scaler_path, config.task)
    manifest = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    expected_hashes = {
        "dataset_sha256": sha256_file(dataset_path),
        "feature_sha256": sha256_file(feature_path),
        "feature_scaler_sha256": sha256_file(scaler_path),
        "target_transform_sha256": sha256_file(run_root / "target_transform.npz"),
    }
    for key, value in expected_hashes.items():
        if manifest.get(key) != value:
            raise ValueError(f"regression add-on manifest {key} mismatch")
    with np.load(run_root / "target_transform.npz", allow_pickle=False) as saved:
        transform = {name: np.asarray(saved[name]) for name in saved.files}
    expected_transform = fit_target_transform(config.task, data["y_train"])
    for key in expected_transform:
        if not np.array_equal(transform[key], expected_transform[key]):
            raise ValueError(f"regression add-on target-transform replay mismatch: {key}")
    rows = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))["snapshots"]
    if [int(row["epoch"]) for row in rows] != list(SNAPSHOT_EPOCHS):
        raise ValueError("regression add-on snapshot matrix is incomplete")
    metadata = _metadata(data)
    for row in rows:
        epoch = int(row["epoch"])
        snapshot = run_root / f"e{epoch}"
        checkpoint_path = snapshot / "checkpoint.pth"
        predictions_path = snapshot / "predictions.npz"
        history_path = snapshot / "history.npz"
        metrics_path = snapshot / "metrics.json"
        if not all(path.is_file() for path in (checkpoint_path, predictions_path, history_path, metrics_path)):
            raise ValueError(f"regression add-on e{epoch} artifacts are incomplete")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if sha256_file(checkpoint_path) != metrics["checkpoint_sha256"]:
            raise ValueError(f"regression add-on e{epoch} checkpoint hash mismatch")
        if sha256_file(predictions_path) != metrics["predictions_sha256"]:
            raise ValueError(f"regression add-on e{epoch} predictions hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_head("regression")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        replay_optimization = predict(
            model, data["X_test"], config.batch_size, torch.device("cpu")
        ).reshape(-1)
        replay_scientific = invert_target_transform(replay_optimization, transform)
        with np.load(predictions_path, allow_pickle=False) as saved:
            if not np.allclose(
                replay_scientific,
                saved["prediction_scientific_target"],
                rtol=tolerance,
                atol=tolerance,
            ):
                raise ValueError(f"regression add-on e{epoch} CPU prediction replay mismatch")
            if not np.array_equal(saved["target_scientific"], data["y_test"]):
                raise ValueError(f"regression add-on e{epoch} target replay mismatch")
            for key, expected in metadata.items():
                if not np.array_equal(saved[key], expected):
                    raise ValueError(f"regression add-on e{epoch} identity mismatch: {key}")
        with np.load(history_path, allow_pickle=False) as history:
            if len(history["epochs"]) != epoch or int(history["epochs"][-1]) != epoch:
                raise ValueError(f"regression add-on e{epoch} history mismatch")
    complete = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("principal_epoch") != 50:
        raise ValueError("regression add-on completion marker is invalid")
    principal = rows[-1]
    return {
        "valid": True,
        "task": config.task,
        "walk": config.walk,
        "snapshots": list(SNAPSHOT_EPOCHS),
        **{key: principal[key] for key in ("mae", "rmse", "pearson", "spearman", "sign_agreement", "reconstructed_probability_mae")},
    }
