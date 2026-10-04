"""Native-width common probes for Phase 6.7 frozen representations."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_processing.phase5_walks import CLASS_NAMES, sha256_arrays, sha256_file
from features.phase5_features import IDENTITY_FIELDS, array_sha256
from features.phase6_7_external_features import (
    METHOD,
    OUTPUT_DIM,
    TASKS,
    load_external_task_features,
    validate_external_feature_store,
)
from tasks.phase2_classification.metrics import probabilistic_classification_metrics
from tasks.phase2_classification.protocols import LogitAdjustedCrossEntropy, class_priors
from tasks.trend_classification import TrendClassifier
from tasks.volatility_prediction import VolatilityRegressor
from training.phase5_absolute_price import AbsolutePriceRegressor, price_breakdowns
from training.phase5_downstream import regression_metrics
from training.phase5_encoder import environment_manifest, resolve_device, set_seed, write_json
from training.phase6_volatility import historical_persistence, volatility_metrics


SNAPSHOT_EPOCHS = (5, 15, 50)
SCHEMA_VERSION = "phase6-7-external-downstream-v1"
METHOD_DIMS = {METHOD: OUTPUT_DIM, "lwa_frozen": 384}


@dataclass(frozen=True)
class Phase67DownstreamConfig:
    task: str
    method: str
    walk: int
    input_dim: int = OUTPUT_DIM
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    hidden_dim: int = 128
    volatility_target_multiplier: float = 10_000.0
    smooth_l1_beta: float = 1.0
    volatility_gradient_clip_norm: float = 5.0
    logit_adjustment_strength: float = 1.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_epochs", tuple(self.snapshot_epochs))
        if self.task not in TASKS or self.method not in METHOD_DIMS:
            raise ValueError("invalid Phase 6.7 downstream task/method")
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.input_dim != METHOD_DIMS[self.method]:
            raise ValueError("external method requires its declared native representation width")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("Phase 6.7 downstream requires 50 epochs and 5/15/50 snapshots")
        if self.seed != 0 or self.batch_size != 512 or self.learning_rate != 1e-4:
            raise ValueError("Phase 6.7 downstream recipe is frozen")
        if self.weight_decay != 0.0 or self.hidden_dim != 128:
            raise ValueError("Phase 6.7 downstream head contract changed")
        if (
            self.volatility_target_multiplier != 10_000.0
            or self.smooth_l1_beta != 1.0
            or self.volatility_gradient_clip_norm != 5.0
        ):
            raise ValueError("Phase 6.7 volatility optimization contract changed")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot_epochs"] = list(self.snapshot_epochs)
        return payload


def fit_external_standardizer(train: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(train, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] <= 0:
        raise ValueError("external standardizer requires a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("external standardizer requires finite values")
    mean = values.mean(axis=0)
    raw_std = values.std(axis=0, ddof=0)
    return {
        "mean": mean,
        "raw_std": raw_std,
        "scale": np.where(raw_std < 1e-8, 1.0, raw_std),
        "clip_low": np.asarray(-10.0),
        "clip_high": np.asarray(10.0),
    }


def apply_external_standardizer(
    values: np.ndarray, scaler: Mapping[str, np.ndarray]
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != len(scaler["mean"]):
        raise ValueError("external feature/scaler width mismatch")
    result = (source - scaler["mean"]) / scaler["scale"]
    result = np.clip(
        result, float(scaler["clip_low"]), float(scaler["clip_high"])
    ).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("external feature standardization produced non-finite values")
    return result


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _config_from_payload(payload: Mapping[str, Any], device: str) -> Phase67DownstreamConfig:
    values = dict(payload)
    values["snapshot_epochs"] = tuple(values["snapshot_epochs"])
    values["device"] = device
    return Phase67DownstreamConfig(**values)


def _load_task_data(
    dataset_path: Path, feature_path: Path, config: Phase67DownstreamConfig
) -> dict[str, Any]:
    if config.method == METHOD:
        validation = validate_external_feature_store(feature_path, replay=False)
        train_raw, test_raw = load_external_task_features(feature_path, config.task)
    else:
        from features.phase6_7_lwa_features import (
            load_lwa_task_features,
            validate_lwa_feature_store,
        )

        validation = validate_lwa_feature_store(feature_path, replay=False)
        train_raw, test_raw = load_lwa_task_features(
            feature_path, config.task, validate=False
        )
    if validation["walk"] != config.walk or validation["method"] != config.method:
        raise ValueError("external downstream feature method/walk mismatch")
    feature_manifest = json.loads(
        Path(f"{feature_path}.manifest.json").read_text(encoding="utf-8")
    )
    task_record = feature_manifest["task_provenance"][config.task]
    if str(Path(task_record["dataset_path"]).resolve()).casefold() != str(
        dataset_path.resolve()
    ).casefold():
        raise ValueError("external downstream dataset path differs from feature source")
    if task_record["dataset_sha256"] != sha256_file(dataset_path):
        raise ValueError("external downstream dataset hash differs from feature source")
    if train_raw.shape[1] != config.input_dim or test_raw.shape[1] != config.input_dim:
        raise ValueError("external downstream native feature width mismatch")
    scaler = fit_external_standardizer(train_raw)
    result: dict[str, Any] = {
        "X_train": apply_external_standardizer(train_raw, scaler),
        "X_test": apply_external_standardizer(test_raw, scaler),
        "scaler": scaler,
    }
    with np.load(dataset_path, allow_pickle=False) as source:
        for split in ("train", "test"):
            for field in IDENTITY_FIELDS:
                result[f"{split}_{field}"] = np.asarray(source[f"{split}_{field}"])
            if config.task == "classification_h2":
                result[f"y_{split}"] = np.asarray(
                    source[f"{split}_classification_labels"], dtype=np.int64
                )
            elif config.task == "absolute_price_h8":
                result[f"y_{split}"] = np.asarray(
                    source[f"{split}_target_close"], dtype=np.float64
                )
                result[f"{split}_current_close"] = np.asarray(
                    source[f"{split}_current_close"], dtype=np.float64
                )
                result[f"{split}_raw_sequences"] = np.asarray(
                    source[f"{split}_raw_sequences"], dtype=np.float32
                )
                for name in ("context_imputed_rows", "lifecycle_stage"):
                    result[f"{split}_{name}"] = np.asarray(source[f"{split}_{name}"])
            else:
                result[f"y_{split}"] = np.asarray(
                    source[f"{split}_realised_variance"], dtype=np.float64
                )
                result[f"{split}_raw_sequences"] = np.asarray(
                    source[f"{split}_raw_sequences"], dtype=np.float32
                )
                for name in (
                    "context_imputed_rows",
                    "lifecycle_stage",
                    "future_update_count",
                ):
                    result[f"{split}_{name}"] = np.asarray(source[f"{split}_{name}"])
    with np.load(feature_path, allow_pickle=False) as features:
        for split in ("train", "test"):
            prefix = f"{config.task}_{split}"
            for field in IDENTITY_FIELDS:
                if not np.array_equal(
                    features[f"{prefix}_{field}"], result[f"{split}_{field}"]
                ):
                    raise ValueError(f"external downstream identity mismatch: {split}_{field}")
    if len(result["X_train"]) != len(result["y_train"]):
        raise ValueError("external training features/targets are not aligned")
    if len(result["X_test"]) != len(result["y_test"]):
        raise ValueError("external evaluation features/targets are not aligned")
    return result


def build_external_head(config: Phase67DownstreamConfig) -> nn.Module:
    if config.task == "classification_h2":
        return TrendClassifier(config.input_dim, hidden_dim=config.hidden_dim, n_classes=3)
    if config.task == "absolute_price_h8":
        return AbsolutePriceRegressor(config.input_dim, hidden_dim=config.hidden_dim)
    return VolatilityRegressor(config.input_dim, hidden_dim=config.hidden_dim)


def smoke_test_external_downstream(method: str = METHOD) -> dict[str, Any]:
    if method not in METHOD_DIMS:
        raise ValueError(f"unknown external downstream method: {method}")
    set_seed(0)
    results = {}
    for task in TASKS:
        config = Phase67DownstreamConfig(
            task, method, walk=1, input_dim=METHOD_DIMS[method], device="cpu"
        )
        model = build_external_head(config)
        output = model(torch.randn(4, config.input_dim))
        if task == "classification_h2":
            loss = LogitAdjustedCrossEntropy(np.asarray([0.2, 0.6, 0.2]))(
                output, torch.tensor([0, 1, 2, 1])
            )
        elif task == "realised_variance":
            loss = nn.SmoothL1Loss(beta=1.0)(output, torch.rand(4, 1))
        else:
            loss = nn.MSELoss()(output, torch.rand(4, 1))
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError("external downstream smoke produced non-finite loss")
        results[task] = {"output_shape": list(output.shape), "loss_finite": True}
    return {"valid": True, "method": method, "heads": results}


@torch.no_grad()
def _predict(
    model: nn.Module, values: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    outputs = [model(batch.to(device)).cpu().numpy() for batch in tensor.split(batch_size)]
    result = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise FloatingPointError("external downstream prediction is non-finite")
    return result


def _cross_sectional_rank_ic(
    score: np.ndarray, target: np.ndarray, timestamps: np.ndarray
) -> dict[str, float | int]:
    values = []
    for timestamp in np.unique(timestamps):
        mask = timestamps == timestamp
        if int(mask.sum()) < 5:
            continue
        value = regression_metrics(score[mask], target[mask])["spearman"]
        if np.isfinite(value):
            values.append(float(value))
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "icir_unannualized": float("nan"),
            "positive_fraction": float("nan"),
        }
    std = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": std,
        "icir_unannualized": float(array.mean() / std) if std > 0.0 else float("nan"),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def _evaluate(
    config: Phase67DownstreamConfig,
    output: np.ndarray,
    data: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    identities = {field: np.asarray(data[f"test_{field}"]) for field in IDENTITY_FIELDS}
    target = np.asarray(data["y_test"])
    if config.task == "classification_h2":
        metrics, arrays = probabilistic_classification_metrics(output, target, CLASS_NAMES)
        stable_logits = np.full((len(target), 3), -30.0, dtype=np.float32)
        stable_logits[:, 1] = 0.0
        stable_metrics, _ = probabilistic_classification_metrics(
            stable_logits, target, CLASS_NAMES
        )
        priors = class_priors(np.asarray(data["y_train"]), n_classes=3)
        prior_logits = np.broadcast_to(np.log(priors), (len(target), 3)).copy()
        prior_metrics, _ = probabilistic_classification_metrics(
            prior_logits, target, CLASS_NAMES
        )
        return {
            "framework": metrics,
            "always_stable_reference": stable_metrics,
            "repeated_training_prior_reference": prior_metrics,
        }, {**arrays, **identities}
    prediction = output.reshape(-1).astype(np.float64)
    if config.task == "absolute_price_h8":
        current = np.asarray(data["test_current_close"], dtype=np.float64)
        metadata = {
            "condition_ids": identities["condition_ids"],
            "current_close": current,
            "context_imputed_rows": data["test_context_imputed_rows"],
            "lifecycle_stage": data["test_lifecycle_stage"],
        }
        price, _ = price_breakdowns(prediction, target, metadata)
        persistence, _ = price_breakdowns(current, target, metadata)
        prediction_delta = prediction - current
        target_delta = target - current
        raw_sequences = np.asarray(data["test_raw_sequences"], dtype=np.float64)
        reversal = -(raw_sequences[:, -1, 3] - raw_sequences[:, -2, 3])
        timestamps = identities["decision_date_ns"]
        nonzero = target_delta != 0.0
        metrics = {
            "price": price,
            "persistence_reference": persistence,
            "implied_movement": regression_metrics(prediction_delta, target_delta),
            "implied_movement_cross_sectional_rank_ic": _cross_sectional_rank_ic(
                prediction_delta, target_delta, timestamps
            ),
            "last_hour_reversal_cross_sectional_rank_ic": _cross_sectional_rank_ic(
                reversal, target_delta, timestamps
            ),
            "implied_movement_nonzero_sign_agreement": (
                float(np.mean(np.sign(prediction_delta[nonzero]) == np.sign(target_delta[nonzero])))
                if np.any(nonzero)
                else float("nan")
            ),
        }
        return metrics, {
            "prediction_future_price": prediction,
            "target_future_price": target,
            "current_close": current,
            "prediction_delta_h8": prediction_delta,
            "target_delta_h8": target_delta,
            "last_hour_reversal": reversal,
            **identities,
        }
    prediction = prediction / config.volatility_target_multiplier
    zero = np.zeros_like(target)
    median = np.full_like(target, np.median(data["y_train"]))
    persistence = historical_persistence(np.asarray(data["test_raw_sequences"]))
    return {
        "model": volatility_metrics(prediction, target),
        "zero_reference": volatility_metrics(zero, target),
        "training_median_reference": volatility_metrics(median, target),
        "historical_persistence_reference": volatility_metrics(persistence, target),
    }, {
        "prediction_realised_variance": prediction,
        "prediction_scaled_squared_probability_points": output.reshape(-1),
        "target_realised_variance": target,
        "zero_reference": zero,
        "training_median_reference": median,
        "historical_persistence_reference": persistence,
        **identities,
    }


def run_external_downstream(
    dataset_path: Path,
    feature_path: Path,
    run_root: Path,
    config: Phase67DownstreamConfig,
) -> list[dict[str, Any]]:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite external downstream run: {run_root}")
    data = _load_task_data(dataset_path, feature_path, config)
    device = resolve_device(config.device)
    set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    y_train = np.asarray(data["y_train"])
    if config.task == "classification_h2":
        target_tensor = torch.from_numpy(y_train.astype(np.int64))
        priors = class_priors(y_train, n_classes=3)
        criterion: nn.Module = LogitAdjustedCrossEntropy(
            priors, config.logit_adjustment_strength
        )
    elif config.task == "realised_variance":
        target_tensor = torch.from_numpy(
            (y_train * config.volatility_target_multiplier)
            .astype(np.float32)
            .reshape(-1, 1)
        )
        criterion = nn.SmoothL1Loss(beta=config.smooth_l1_beta)
        priors = None
    else:
        target_tensor = torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1))
        criterion = nn.MSELoss()
        priors = None
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data["X_train"]), target_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = build_external_head(config).to(device)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "config.json", config.to_dict())
    write_json(run_root / "environment.json", environment_manifest(device))
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    write_json(
        run_root / "architecture_manifest.json",
        {
            "method": config.method,
            "input_dim": config.input_dim,
            "native_width_retained": True,
            "hidden_layers": [128, 64],
            "task_head": {
                "classification_h2": "TrendClassifier",
                "absolute_price_h8": "AbsolutePriceRegressor",
                "realised_variance": "VolatilityRegressor",
            }[config.task],
            "parameter_count": parameter_count,
        },
    )
    scaler_path = run_root / "feature_standardizer.npz"
    _atomic_savez(scaler_path, data["scaler"])
    write_json(
        run_root / "feature_standardizer.manifest.json",
        {
            "fit_population": "task training rows only",
            "evaluation_used": False,
            "dimension": config.input_dim,
            "sha256": sha256_file(scaler_path),
            "array_hashes": {
                name: array_sha256(value) for name, value in data["scaler"].items()
            },
        },
    )
    write_json(
        run_root / "dataset_manifest.json",
        {
            "phase": "6.7",
            "task": config.task,
            "method": config.method,
            "walk": config.walk,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": sha256_file(dataset_path),
            "feature_path": str(feature_path.resolve()),
            "feature_sha256": sha256_file(feature_path),
            "train_rows": len(data["X_train"]),
            "test_rows": len(data["X_test"]),
            "evaluation_used_for_selection": False,
            "train_identity_hash": sha256_arrays(
                *(data[f"train_{field}"] for field in IDENTITY_FIELDS)
            ),
            "test_identity_hash": sha256_arrays(
                *(data[f"test_{field}"] for field in IDENTITY_FIELDS)
            ),
            "train_target_hash": array_sha256(data["y_train"]),
            "test_target_hash": array_sha256(data["y_test"]),
            "input_dim": config.input_dim,
        },
    )
    write_json(
        run_root / "training_contract.json",
        {
            "optimizer": "Adam",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "loss": {
                "classification_h2": "train-prior logit-adjusted cross-entropy",
                "absolute_price_h8": "mean squared error",
                "realised_variance": "Smooth L1 on 10000 * raw realised variance",
            }[config.task],
            "training_class_priors": priors.tolist() if priors is not None else None,
            "logit_adjustment_strength": (
                config.logit_adjustment_strength
                if config.task == "classification_h2"
                else None
            ),
            "smooth_l1_beta": (
                config.smooth_l1_beta if config.task == "realised_variance" else None
            ),
            "gradient_clip_norm": (
                config.volatility_gradient_clip_norm
                if config.task == "realised_variance"
                else None
            ),
            "evaluation_used_for_fitting_or_selection": False,
            "principal_epoch": 50,
        },
    )
    losses: list[float] = []
    seconds: list[float] = []
    checkpoint_paths: list[Path] = []
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite external downstream loss")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError("non-finite external downstream gradient")
            if config.task == "realised_variance":
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.volatility_gradient_clip_norm
                )
            optimizer.step()
            total += float(loss.detach()) * len(inputs)
            count += len(inputs)
        losses.append(total / count)
        seconds.append(time.perf_counter() - started)
        if epoch in SNAPSHOT_EPOCHS:
            snapshot = run_root / f"e{epoch}"
            snapshot.mkdir()
            checkpoint_path = snapshot / "checkpoint.pth"
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase": "6.7",
                    "purpose": "external_frozen_representation_probe",
                    "task": config.task,
                    "method": config.method,
                    "walk": config.walk,
                    "completed_epoch": epoch,
                    "config": config.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dataset_sha256": sha256_file(dataset_path),
                    "feature_sha256": sha256_file(feature_path),
                    "scaler_sha256": sha256_file(scaler_path),
                },
                checkpoint_path,
            )
            _atomic_savez(
                snapshot / "history.npz",
                {
                    "epochs": np.arange(1, epoch + 1, dtype=np.int32),
                    "train_loss": np.asarray(losses),
                    "epoch_seconds": np.asarray(seconds),
                },
            )
            checkpoint_paths.append(checkpoint_path)
    snapshots = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        evaluated = build_external_head(config).to(device)
        evaluated.load_state_dict(checkpoint["model_state_dict"], strict=True)
        inference_started = time.perf_counter()
        output = _predict(evaluated, data["X_test"], config.batch_size, device)
        inference_seconds = time.perf_counter() - inference_started
        metrics, arrays = _evaluate(config, output, data)
        snapshot = checkpoint_path.parent
        _atomic_savez(snapshot / "predictions.npz", arrays)
        epoch = int(checkpoint["completed_epoch"])
        row = {
            "epoch": epoch,
            "train_loss": losses[epoch - 1],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(snapshot / "predictions.npz"),
            "elapsed_training_seconds": float(sum(seconds[:epoch])),
            "inference_seconds": inference_seconds,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        }
        write_json(snapshot / "metrics.json", {**row, "metrics": metrics})
        snapshots.append(row)
    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    write_json(
        run_root / "training_complete.json",
        {
            "complete": True,
            "snapshots": list(SNAPSHOT_EPOCHS),
            "principal_epoch": 50,
            "selection_rule": "epoch 50 predeclared; no evaluation-driven selection",
        },
    )
    return snapshots


def _assert_metric_replay(saved: Any, replayed: Any, path: str = "metrics") -> None:
    if isinstance(saved, Mapping) and isinstance(replayed, Mapping):
        if set(saved) != set(replayed):
            raise ValueError(f"external downstream {path} fields mismatch")
        for key in saved:
            _assert_metric_replay(saved[key], replayed[key], f"{path}.{key}")
        return
    if isinstance(saved, list) and isinstance(replayed, list):
        if len(saved) != len(replayed):
            raise ValueError(f"external downstream {path} length mismatch")
        for index, (left, right) in enumerate(zip(saved, replayed)):
            _assert_metric_replay(left, right, f"{path}[{index}]")
        return
    if isinstance(saved, (int, float)) and isinstance(replayed, (int, float)):
        if isinstance(saved, int) and isinstance(replayed, int):
            if saved != replayed:
                raise ValueError(f"external downstream {path} mismatch")
        elif not np.isclose(
            float(saved), float(replayed), rtol=1e-4, atol=1e-7, equal_nan=True
        ):
            raise ValueError(f"external downstream {path} mismatch")
        return
    if saved != replayed:
        raise ValueError(f"external downstream {path} mismatch")


def validate_external_downstream(
    dataset_path: Path, feature_path: Path, run_root: Path
) -> dict[str, Any]:
    required = {
        "config.json",
        "environment.json",
        "architecture_manifest.json",
        "feature_standardizer.npz",
        "feature_standardizer.manifest.json",
        "dataset_manifest.json",
        "training_contract.json",
        "sweep_metrics.json",
        "training_complete.json",
    }
    missing = sorted(name for name in required if not (run_root / name).is_file())
    if missing:
        raise ValueError(f"external downstream run is missing artifacts: {missing}")
    config = _config_from_payload(
        json.loads((run_root / "config.json").read_text(encoding="utf-8")), "cpu"
    )
    data = _load_task_data(dataset_path, feature_path, config)
    source = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if source.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("external downstream dataset hash mismatch")
    if source.get("feature_sha256") != sha256_file(feature_path):
        raise ValueError("external downstream feature hash mismatch")
    for split in ("train", "test"):
        if source.get(f"{split}_identity_hash") != sha256_arrays(
            *(data[f"{split}_{field}"] for field in IDENTITY_FIELDS)
        ):
            raise ValueError(f"external downstream {split} identity hash mismatch")
        if source.get(f"{split}_target_hash") != array_sha256(data[f"y_{split}"]):
            raise ValueError(f"external downstream {split} target hash mismatch")
    scaler_path = run_root / "feature_standardizer.npz"
    scaler_manifest = json.loads(
        (run_root / "feature_standardizer.manifest.json").read_text(encoding="utf-8")
    )
    if scaler_manifest.get("sha256") != sha256_file(scaler_path):
        raise ValueError("external downstream scaler hash mismatch")
    with np.load(scaler_path, allow_pickle=False) as saved_scaler:
        for name, value in data["scaler"].items():
            if not np.array_equal(saved_scaler[name], value):
                raise ValueError(f"external downstream scaler replay mismatch: {name}")
    rows = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))[
        "snapshots"
    ]
    if [int(row["epoch"]) for row in rows] != list(SNAPSHOT_EPOCHS):
        raise ValueError("external downstream snapshot matrix is incomplete")
    for row in rows:
        epoch = int(row["epoch"])
        snapshot = run_root / f"e{epoch}"
        checkpoint_path = snapshot / "checkpoint.pth"
        predictions_path = snapshot / "predictions.npz"
        history_path = snapshot / "history.npz"
        metrics_path = snapshot / "metrics.json"
        if not all(
            path.is_file()
            for path in (checkpoint_path, predictions_path, history_path, metrics_path)
        ):
            raise ValueError(f"external downstream e{epoch} snapshot is incomplete")
        if sha256_file(checkpoint_path) != row["checkpoint_sha256"]:
            raise ValueError(f"external downstream e{epoch} checkpoint hash mismatch")
        if sha256_file(predictions_path) != row["predictions_sha256"]:
            raise ValueError(f"external downstream e{epoch} prediction hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("schema_version") != SCHEMA_VERSION
            or int(checkpoint.get("completed_epoch", -1)) != epoch
            or checkpoint.get("dataset_sha256") != sha256_file(dataset_path)
            or checkpoint.get("feature_sha256") != sha256_file(feature_path)
            or checkpoint.get("scaler_sha256") != sha256_file(scaler_path)
        ):
            raise ValueError(f"external downstream e{epoch} checkpoint contract mismatch")
        model = build_external_head(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        replay = _predict(model, data["X_test"], config.batch_size, torch.device("cpu"))
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        with np.load(predictions_path, allow_pickle=False) as saved:
            output_key = {
                "classification_h2": "logits",
                "absolute_price_h8": "prediction_future_price",
                "realised_variance": "prediction_scaled_squared_probability_points",
            }[config.task]
            saved_output = np.asarray(saved[output_key])
            replay_output = replay if config.task == "classification_h2" else replay.reshape(-1)
            if not np.allclose(saved_output, replay_output, rtol=2e-5, atol=2e-5):
                raise ValueError(f"external downstream prediction replay mismatch: {output_key}")
            replay_metrics, expected = _evaluate(config, saved_output, data)
            _assert_metric_replay(metrics_payload["metrics"], replay_metrics)
            if set(saved.files) != set(expected):
                raise ValueError("external prediction store has missing or unexpected arrays")
            exact_fields = set(IDENTITY_FIELDS)
            if config.task == "classification_h2":
                exact_fields.add("targets")
            elif config.task == "absolute_price_h8":
                exact_fields.update(
                    {
                        "target_future_price",
                        "current_close",
                        "target_delta_h8",
                        "last_hour_reversal",
                    }
                )
            else:
                exact_fields.update(
                    {
                        "target_realised_variance",
                        "zero_reference",
                        "training_median_reference",
                        "historical_persistence_reference",
                    }
                )
            for name in exact_fields:
                if not np.array_equal(saved[name], expected[name]):
                    raise ValueError(f"external prediction identity/reference mismatch: {name}")
        with np.load(history_path, allow_pickle=False) as history:
            if set(history.files) != {"epochs", "train_loss", "epoch_seconds"}:
                raise ValueError(f"external downstream e{epoch} history fields mismatch")
            if len(history["epochs"]) != epoch or int(history["epochs"][-1]) != epoch:
                raise ValueError(f"external downstream e{epoch} history mismatch")
    complete = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("principal_epoch") != 50:
        raise ValueError("external downstream completion marker is invalid")
    return {
        "valid": True,
        "task": config.task,
        "method": config.method,
        "walk": config.walk,
        "input_dim": config.input_dim,
        "snapshots": list(SNAPSHOT_EPOCHS),
    }
