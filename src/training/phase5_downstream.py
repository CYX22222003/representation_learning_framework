"""Leakage-safe Phase 5 framework-only downstream training and evaluation."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from torch.utils.data import DataLoader, TensorDataset

from data_processing.phase5_walks import CLASS_NAMES, validate_phase5_bundle_files
from features.phase5_features import (
    IDENTITY_FIELDS,
    array_sha256,
    load_concat_features,
    validate_phase5_feature_bundle,
)
from tasks.phase2_classification.metrics import probabilistic_classification_metrics
from tasks.phase2_classification.protocols import LogitAdjustedCrossEntropy, class_priors
from tasks.price_prediction import PriceRegressor
from tasks.trend_classification import TrendClassifier
from training.phase5_encoder import sha256_file, write_json


TASKS = ("regression", "classification")
SNAPSHOT_EPOCHS = (5, 15, 50)
SCALER_SCHEMA_VERSION = "phase5-feature-standardizer-v1"


@dataclass(frozen=True)
class Phase5DownstreamConfig:
    task: str
    walk: int
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    hidden_dim: int = 128
    regression_target_multiplier: float = 100.0
    logit_adjustment_strength: float = 1.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("Phase 5 downstream runs require 50 epochs and 5/15/50 snapshots")
        if self.seed != 0:
            raise ValueError("the initial Phase 5 downstream matrix is frozen to seed 0")
        if self.batch_size != 512 or self.learning_rate != 1e-4:
            raise ValueError("Phase 5 downstream batch size/lr are frozen to 512/1e-4")
        if self.regression_target_multiplier != 100.0:
            raise ValueError("Phase 5 regression is frozen to probability-point units (100 * delta)")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["snapshot_epochs"] = list(self.snapshot_epochs)
        return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def environment_manifest(device: torch.device) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "python": os.sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": commit,
    }


def fit_feature_standardizer(train: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(train, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 445 or not np.isfinite(values).all():
        raise ValueError("standardizer input must be finite [N,445]")
    mean = values.mean(axis=0)
    raw_std = values.std(axis=0, ddof=0)
    scale = np.where(raw_std < 1e-8, 1.0, raw_std)
    return {
        "mean": mean.astype(np.float64),
        "raw_std": raw_std.astype(np.float64),
        "scale": scale.astype(np.float64),
        "std_floor": np.asarray(1e-8, dtype=np.float64),
        "clip_low": np.asarray(-10.0, dtype=np.float64),
        "clip_high": np.asarray(10.0, dtype=np.float64),
    }


def apply_feature_standardizer(values: np.ndarray, scaler: Mapping[str, np.ndarray]) -> np.ndarray:
    result = (np.asarray(values, dtype=np.float64) - scaler["mean"]) / scaler["scale"]
    result = np.clip(result, float(scaler["clip_low"]), float(scaler["clip_high"]))
    result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("feature standardization produced non-finite values")
    return result


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def prepare_feature_standardizer(
    feature_path: Path, scaler_path: Path, *, dataset_path: Path
) -> dict[str, Any]:
    """Fit once on supervised training features and prove evaluation independence."""

    validate_phase5_feature_bundle(feature_path, dataset_path=dataset_path)
    train, test = load_concat_features(feature_path)
    fitted = fit_feature_standardizer(train)
    # Deliberately perturb evaluation features and repeat the train-only fit.
    perturbed_test = test.astype(np.float64) * 17.0 + 3.0
    repeated = fit_feature_standardizer(train)
    if not all(np.array_equal(fitted[key], repeated[key]) for key in fitted):
        raise RuntimeError("train-only scaler changed under evaluation perturbation")
    before = {key: array_sha256(value) for key, value in fitted.items()}
    apply_feature_standardizer(perturbed_test[: min(32, len(perturbed_test))], fitted)
    after = {key: array_sha256(value) for key, value in fitted.items()}
    if before != after:
        raise RuntimeError("evaluation transform mutated fitted scaler state")
    if scaler_path.exists() or Path(f"{scaler_path}.manifest.json").exists():
        return validate_feature_standardizer(feature_path, scaler_path, dataset_path=dataset_path)
    _atomic_savez(scaler_path, fitted)
    manifest = {
        "schema_version": SCALER_SCHEMA_VERSION,
        "phase": 5,
        "purpose": "walk_local_train_only_downstream_feature_standardizer",
        "walk": int(json.loads(Path(f"{feature_path}.manifest.json").read_text())["walk"]),
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "source_dataset_path": str(dataset_path),
        "source_dataset_sha256": sha256_file(dataset_path),
        "fit_population": "supervised train features only",
        "evaluation_values_used_for_fit": False,
        "dimension": 445,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "zero_variance_coordinates": int(np.count_nonzero(fitted["raw_std"] < 1e-8)),
        "array_hashes": before,
        "perturbation_invariance_verified": True,
        "scaler_sha256": sha256_file(scaler_path),
    }
    write_json(Path(f"{scaler_path}.manifest.json"), manifest)
    return validate_feature_standardizer(feature_path, scaler_path, dataset_path=dataset_path)


def validate_feature_standardizer(
    feature_path: Path, scaler_path: Path, *, dataset_path: Path
) -> dict[str, Any]:
    manifest_path = Path(f"{scaler_path}.manifest.json")
    if not scaler_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("feature standardizer or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCALER_SCHEMA_VERSION:
        raise ValueError("unsupported feature standardizer schema")
    if sha256_file(scaler_path) != manifest.get("scaler_sha256"):
        raise ValueError("feature standardizer file hash mismatch")
    if sha256_file(feature_path) != manifest.get("feature_sha256"):
        raise ValueError("standardizer feature source hash mismatch")
    if sha256_file(dataset_path) != manifest.get("source_dataset_sha256"):
        raise ValueError("standardizer dataset source hash mismatch")
    train, test = load_concat_features(feature_path)
    expected = fit_feature_standardizer(train)
    with np.load(scaler_path, allow_pickle=False) as saved:
        for key, value in expected.items():
            if not np.array_equal(saved[key], value):
                raise ValueError(f"standardizer replay mismatch for {key}")
            if array_sha256(saved[key]) != manifest["array_hashes"][key]:
                raise ValueError(f"standardizer array hash mismatch for {key}")
        transformed_train = apply_feature_standardizer(train, saved)
        transformed_test = apply_feature_standardizer(test, saved)
    return {
        "valid": True,
        "walk": int(manifest["walk"]),
        "dimension": 445,
        "train_rows": len(transformed_train),
        "test_rows": len(transformed_test),
        "zero_variance_coordinates": int(manifest["zero_variance_coordinates"]),
        "perturbation_invariance_verified": bool(manifest["perturbation_invariance_verified"]),
    }


def load_phase5_downstream_data(
    dataset_path: Path, feature_path: Path, scaler_path: Path
) -> dict[str, Any]:
    validate_phase5_bundle_files(dataset_path)
    validate_phase5_feature_bundle(feature_path, dataset_path=dataset_path)
    validate_feature_standardizer(feature_path, scaler_path, dataset_path=dataset_path)
    train_raw, test_raw = load_concat_features(feature_path)
    with np.load(scaler_path, allow_pickle=False) as scaler:
        train = apply_feature_standardizer(train_raw, scaler)
        test = apply_feature_standardizer(test_raw, scaler)
    with np.load(dataset_path, allow_pickle=False) as source:
        result: dict[str, Any] = {
            "X_train": train,
            "X_test": test,
            "y_regression_train": np.asarray(source["train_regression_labels"], dtype=np.float64),
            "y_regression_test": np.asarray(source["test_regression_labels"], dtype=np.float64),
            "y_classification_train": np.asarray(source["train_classification_labels"], dtype=np.int64),
            "y_classification_test": np.asarray(source["test_classification_labels"], dtype=np.int64),
        }
        metadata_fields = (
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
            for field in metadata_fields:
                result[f"{split}_{field}"] = np.asarray(source[f"{split}_{field}"])
    if len(train) != len(result["y_regression_train"]) or len(test) != len(result["y_regression_test"]):
        raise ValueError("feature and target row counts are not aligned")
    return result


def build_head(task: str) -> nn.Module:
    if task == "regression":
        return PriceRegressor(input_dim=445, hidden_dim=128)
    if task == "classification":
        return TrendClassifier(input_dim=445, hidden_dim=128, n_classes=3)
    raise ValueError(f"unknown task: {task}")


def smoke_test_heads() -> dict[str, Any]:
    set_seed(0)
    x = torch.randn(16, 445)
    regression = build_head("regression")
    regression_loss = nn.MSELoss()(regression(x), torch.randn(16, 1))
    regression_loss.backward()
    priors = np.asarray([0.2, 0.6, 0.2], dtype=np.float32)
    classification = build_head("classification")
    classification_loss = LogitAdjustedCrossEntropy(priors)(
        classification(x), torch.arange(16) % 3
    )
    classification_loss.backward()
    if not torch.isfinite(regression_loss) or not torch.isfinite(classification_loss):
        raise FloatingPointError("downstream CPU smoke test produced a non-finite loss")
    return {
        "valid": True,
        "regression_output_shape": list(regression(x).shape),
        "classification_output_shape": list(classification(x).shape),
        "regression_loss": float(regression_loss.detach()),
        "classification_loss": float(classification_loss.detach()),
    }


@torch.no_grad()
def predict(model: nn.Module, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    output = [model(batch.to(device)).cpu().numpy() for batch in tensor.split(batch_size)]
    result = np.concatenate(output, axis=0).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("downstream model produced non-finite predictions")
    return result


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(prediction_delta: np.ndarray, target_delta: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction_delta, dtype=np.float64).reshape(-1)
    target = np.asarray(target_delta, dtype=np.float64).reshape(-1)
    if pred.shape != target.shape or len(target) == 0:
        raise ValueError("regression prediction/target shape mismatch or empty group")
    error = pred - target
    return {
        "count": int(len(target)),
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(np.square(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson": _correlation(pred, target),
        "spearman": _correlation(rankdata(pred), rankdata(target)),
        "sign_agreement": float(np.mean(np.sign(pred) == np.sign(target))),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
    }


def _masked_regression_metrics(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, Any] | None:
    selected = np.asarray(mask, dtype=bool)
    return regression_metrics(pred[selected], target[selected]) if np.any(selected) else None


def regression_breakdowns(
    prediction_delta: np.ndarray,
    target_delta: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    *,
    tau: float = 0.001,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = np.asarray(target_delta)
    pred = np.asarray(prediction_delta)
    groups: dict[str, Any] = {
        "overall": regression_metrics(pred, target),
        "exact_zero": _masked_regression_metrics(pred, target, target == 0.0),
        "non_zero": _masked_regression_metrics(pred, target, target != 0.0),
        "within_classification_threshold": _masked_regression_metrics(pred, target, np.abs(target) <= tau),
        "threshold_exceeding": _masked_regression_metrics(pred, target, np.abs(target) > tau),
        "context_has_imputation": _masked_regression_metrics(
            pred, target, np.asarray(metadata["context_imputed_rows"]) > 0
        ),
        "context_observed_only": _masked_regression_metrics(
            pred, target, np.asarray(metadata["context_imputed_rows"]) == 0
        ),
    }
    stage_names = {-1: "unknown", 0: "early", 1: "middle", 2: "late"}
    stage = np.asarray(metadata["lifecycle_stage"])
    groups["lifecycle"] = {
        name: _masked_regression_metrics(pred, target, stage == code)
        for code, name in stage_names.items()
    }
    per_contract: dict[str, Any] = {}
    condition_ids = np.asarray(metadata["condition_ids"])
    for condition_id in np.unique(condition_ids):
        mask = condition_ids == condition_id
        per_contract[str(condition_id)] = regression_metrics(pred[mask], target[mask])
    metric_names = ("mae", "mse", "rmse", "pearson", "spearman", "sign_agreement")
    groups["contract_macro"] = {
        metric: float(
            np.mean(
                [row[metric] for row in per_contract.values() if math.isfinite(float(row[metric]))]
            )
        )
        for metric in metric_names
        if any(math.isfinite(float(row[metric])) for row in per_contract.values())
    }
    groups["contract_count"] = len(per_contract)
    return groups, per_contract


def _prediction_metadata(data: Mapping[str, Any], split: str = "test") -> dict[str, np.ndarray]:
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


def _save_history(path: Path, losses: list[float], seconds: list[float]) -> None:
    _atomic_savez(
        path,
        {
            "epochs": np.arange(1, len(losses) + 1, dtype=np.int32),
            "train_loss": np.asarray(losses, dtype=np.float64),
            "epoch_seconds": np.asarray(seconds, dtype=np.float64),
        },
    )


def _model_spec(task: str) -> dict[str, Any]:
    return {
        "input_dim": 445,
        "hidden_layers": [128, 64],
        "activation": "GELU",
        "dropout": 0.1 if task == "regression" else 0.2,
        "output_dim": 1 if task == "regression" else 3,
        "aggregation": "fixed five-branch concat; no trainable aggregator parameters",
    }


def run_downstream_training(
    dataset_path: Path,
    feature_path: Path,
    scaler_path: Path,
    run_root: Path,
    config: Phase5DownstreamConfig,
) -> list[dict[str, Any]]:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite downstream run: {run_root}")
    data = load_phase5_downstream_data(dataset_path, feature_path, scaler_path)
    device = resolve_device(config.device)
    set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    X_train = data["X_train"]
    X_test = data["X_test"]
    if config.task == "regression":
        y_train = (data["y_regression_train"] * config.regression_target_multiplier).astype(np.float32)
        y_test = data["y_regression_test"]
        target_tensor = torch.from_numpy(y_train.reshape(-1, 1))
        criterion: nn.Module = nn.MSELoss()
        priors = None
    else:
        y_train = data["y_classification_train"]
        y_test = data["y_classification_test"]
        target_tensor = torch.from_numpy(y_train)
        priors = class_priors(y_train, n_classes=3)
        criterion = LogitAdjustedCrossEntropy(
            priors, strength=config.logit_adjustment_strength
        )
    dataset = TensorDataset(torch.from_numpy(X_train), target_tensor)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    model = build_head(config.task).to(device)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "config.json", config.to_dict())
    write_json(run_root / "environment.json", environment_manifest(device))
    write_json(run_root / "architecture_manifest.json", {**_model_spec(config.task), "parameter_count": parameter_count})
    feature_manifest = json.loads(Path(f"{feature_path}.manifest.json").read_text(encoding="utf-8"))
    data_manifest = json.loads(Path(f"{dataset_path}.manifest.json").read_text(encoding="utf-8"))
    write_json(
        run_root / "dataset_manifest.json",
        {
            "phase": 5,
            "purpose": "framework_downstream_probe",
            "walk": config.walk,
            "task": config.task,
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "feature_path": str(feature_path),
            "feature_sha256": sha256_file(feature_path),
            "scaler_path": str(scaler_path),
            "scaler_sha256": sha256_file(scaler_path),
            "train_identity_hash": data_manifest["identity_hashes"]["train"],
            "test_identity_hash": data_manifest["identity_hashes"]["test"],
            "branch_order": feature_manifest["branch_order"],
            "branch_hashes": feature_manifest["branch_hashes"],
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "evaluation_used_for_fitting_or_selection": False,
            "checkpoint_selection": "epoch 50 predeclared",
            "target_unit": "probability_points" if config.task == "regression" else "class_index",
            "scientific_reporting_unit": "raw_probability_delta" if config.task == "regression" else "DOWN/STABLE/UP",
        },
    )
    if priors is not None:
        write_json(
            run_root / "imbalance_manifest.json",
            {
                "sampling": "natural rows",
                "loss": "logit-adjusted cross-entropy",
                "strength": config.logit_adjustment_strength,
                "training_class_counts": np.bincount(y_train, minlength=3).tolist(),
                "training_class_priors": priors.tolist(),
                "evaluation_distribution_untouched": True,
            },
        )

    losses: list[float] = []
    epoch_seconds: list[float] = []
    checkpoint_paths: list[Path] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        row_count = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {config.task} loss at epoch {epoch}")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"non-finite {config.task} gradient at epoch {epoch}")
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(x_batch)
            row_count += len(x_batch)
        losses.append(loss_sum / row_count)
        epoch_seconds.append(time.perf_counter() - epoch_started)
        print(
            f"walk={config.walk} task={config.task} epoch={epoch} "
            f"loss={losses[-1]:.8f} seconds={epoch_seconds[-1]:.2f}",
            flush=True,
        )
        if epoch in config.snapshot_epochs:
            snapshot_dir = run_root / f"e{epoch}"
            snapshot_dir.mkdir()
            checkpoint_path = snapshot_dir / "checkpoint.pth"
            torch.save(
                {
                    "phase": 5,
                    "purpose": "framework_downstream_probe",
                    "walk": config.walk,
                    "task": config.task,
                    "completed_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "model_spec": _model_spec(config.task),
                    "parameter_count": parameter_count,
                    "dataset_sha256": sha256_file(dataset_path),
                    "feature_sha256": sha256_file(feature_path),
                    "scaler_sha256": sha256_file(scaler_path),
                },
                checkpoint_path,
            )
            _save_history(snapshot_dir / "history.npz", losses, epoch_seconds)
            checkpoint_paths.append(checkpoint_path)

    metadata = _prediction_metadata(data)
    snapshots: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        evaluated = build_head(config.task).to(device)
        evaluated.load_state_dict(checkpoint["model_state_dict"], strict=True)
        inference_started = time.perf_counter()
        output = predict(evaluated, X_test, config.batch_size, device)
        inference_seconds = time.perf_counter() - inference_started
        snapshot_dir = checkpoint_path.parent
        if config.task == "regression":
            prediction_pp = output.reshape(-1).astype(np.float64)
            prediction_delta = prediction_pp / config.regression_target_multiplier
            metrics, per_contract = regression_breakdowns(
                prediction_delta, y_test, metadata
            )
            reference, reference_per_contract = regression_breakdowns(
                np.zeros_like(y_test), y_test, metadata
            )
            prediction_arrays = {
                "prediction_probability_points": prediction_pp,
                "prediction_delta": prediction_delta,
                "target_probability_points": np.asarray(y_test) * 100.0,
                "target_delta": np.asarray(y_test),
                **metadata,
            }
            write_json(snapshot_dir / "per_contract_metrics.json", per_contract)
            write_json(snapshot_dir / "zero_reference_per_contract_metrics.json", reference_per_contract)
            principal = metrics["overall"]
            row = {
                "mae": principal["mae"],
                "rmse": principal["rmse"],
                "mse": principal["mse"],
                "pearson": principal["pearson"],
                "spearman": principal["spearman"],
                "sign_agreement": principal["sign_agreement"],
            }
            payload = {
                "framework": metrics,
                "exact_zero_reference": reference,
                "regression_unit_contract": {
                    "optimization": "100 * raw probability delta",
                    "reporting": "raw probability delta",
                    "inverse_applied": True,
                },
            }
        else:
            metrics, metric_arrays = probabilistic_classification_metrics(
                output, y_test, CLASS_NAMES
            )
            stable_logits = np.full((len(y_test), 3), -30.0, dtype=np.float32)
            stable_logits[:, 1] = 0.0
            stable_metrics, _ = probabilistic_classification_metrics(
                stable_logits, y_test, CLASS_NAMES
            )
            prior_logits = np.broadcast_to(
                np.log(np.asarray(priors, dtype=np.float64)), (len(y_test), 3)
            ).copy()
            prior_metrics, _ = probabilistic_classification_metrics(
                prior_logits, y_test, CLASS_NAMES
            )
            prediction_arrays = {**metric_arrays, **metadata}
            row = {
                key: metrics[key]
                for key in (
                    "accuracy",
                    "macro_f1",
                    "weighted_f1",
                    "balanced_accuracy",
                    "macro_roc_auc",
                    "macro_average_precision",
                    "nll",
                    "multiclass_brier",
                )
            }
            payload = {
                "framework": metrics,
                "always_stable_reference": stable_metrics,
                "repeated_training_prior_reference": prior_metrics,
                "class_names": list(CLASS_NAMES),
            }
        _atomic_savez(snapshot_dir / "predictions.npz", prediction_arrays)
        metrics_row = {
            "phase": 5,
            "walk": config.walk,
            "task": config.task,
            "epoch": int(checkpoint["completed_epoch"]),
            "seed": config.seed,
            "train_loss": float(losses[int(checkpoint["completed_epoch"]) - 1]),
            "parameter_count": parameter_count,
            "elapsed_training_seconds": float(sum(epoch_seconds[: int(checkpoint["completed_epoch"])])),
            "inference_seconds": inference_seconds,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(snapshot_dir / "predictions.npz"),
            **row,
        }
        write_json(snapshot_dir / "metrics.json", {**metrics_row, **payload})
        snapshots.append(metrics_row)

    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    write_json(
        run_root / "training_complete.json",
        {
            "complete": True,
            "phase": 5,
            "walk": config.walk,
            "task": config.task,
            "snapshots": list(config.snapshot_epochs),
            "principal_checkpoint": str(run_root / "e50" / "checkpoint.pth"),
            "selection_rule": "epoch 50 predeclared; evaluation metrics did not select a checkpoint",
        },
    )
    write_downstream_summary(run_root)
    return snapshots


def write_downstream_summary(run_root: Path) -> None:
    config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    rows = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))["snapshots"]
    task = config["task"]
    lines = [
        f"# Phase 5 Walk {config['walk']} {task.title()} Framework Probe",
        "",
        "One uninterrupted seed-0 trajectory. Epoch 50 is the predeclared principal result; evaluation did not select a checkpoint.",
        "",
    ]
    if task == "regression":
        lines.extend([
            "| Epoch | Train MSE (pp) | MAE | RMSE | Pearson | Spearman | Sign agreement |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                f"| {row['epoch']} | {row['train_loss']:.8f} | {row['mae']:.8f} | "
                f"{row['rmse']:.8f} | {row['pearson']:.6f} | {row['spearman']:.6f} | "
                f"{row['sign_agreement']:.6f} |"
            )
    else:
        lines.extend([
            "| Epoch | Train loss | Macro-F1 | Balanced accuracy | Accuracy | ROC-AUC | PR-AUC |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                f"| {row['epoch']} | {row['train_loss']:.8f} | {row['macro_f1']:.6f} | "
                f"{row['balanced_accuracy']:.6f} | {row['accuracy']:.6f} | "
                f"{row['macro_roc_auc']:.6f} | {row['macro_average_precision']:.6f} |"
            )
    lines.extend(["", "All metrics are descriptive snapshots; only epoch 50 is used for the principal Phase 5 result.", ""])
    (run_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _config_from_run(run_root: Path) -> Phase5DownstreamConfig:
    payload = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
    return Phase5DownstreamConfig(**payload)


def validate_downstream_run(
    dataset_path: Path,
    feature_path: Path,
    scaler_path: Path,
    run_root: Path,
    *,
    replay_tolerance: float = 2e-5,
) -> dict[str, Any]:
    required = {
        "config.json",
        "environment.json",
        "architecture_manifest.json",
        "dataset_manifest.json",
        "sweep_metrics.json",
        "training_complete.json",
        "summary.md",
    }
    missing = sorted(name for name in required if not (run_root / name).is_file())
    if missing:
        raise ValueError(f"downstream run is missing artifacts: {missing}")
    config = _config_from_run(run_root)
    data = load_phase5_downstream_data(dataset_path, feature_path, scaler_path)
    recorded = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    expected_hashes = {
        "dataset_sha256": sha256_file(dataset_path),
        "feature_sha256": sha256_file(feature_path),
        "scaler_sha256": sha256_file(scaler_path),
    }
    for key, expected in expected_hashes.items():
        if recorded.get(key) != expected:
            raise ValueError(f"downstream {key} mismatch")
    if recorded.get("evaluation_used_for_fitting_or_selection") is not False:
        raise ValueError("downstream run does not declare the evaluation isolation invariant")
    snapshots = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))["snapshots"]
    if [int(row["epoch"]) for row in snapshots] != list(SNAPSHOT_EPOCHS):
        raise ValueError("downstream snapshot matrix is incomplete")
    metadata = _prediction_metadata(data)
    for row in snapshots:
        epoch = int(row["epoch"])
        snapshot_dir = run_root / f"e{epoch}"
        checkpoint_path = snapshot_dir / "checkpoint.pth"
        predictions_path = snapshot_dir / "predictions.npz"
        metrics_path = snapshot_dir / "metrics.json"
        history_path = snapshot_dir / "history.npz"
        if not all(path.is_file() for path in (checkpoint_path, predictions_path, metrics_path, history_path)):
            raise ValueError(f"downstream e{epoch} artifacts are incomplete")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if sha256_file(checkpoint_path) != metrics.get("checkpoint_sha256"):
            raise ValueError(f"downstream e{epoch} checkpoint hash mismatch")
        if sha256_file(predictions_path) != metrics.get("predictions_sha256"):
            raise ValueError(f"downstream e{epoch} prediction hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, value in {
            "phase": 5,
            "purpose": "framework_downstream_probe",
            "walk": config.walk,
            "task": config.task,
            "completed_epoch": epoch,
            **expected_hashes,
        }.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"downstream e{epoch} checkpoint {key} mismatch")
        model = build_head(config.task)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        replay = predict(model, data["X_test"], config.batch_size, torch.device("cpu"))
        with np.load(predictions_path, allow_pickle=False) as saved:
            for field, expected in metadata.items():
                if not np.array_equal(saved[field], expected):
                    raise ValueError(f"downstream e{epoch} prediction identity mismatch: {field}")
            stored = (
                saved["prediction_probability_points"].reshape(-1, 1)
                if config.task == "regression"
                else saved["logits"]
            )
            if not np.allclose(replay, stored, rtol=replay_tolerance, atol=replay_tolerance):
                maximum = float(np.max(np.abs(replay.astype(np.float64) - stored.astype(np.float64))))
                raise ValueError(f"downstream e{epoch} CPU prediction replay mismatch: max_abs={maximum}")
        with np.load(history_path, allow_pickle=False) as history:
            if len(history["epochs"]) != epoch or int(history["epochs"][-1]) != epoch:
                raise ValueError(f"downstream e{epoch} history mismatch")
    complete = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("snapshots") != list(SNAPSHOT_EPOCHS):
        raise ValueError("downstream completion marker is invalid")
    principal = snapshots[-1]
    result = {
        "valid": True,
        "walk": config.walk,
        "task": config.task,
        "snapshots": list(SNAPSHOT_EPOCHS),
        "epoch50_checkpoint_sha256": principal["checkpoint_sha256"],
        "epoch50_predictions_sha256": principal["predictions_sha256"],
    }
    if config.task == "regression":
        result.update({key: principal[key] for key in ("mae", "rmse", "pearson", "spearman")})
    else:
        result.update({key: principal[key] for key in ("macro_f1", "balanced_accuracy", "accuracy")})
    return result
