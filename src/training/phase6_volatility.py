"""Frozen Phase 6 future-realised-variance neural training and replay."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from torch.utils.data import DataLoader, TensorDataset

from baselines.mlp_baseline.mlp_model import RawOHLCVMLP
from data_processing.phase5_walks import sha256_arrays, sha256_file
from data_processing.phase6_volatility_labels import validate_volatility_label_bundle_files
from features.phase5_features import BRANCH_ORDER, array_sha256
from features.phase6_volatility_features import validate_phase6_volatility_feature_bundle
from tasks.volatility_prediction import VolatilityRegressor
from training.phase5_baselines import RawOHLCVLSTM
from training.phase5_downstream import environment_manifest, resolve_device, set_seed
from training.phase5_encoder import write_json


MODELS = ("framework_h0", "raw_ohlcv_mlp", "raw_lstm")
SNAPSHOT_EPOCHS = (5, 15, 50)
TARGET_MULTIPLIER = 10_000.0
SCALER_FLOOR = 1e-8
SCALER_CLIP = 10.0


@dataclass(frozen=True)
class Phase6VolatilityConfig:
    model: str
    walk: int
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    hidden_dim: int = 128
    target_multiplier: float = TARGET_MULTIPLIER
    smooth_l1_beta: float = 1.0
    gradient_clip_norm: float = 5.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("Phase 6 volatility requires 50 epochs and 5/15/50 snapshots")
        if self.seed != 0 or self.batch_size != 512 or self.learning_rate != 1e-4:
            raise ValueError("Phase 6 volatility is frozen to seed0/batch512/lr1e-4")
        if self.weight_decay != 0.0 or self.hidden_dim != 128:
            raise ValueError("Phase 6 volatility weight decay and head width are frozen")
        if (
            self.target_multiplier != TARGET_MULTIPLIER
            or self.smooth_l1_beta != 1.0
            or self.gradient_clip_norm != 5.0
        ):
            raise ValueError("Phase 6 volatility scaling/loss/gradient clip contract changed")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["snapshot_epochs"] = list(self.snapshot_epochs)
        return result


class RawVolatilityModel(nn.Module):
    """Frozen raw MLP or stacked-LSTM encoder with the common positive head."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        if model_name == "raw_ohlcv_mlp":
            self.encoder: nn.Module = RawOHLCVMLP(
                seq_len=64,
                n_features=5,
                hidden_dims=[512, 512, 256, 256, 128],
                output_dim=128,
                dropout=0.1,
            )
            output_dim = 128
        elif model_name == "raw_lstm":
            self.encoder = RawOHLCVLSTM(n_features=5)
            output_dim = 20
        else:
            raise ValueError("RawVolatilityModel requires a raw baseline model name")
        self.head = VolatilityRegressor(output_dim, hidden_dim=128)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(values))


def build_model(config: Phase6VolatilityConfig, *, feature_dim: int = 445) -> nn.Module:
    if config.model == "framework_h0":
        return VolatilityRegressor(feature_dim, hidden_dim=config.hidden_dim)
    return RawVolatilityModel(config.model)


def smoke_test_volatility_models() -> dict[str, Any]:
    """Run CPU forward/backward checks for all frozen head widths and raw models."""

    set_seed(0)
    criterion = nn.SmoothL1Loss(beta=1.0)
    results: dict[str, Any] = {}
    for width in (445, 573):
        model = VolatilityRegressor(width, hidden_dim=128)
        output = model(torch.randn(8, width))
        loss = criterion(output, torch.rand(8, 1) * 10.0)
        loss.backward()
        if output.shape != (8, 1) or not torch.isfinite(loss) or torch.any(output < 0.0):
            raise RuntimeError(f"framework volatility smoke test failed at width {width}")
        results[f"framework_width_{width}"] = {
            "output_shape": list(output.shape),
            "loss": float(loss.detach()),
        }
    for model_name in ("raw_ohlcv_mlp", "raw_lstm"):
        config = Phase6VolatilityConfig(model=model_name, walk=1, device="cpu")
        model = build_model(config)
        output = model(torch.randn(8, 64, 5))
        loss = criterion(output, torch.rand(8, 1) * 10.0)
        loss.backward()
        if output.shape != (8, 1) or not torch.isfinite(loss) or torch.any(output < 0.0):
            raise RuntimeError(f"{model_name} volatility smoke test failed")
        results[model_name] = {"output_shape": list(output.shape), "loss": float(loss.detach())}
    return {"valid": True, "models": results}


def _fit_standardizer(train: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(train, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("feature standardizer requires a finite matrix")
    mean = values.mean(axis=0)
    raw_std = values.std(axis=0, ddof=0)
    scale = np.where(raw_std < SCALER_FLOOR, 1.0, raw_std)
    return {"mean": mean, "raw_std": raw_std, "scale": scale}


def _apply_standardizer(values: np.ndarray, scaler: Mapping[str, np.ndarray]) -> np.ndarray:
    result = (np.asarray(values, dtype=np.float64) - scaler["mean"]) / scaler["scale"]
    result = np.clip(result, -SCALER_CLIP, SCALER_CLIP).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("feature standardization produced non-finite values")
    return result


def _concat_features(feature_path: Path, split: str) -> np.ndarray:
    with np.load(feature_path, allow_pickle=False) as stored:
        result = np.concatenate(
            [np.asarray(stored[f"{split}_{branch}"], dtype=np.float32) for branch in BRANCH_ORDER],
            axis=1,
        )
    if result.shape[1] != 445 or not np.isfinite(result).all():
        raise ValueError("canonical Phase 6 H0 features must be finite [N,445]")
    return result


def load_volatility_data(
    label_path: Path, feature_path: Path | None, model_name: str
) -> dict[str, Any]:
    label_validation = validate_volatility_label_bundle_files(label_path)
    if model_name == "framework_h0":
        if feature_path is None:
            raise ValueError("framework_h0 requires a feature bundle")
        feature_validation = validate_phase6_volatility_feature_bundle(feature_path)
        if int(feature_validation["walk"]) != int(label_validation["walk"]):
            raise ValueError("label/feature walk mismatch")
        raw_train = _concat_features(feature_path, "train")
        raw_test = _concat_features(feature_path, "test")
        scaler = _fit_standardizer(raw_train)
        X_train = _apply_standardizer(raw_train, scaler)
        X_test = _apply_standardizer(raw_test, scaler)
    else:
        scaler = None
        with np.load(label_path, allow_pickle=False) as labels:
            X_train = np.asarray(labels["train_sequences"], dtype=np.float32)
            X_test = np.asarray(labels["test_sequences"], dtype=np.float32)
    with np.load(label_path, allow_pickle=False) as labels:
        result: dict[str, Any] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": np.asarray(labels["train_realised_variance"], dtype=np.float64),
            "y_test": np.asarray(labels["test_realised_variance"], dtype=np.float64),
            "scaler": scaler,
            "test_raw_sequences": np.asarray(labels["test_raw_sequences"], dtype=np.float32),
        }
        metadata_fields = (
            "condition_ids",
            "window_start_ns",
            "decision_date_ns",
            "decision_availability_ns",
            "target_start_ns",
            "target_end_ns",
            "target_availability_ns",
            "context_imputed_rows",
            "lifecycle_fraction",
            "lifecycle_stage",
            "categories",
            "event_families",
            "selection_ranks",
            "activity_change_count_24h",
            "future_update_count",
            "shared_row_indices",
        )
        for split in ("train", "test"):
            for field in metadata_fields:
                result[f"{split}_{field}"] = np.asarray(labels[f"{split}_{field}"])
    if len(X_train) != len(result["y_train"]) or len(X_test) != len(result["y_test"]):
        raise ValueError("Phase 6 volatility inputs and targets are not aligned")
    return result


def historical_persistence(raw_sequences: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_sequences, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (64, 5):
        raise ValueError("historical persistence requires [N,64,5] raw contexts")
    closes = values[:, -9:, 3]
    return np.square(np.diff(closes, axis=1)).sum(axis=1)


def volatility_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != y.shape or not len(y) or not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise ValueError("invalid volatility prediction/target arrays")
    if np.any(pred < 0.0):
        raise ValueError("volatility predictions must be nonnegative")
    error = pred - y

    def corr(left: np.ndarray, right: np.ndarray) -> float:
        if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])

    return {
        "count": int(len(y)),
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(np.square(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson": corr(pred, y),
        "spearman": corr(rankdata(pred), rankdata(y)),
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
    }


@torch.no_grad()
def _predict(model: nn.Module, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    chunks = [model(batch.to(device)).cpu().numpy() for batch in tensor.split(batch_size)]
    result = np.concatenate(chunks).reshape(-1).astype(np.float64)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise FloatingPointError("model emitted an invalid volatility prediction")
    return result


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _model_spec(config: Phase6VolatilityConfig) -> dict[str, Any]:
    encoder: dict[str, Any]
    if config.model == "framework_h0":
        encoder = {"type": "frozen_five_branch_concat", "input_dim": 445}
    elif config.model == "raw_ohlcv_mlp":
        encoder = {"type": "raw_ohlcv_mlp", "hidden_layers": [512, 512, 256, 256, 128]}
    else:
        encoder = {"type": "raw_lstm", "hidden_layers": [50, 30, 20]}
    return {
        "encoder": encoder,
        "head": {
            "hidden_layers": [128, 64],
            "activation": "GELU",
            "output": "Softplus(beta=1,threshold=20)",
        },
    }


def run_volatility_training(
    label_path: Path,
    feature_path: Path | None,
    run_root: Path,
    config: Phase6VolatilityConfig,
) -> list[dict[str, Any]]:
    """Fit one uninterrupted trajectory, then evaluate only frozen snapshots."""

    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite Phase 6 volatility run: {run_root}")
    data = load_volatility_data(label_path, feature_path, config.model)
    device = resolve_device(config.device)
    set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config).to(device)
    criterion = nn.SmoothL1Loss(beta=config.smooth_l1_beta)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    y_scaled = (data["y_train"] * config.target_multiplier).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(data["X_train"], dtype=np.float32)),
            torch.from_numpy(y_scaled.reshape(-1, 1)),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed),
    )
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "config.json", config.to_dict())
    write_json(run_root / "environment.json", environment_manifest(device))
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    write_json(
        run_root / "architecture_manifest.json",
        {**_model_spec(config), "parameter_count": parameter_count},
    )
    label_manifest = json.loads(Path(f"{label_path}.manifest.json").read_text(encoding="utf-8"))
    write_json(
        run_root / "dataset_manifest.json",
        {
            "phase": 6,
            "task": "future_realised_variance_h8",
            "walk": config.walk,
            "label_path": str(label_path.resolve()),
            "label_sha256": sha256_file(label_path),
            "label_manifest_sha256": sha256_file(Path(f"{label_path}.manifest.json")),
            "feature_path": str(feature_path.resolve()) if feature_path else None,
            "feature_sha256": sha256_file(feature_path) if feature_path else None,
            "train_identity_hash": label_manifest["identity_hashes"]["train"],
            "test_identity_hash": label_manifest["identity_hashes"]["test"],
            "target_hashes": label_manifest["target_hashes"],
            "train_rows": int(len(data["y_train"])),
            "test_rows": int(len(data["y_test"])),
            "evaluation_used_for_fitting_or_selection": False,
        },
    )
    if data["scaler"] is not None:
        scaler_path = run_root / "feature_standardizer.npz"
        _atomic_savez(scaler_path, data["scaler"])
        write_json(
            run_root / "feature_standardizer.manifest.json",
            {
                "fit_population": "Phase 6 volatility training rows only",
                "evaluation_used_for_fit": False,
                "std_floor": SCALER_FLOOR,
                "clip": [-SCALER_CLIP, SCALER_CLIP],
                "array_hashes": {
                    name: array_sha256(value) for name, value in data["scaler"].items()
                },
                "sha256": sha256_file(scaler_path),
            },
        )

    losses: list[float] = []
    epoch_seconds: list[float] = []
    checkpoints: list[Path] = []
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        total = 0.0
        rows = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite volatility loss at epoch {epoch}")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"non-finite volatility gradient at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total += float(loss.detach()) * len(x_batch)
            rows += len(x_batch)
        losses.append(total / rows)
        epoch_seconds.append(time.perf_counter() - started)
        print(
            f"walk={config.walk} model={config.model} epoch={epoch} "
            f"loss={losses[-1]:.8f} seconds={epoch_seconds[-1]:.2f}",
            flush=True,
        )
        if epoch in config.snapshot_epochs:
            snapshot = run_root / f"e{epoch}"
            snapshot.mkdir()
            checkpoint = snapshot / "checkpoint.pth"
            torch.save(
                {
                    "phase": 6,
                    "purpose": "future_realised_variance_h8",
                    "walk": config.walk,
                    "model": config.model,
                    "completed_epoch": epoch,
                    "config": config.to_dict(),
                    "model_spec": _model_spec(config),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "label_sha256": sha256_file(label_path),
                    "feature_sha256": sha256_file(feature_path) if feature_path else None,
                    "parameter_count": parameter_count,
                },
                checkpoint,
            )
            _atomic_savez(
                snapshot / "history.npz",
                {
                    "epochs": np.arange(1, epoch + 1, dtype=np.int32),
                    "train_loss": np.asarray(losses, dtype=np.float64),
                    "epoch_seconds": np.asarray(epoch_seconds, dtype=np.float64),
                },
            )
            checkpoints.append(checkpoint)

    y_test = np.asarray(data["y_test"], dtype=np.float64)
    median_prediction = np.full_like(y_test, np.median(data["y_train"]))
    persistence = historical_persistence(data["test_raw_sequences"])
    reference_metrics = {
        "zero": volatility_metrics(np.zeros_like(y_test), y_test),
        "training_median": volatility_metrics(median_prediction, y_test),
        "historical_persistence": volatility_metrics(persistence, y_test),
    }
    snapshots: list[dict[str, Any]] = []
    metadata_fields = (
        "condition_ids",
        "window_start_ns",
        "decision_date_ns",
        "decision_availability_ns",
        "target_start_ns",
        "target_end_ns",
        "target_availability_ns",
        "context_imputed_rows",
        "lifecycle_fraction",
        "lifecycle_stage",
        "categories",
        "event_families",
        "selection_ranks",
        "activity_change_count_24h",
        "future_update_count",
        "shared_row_indices",
    )
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        evaluated = build_model(config).to(device)
        evaluated.load_state_dict(checkpoint["model_state_dict"], strict=True)
        inference_started = time.perf_counter()
        prediction_scaled = _predict(evaluated, data["X_test"], config.batch_size, device)
        inference_seconds = time.perf_counter() - inference_started
        prediction = prediction_scaled / config.target_multiplier
        metrics = volatility_metrics(prediction, y_test)
        snapshot = checkpoint_path.parent
        prediction_arrays = {
            "prediction_scaled_squared_probability_points": prediction_scaled,
            "prediction_realised_variance": prediction,
            "target_realised_variance": y_test,
            "zero_reference": np.zeros_like(y_test),
            "training_median_reference": median_prediction,
            "historical_persistence_reference": persistence,
            **{field: np.asarray(data[f"test_{field}"]) for field in metadata_fields},
        }
        _atomic_savez(snapshot / "predictions.npz", prediction_arrays)
        row = {
            "phase": 6,
            "walk": config.walk,
            "model": config.model,
            "epoch": int(checkpoint["completed_epoch"]),
            "seed": config.seed,
            "train_loss": float(losses[int(checkpoint["completed_epoch"]) - 1]),
            "elapsed_training_seconds": float(
                sum(epoch_seconds[: int(checkpoint["completed_epoch"])])
            ),
            "inference_seconds": inference_seconds,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "parameter_count": parameter_count,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(snapshot / "predictions.npz"),
            **{name: metrics[name] for name in ("mae", "rmse", "mse", "pearson", "spearman")},
        }
        write_json(
            snapshot / "metrics.json",
            {
                **row,
                "model_metrics": metrics,
                "references": reference_metrics,
                "unit_contract": {
                    "optimization": "10000 * raw future realised variance",
                    "reporting": "raw future realised variance",
                    "inverse_applied": True,
                },
            },
        )
        snapshots.append(row)
    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    write_json(
        run_root / "training_complete.json",
        {
            "complete": True,
            "phase": 6,
            "walk": config.walk,
            "model": config.model,
            "snapshots": list(config.snapshot_epochs),
            "principal_epoch": 50,
            "evaluation_used_for_selection": False,
        },
    )
    return snapshots


def validate_volatility_run(
    label_path: Path,
    feature_path: Path | None,
    run_root: Path,
) -> dict[str, Any]:
    """Replay every saved evaluation prediction from its checkpoint on CPU."""

    config_payload = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    config_payload["snapshot_epochs"] = tuple(config_payload["snapshot_epochs"])
    config = Phase6VolatilityConfig(**{**config_payload, "device": "cpu"})
    data = load_volatility_data(label_path, feature_path, config.model)
    if config.model == "framework_h0":
        scaler_path = run_root / "feature_standardizer.npz"
        scaler_manifest = json.loads(
            (run_root / "feature_standardizer.manifest.json").read_text(encoding="utf-8")
        )
        if sha256_file(scaler_path) != scaler_manifest["sha256"]:
            raise ValueError("volatility feature standardizer hash mismatch")
        with np.load(scaler_path, allow_pickle=False) as stored:
            for name in ("mean", "raw_std", "scale"):
                if not np.array_equal(stored[name], data["scaler"][name]):
                    raise ValueError(f"volatility standardizer replay mismatch: {name}")
    rows = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))[
        "snapshots"
    ]
    if [int(row["epoch"]) for row in rows] != list(SNAPSHOT_EPOCHS):
        raise ValueError("volatility snapshot matrix is incomplete")
    replayed = []
    for row in rows:
        epoch = int(row["epoch"])
        snapshot = run_root / f"e{epoch}"
        checkpoint_path = snapshot / "checkpoint.pth"
        predictions_path = snapshot / "predictions.npz"
        metrics_path = snapshot / "metrics.json"
        if sha256_file(checkpoint_path) != row["checkpoint_sha256"]:
            raise ValueError(f"e{epoch} volatility checkpoint hash mismatch")
        if sha256_file(predictions_path) != row["predictions_sha256"]:
            raise ValueError(f"e{epoch} volatility prediction hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_model(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        replay_scaled = _predict(model, data["X_test"], config.batch_size, torch.device("cpu"))
        with np.load(predictions_path, allow_pickle=False) as stored:
            saved_scaled = np.asarray(
                stored["prediction_scaled_squared_probability_points"], dtype=np.float64
            )
            for field in (
                "condition_ids",
                "window_start_ns",
                "decision_date_ns",
                "decision_availability_ns",
                "shared_row_indices",
            ):
                if not np.array_equal(stored[field], data[f"test_{field}"]):
                    raise ValueError(f"e{epoch} prediction identity mismatch: {field}")
        # cuDNN and the CPU LSTM backend accumulate small recurrent reduction
        # differences.  The 6e-2 tolerance is in the 10,000x optimization
        # unit, i.e. 6e-6 in raw realised-variance units. CUDA replay remains
        # represented by the immutable saved prediction hash.
        tolerance = 6e-2 if config.model == "raw_lstm" else 2e-5
        if not np.allclose(replay_scaled, saved_scaled, rtol=1e-5, atol=tolerance):
            raise ValueError(f"e{epoch} CPU prediction replay mismatch")
        replay_metrics = volatility_metrics(replay_scaled / config.target_multiplier, data["y_test"])
        saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["model_metrics"]
        for name in ("mae", "rmse", "mse", "pearson", "spearman"):
            left, right = float(replay_metrics[name]), float(saved_metrics[name])
            if math.isnan(left) and math.isnan(right):
                continue
            metric_atol = 1e-10
            if config.model == "raw_lstm":
                metric_atol = 1e-4 if name in ("pearson", "spearman") else 5e-8
            if not np.isclose(left, right, rtol=1e-5, atol=metric_atol):
                raise ValueError(f"e{epoch} metric replay mismatch: {name}")
        replayed.append(epoch)
    complete = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("principal_epoch") != 50:
        raise ValueError("volatility completion marker is invalid")
    return {
        "valid": True,
        "walk": config.walk,
        "model": config.model,
        "snapshots": replayed,
        "cpu_prediction_replay": True,
        "test_identity_hash": sha256_arrays(
            data["test_condition_ids"], data["test_decision_date_ns"]
        ),
    }
