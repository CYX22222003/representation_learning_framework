"""Phase 5 eight-hour absolute future-price exploratory probe."""

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

from training.phase5_downstream import (
    _atomic_savez,
    environment_manifest,
    predict,
    regression_metrics,
    resolve_device,
    set_seed,
)
from training.phase5_encoder import sha256_file, write_json
from training.phase5_regression_sensitivities import (
    _metadata,
    load_sensitivity_data,
    sensitivity_breakdowns,
)


SNAPSHOT_EPOCHS = (5, 15, 50)


class AbsolutePriceRegressor(nn.Module):
    """Canonical shallow probe with a probability-bounded output."""

    def __init__(self, input_dim: int = 445, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


@dataclass(frozen=True)
class AbsolutePriceConfig:
    walk: int
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("absolute-price sensitivity requires 50 epochs and 5/15/50 snapshots")
        if self.seed != 0 or self.batch_size != 512 or self.learning_rate != 1e-4:
            raise ValueError("absolute-price sensitivity is frozen to seed0/batch512/lr1e-4")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["snapshot_epochs"] = list(self.snapshot_epochs)
        return result


def _masked_metrics(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, Any] | None:
    selected = np.asarray(mask, dtype=bool)
    return regression_metrics(prediction[selected], target[selected]) if np.any(selected) else None


def price_breakdowns(
    prediction: np.ndarray,
    target: np.ndarray,
    metadata: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pred = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(target, dtype=np.float64)
    current = np.asarray(metadata["current_close"], dtype=np.float64)
    groups: dict[str, Any] = {
        "overall": regression_metrics(pred, actual),
        "context_observed_only": _masked_metrics(
            pred, actual, np.asarray(metadata["context_imputed_rows"]) == 0
        ),
        "context_has_imputation": _masked_metrics(
            pred, actual, np.asarray(metadata["context_imputed_rows"]) > 0
        ),
        "starting_price_bands": {
            "p_le_0.01": _masked_metrics(pred, actual, current <= 0.01),
            "p_0.01_to_0.10": _masked_metrics(
                pred, actual, (current > 0.01) & (current <= 0.10)
            ),
            "p_0.10_to_0.90": _masked_metrics(
                pred, actual, (current > 0.10) & (current < 0.90)
            ),
            "p_ge_0.90": _masked_metrics(pred, actual, current >= 0.90),
        },
    }
    stage = np.asarray(metadata["lifecycle_stage"])
    groups["lifecycle"] = {
        name: _masked_metrics(pred, actual, stage == code)
        for code, name in ((-1, "unknown"), (0, "early"), (1, "middle"), (2, "late"))
    }
    condition_ids = np.asarray(metadata["condition_ids"])
    per_contract: dict[str, Any] = {}
    for condition_id in np.unique(condition_ids):
        mask = condition_ids == condition_id
        per_contract[str(condition_id)] = regression_metrics(pred[mask], actual[mask])
    metric_names = ("mae", "mse", "rmse", "pearson", "spearman")
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


def relative_skill(model: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    return {
        metric: float(1.0 - float(model[metric]) / float(reference[metric]))
        for metric in ("mae", "mse", "rmse")
        if float(reference[metric]) > 0.0
    }


def run_absolute_price(
    dataset_path: Path,
    feature_path: Path,
    scaler_path: Path,
    run_root: Path,
    config: AbsolutePriceConfig,
) -> list[dict[str, Any]]:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite absolute-price run: {run_root}")
    data = load_sensitivity_data(dataset_path, feature_path, scaler_path, "raw_delta_h8")
    y_train = np.asarray(data["train_target_close"], dtype=np.float32)
    y_test = np.asarray(data["test_target_close"], dtype=np.float64)
    if np.any((y_train < 0.0) | (y_train > 1.0)) or np.any((y_test < 0.0) | (y_test > 1.0)):
        raise ValueError("absolute-price targets must lie in [0,1]")
    device = resolve_device(config.device)
    set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data["X_train"]),
            torch.from_numpy(y_train.reshape(-1, 1)),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = AbsolutePriceRegressor().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "config.json", config.to_dict())
    write_json(run_root / "environment.json", environment_manifest(device))
    write_json(
        run_root / "architecture_manifest.json",
        {
            "input_dim": 445,
            "hidden_layers": [128, 64],
            "activation": "GELU",
            "dropout": 0.1,
            "output": "scalar sigmoid probability",
            "explicit_current_close_skip": False,
            "parameter_count": parameter_count,
        },
    )
    write_json(
        run_root / "dataset_manifest.json",
        {
            "phase": 5,
            "purpose": "exploratory_absolute_price_h8",
            "walk": config.walk,
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "feature_path": str(feature_path),
            "feature_sha256": sha256_file(feature_path),
            "feature_scaler_path": str(scaler_path),
            "feature_scaler_sha256": sha256_file(scaler_path),
            "target": "observed close at decision + 8 hours",
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
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
                raise FloatingPointError(f"non-finite absolute-price loss at epoch {epoch}")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"non-finite absolute-price gradient at epoch {epoch}")
            optimizer.step()
            total += float(loss.detach()) * len(x_batch)
            count += len(x_batch)
        losses.append(total / count)
        seconds.append(time.perf_counter() - started)
        print(
            f"walk={config.walk} task=absolute_price_h8 epoch={epoch} "
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
                    "purpose": "exploratory_absolute_price_h8",
                    "walk": config.walk,
                    "completed_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
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
    current = np.asarray(metadata["current_close"], dtype=np.float64)
    realised_delta = y_test - current
    snapshots: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        evaluated = AbsolutePriceRegressor().to(device)
        evaluated.load_state_dict(checkpoint["model_state_dict"], strict=True)
        inference_started = time.perf_counter()
        predicted_price = predict(
            evaluated, data["X_test"], config.batch_size, device
        ).reshape(-1).astype(np.float64)
        inference_seconds = time.perf_counter() - inference_started
        if np.any((predicted_price < 0.0) | (predicted_price > 1.0)):
            raise ValueError("sigmoid absolute-price prediction escaped [0,1]")
        predicted_delta = predicted_price - current
        price_metrics, per_contract = price_breakdowns(predicted_price, y_test, metadata)
        persistence_metrics, persistence_per_contract = price_breakdowns(
            current, y_test, metadata
        )
        movement_metrics, movement_per_contract = sensitivity_breakdowns(
            "raw_delta_h8", predicted_delta, realised_delta, metadata
        )
        snapshot = checkpoint_path.parent
        _atomic_savez(
            snapshot / "predictions.npz",
            {
                "prediction_future_price": predicted_price,
                "target_future_price": y_test,
                "current_close": current,
                "prediction_delta_h8": predicted_delta,
                "target_delta_h8": realised_delta,
                **{key: value for key, value in metadata.items() if key != "current_close"},
            },
        )
        write_json(snapshot / "per_contract_price_metrics.json", per_contract)
        write_json(snapshot / "persistence_per_contract_metrics.json", persistence_per_contract)
        write_json(snapshot / "per_contract_implied_movement_metrics.json", movement_per_contract)
        overall = price_metrics["overall"]
        persistence = persistence_metrics["overall"]
        movement = movement_metrics["overall"]
        row = {
            "phase": 5,
            "task": "absolute_price_h8",
            "walk": config.walk,
            "epoch": int(checkpoint["completed_epoch"]),
            "seed": config.seed,
            "train_loss": losses[int(checkpoint["completed_epoch"]) - 1],
            "price_mae": overall["mae"],
            "price_rmse": overall["rmse"],
            "price_pearson": overall["pearson"],
            "price_spearman": overall["spearman"],
            "persistence_mae": persistence["mae"],
            "persistence_rmse": persistence["rmse"],
            "implied_delta_pearson": movement["pearson"],
            "implied_delta_spearman": movement["spearman"],
            "implied_delta_sign_agreement": movement["sign_agreement"],
            "inference_seconds": inference_seconds,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(snapshot / "predictions.npz"),
        }
        write_json(
            snapshot / "metrics.json",
            {
                **row,
                "price": price_metrics,
                "persistence_reference": persistence_metrics,
                "persistence_relative_skill": relative_skill(overall, persistence),
                "implied_movement": movement_metrics,
                "prediction_range": {
                    "minimum": float(predicted_price.min()),
                    "maximum": float(predicted_price.max()),
                    "mean": float(predicted_price.mean()),
                    "std": float(predicted_price.std()),
                },
            },
        )
        snapshots.append(row)
    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    write_json(
        run_root / "training_complete.json",
        {
            "complete": True,
            "phase": 5,
            "task": "absolute_price_h8",
            "walk": config.walk,
            "snapshots": list(config.snapshot_epochs),
            "principal_epoch": 50,
            "selection_rule": "epoch 50 predeclared; no evaluation-driven selection",
        },
    )
    return snapshots


def validate_absolute_price_run(
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
        "architecture_manifest.json",
        "dataset_manifest.json",
        "sweep_metrics.json",
        "training_complete.json",
    }
    missing = sorted(name for name in required if not (run_root / name).is_file())
    if missing:
        raise ValueError(f"absolute-price run is missing artifacts: {missing}")
    payload = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
    config = AbsolutePriceConfig(**payload)
    data = load_sensitivity_data(dataset_path, feature_path, scaler_path, "raw_delta_h8")
    manifest = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    expected_hashes = {
        "dataset_sha256": sha256_file(dataset_path),
        "feature_sha256": sha256_file(feature_path),
        "feature_scaler_sha256": sha256_file(scaler_path),
    }
    for key, value in expected_hashes.items():
        if manifest.get(key) != value:
            raise ValueError(f"absolute-price manifest {key} mismatch")
    rows = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))["snapshots"]
    if [int(row["epoch"]) for row in rows] != list(SNAPSHOT_EPOCHS):
        raise ValueError("absolute-price snapshot matrix is incomplete")
    metadata = _metadata(data)
    for row in rows:
        epoch = int(row["epoch"])
        snapshot = run_root / f"e{epoch}"
        checkpoint_path = snapshot / "checkpoint.pth"
        predictions_path = snapshot / "predictions.npz"
        history_path = snapshot / "history.npz"
        metrics_path = snapshot / "metrics.json"
        if not all(path.is_file() for path in (checkpoint_path, predictions_path, history_path, metrics_path)):
            raise ValueError(f"absolute-price e{epoch} artifacts are incomplete")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if sha256_file(checkpoint_path) != metrics["checkpoint_sha256"]:
            raise ValueError(f"absolute-price e{epoch} checkpoint hash mismatch")
        if sha256_file(predictions_path) != metrics["predictions_sha256"]:
            raise ValueError(f"absolute-price e{epoch} prediction hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = AbsolutePriceRegressor()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        replay = predict(model, data["X_test"], config.batch_size, torch.device("cpu")).reshape(-1)
        with np.load(predictions_path, allow_pickle=False) as saved:
            if not np.allclose(
                replay,
                saved["prediction_future_price"],
                rtol=tolerance,
                atol=tolerance,
            ):
                raise ValueError(f"absolute-price e{epoch} CPU prediction replay mismatch")
            if not np.array_equal(saved["target_future_price"], data["test_target_close"]):
                raise ValueError(f"absolute-price e{epoch} target mismatch")
            for key, expected in metadata.items():
                saved_key = "target_future_price" if key == "target_close" else key
                if key == "target_close":
                    continue
                if not np.array_equal(saved[saved_key], expected):
                    raise ValueError(f"absolute-price e{epoch} identity mismatch: {key}")
        with np.load(history_path, allow_pickle=False) as history:
            if len(history["epochs"]) != epoch or int(history["epochs"][-1]) != epoch:
                raise ValueError(f"absolute-price e{epoch} history mismatch")
    complete = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("principal_epoch") != 50:
        raise ValueError("absolute-price completion marker is invalid")
    principal = rows[-1]
    return {
        "valid": True,
        "walk": config.walk,
        "snapshots": list(SNAPSHOT_EPOCHS),
        **{
            key: principal[key]
            for key in (
                "price_mae",
                "price_rmse",
                "price_pearson",
                "price_spearman",
                "persistence_mae",
                "implied_delta_pearson",
                "implied_delta_spearman",
            )
        },
    }
