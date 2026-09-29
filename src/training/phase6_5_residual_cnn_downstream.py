"""Price-only downstream probes for Phase 6.5D residual-CNN features."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_processing.phase5_walks import sha256_arrays, sha256_file
from features.phase5_features import IDENTITY_FIELDS, array_sha256
from features.phase6_5_residual_cnn_features import (
    CONFIG_BRANCHES,
    CONFIG_DIMS,
    TASK,
    load_residual_cnn_configuration,
    validate_residual_cnn_feature_store,
)
from training.phase5_absolute_price import AbsolutePriceRegressor
from training.phase5_downstream import environment_manifest, resolve_device, set_seed
from training.phase5_encoder import write_json
from training.phase6_encoder_variant_downstream import (
    _assert_metric_replay,
    _atomic_savez,
    _evaluate,
    _predict,
    apply_standardizer,
    fit_standardizer,
    project_paths_equal,
)


CONFIGURATIONS = tuple(name for name in CONFIG_BRANCHES if name != "H0")
SNAPSHOT_EPOCHS = (5, 15, 50)


@dataclass(frozen=True)
class Phase65ResidualCNNDownstreamConfig:
    configuration: str
    walk: int
    task: str = TASK
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    hidden_dim: int = 128
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.task != TASK or self.configuration not in CONFIGURATIONS:
            raise ValueError("invalid Phase 6.5D downstream task/configuration")
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("Phase 6.5D downstream requires 50 epochs and 5/15/50 snapshots")
        if self.seed != 0 or self.batch_size != 512 or self.learning_rate != 1e-4:
            raise ValueError("Phase 6.5D downstream recipe is frozen")
        if self.weight_decay != 0.0 or self.hidden_dim != 128:
            raise ValueError("Phase 6.5D downstream head contract changed")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot_epochs"] = list(self.snapshot_epochs)
        return payload


def _load_price_data(
    dataset_path: Path,
    feature_path: Path,
    config: Phase65ResidualCNNDownstreamConfig,
) -> dict[str, Any]:
    validation = validate_residual_cnn_feature_store(feature_path)
    if validation["task"] != TASK or validation["walk"] != config.walk:
        raise ValueError("Phase 6.5D downstream feature task/walk mismatch")
    manifest = json.loads(Path(f"{feature_path}.manifest.json").read_text(encoding="utf-8"))
    if not project_paths_equal(Path(manifest["dataset_path"]), dataset_path):
        raise ValueError("Phase 6.5D downstream dataset path differs from feature source")
    if manifest["dataset_sha256"] != sha256_file(dataset_path):
        raise ValueError("Phase 6.5D downstream dataset hash differs from feature source")
    train_raw, test_raw = load_residual_cnn_configuration(
        feature_path, config.configuration
    )
    scaler = fit_standardizer(train_raw)
    result: dict[str, Any] = {
        "X_train": apply_standardizer(train_raw, scaler),
        "X_test": apply_standardizer(test_raw, scaler),
        "scaler": scaler,
    }
    with np.load(dataset_path, allow_pickle=False) as source:
        for split in ("train", "test"):
            for field in IDENTITY_FIELDS:
                result[f"{split}_{field}"] = np.asarray(source[f"{split}_{field}"])
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
    if len(result["X_train"]) != len(result["y_train"]) or len(result["X_test"]) != len(
        result["y_test"]
    ):
        raise ValueError("Phase 6.5D features and targets are not aligned")
    with np.load(feature_path, allow_pickle=False) as features:
        for split in ("train", "test"):
            for field in IDENTITY_FIELDS:
                key = f"{split}_{field}"
                if not np.array_equal(features[key], result[key]):
                    raise ValueError(f"Phase 6.5D downstream identity mismatch: {key}")
    return result


def build_head(config: Phase65ResidualCNNDownstreamConfig) -> nn.Module:
    return AbsolutePriceRegressor(
        CONFIG_DIMS[config.configuration], hidden_dim=config.hidden_dim
    )


def smoke_test_residual_cnn_downstream() -> dict[str, Any]:
    set_seed(0)
    results = {}
    for configuration in ("HC-SR", "HC-AR"):
        config = Phase65ResidualCNNDownstreamConfig(configuration, 1, device="cpu")
        width = CONFIG_DIMS[configuration]
        model = build_head(config)
        output = model(torch.randn(4, width))
        loss = nn.MSELoss()(output, torch.rand(4, 1))
        loss.backward()
        if output.shape != (4, 1) or not torch.isfinite(loss):
            raise FloatingPointError("Phase 6.5D downstream smoke failed")
        if not bool(((output >= 0.0) & (output <= 1.0)).all()):
            raise RuntimeError("Phase 6.5D future-price output is not sigmoid bounded")
        results[configuration] = {
            "input_dim": width,
            "output_shape": list(output.shape),
            "loss_finite": True,
            "sigmoid_bounded": True,
        }
    return {"valid": True, "heads": results}


def run_residual_cnn_downstream(
    dataset_path: Path,
    feature_path: Path,
    run_root: Path,
    config: Phase65ResidualCNNDownstreamConfig,
) -> list[dict[str, Any]]:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite Phase 6.5D downstream run: {run_root}")
    data = _load_price_data(dataset_path, feature_path, config)
    device = resolve_device(config.device)
    set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    target_tensor = torch.from_numpy(
        np.asarray(data["y_train"], dtype=np.float32).reshape(-1, 1)
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data["X_train"]), target_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = build_head(config).to(device)
    criterion = nn.MSELoss().to(device)
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
            "phase": "6.5D",
            "configuration": config.configuration,
            "branches": list(CONFIG_BRANCHES[config.configuration]),
            "input_dim": CONFIG_DIMS[config.configuration],
            "hidden_layers": [128, 64],
            "task_head": "AbsolutePriceRegressor",
            "output_contract": "scalar sigmoid probability",
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
            "dimension": CONFIG_DIMS[config.configuration],
            "sha256": sha256_file(scaler_path),
            "array_hashes": {
                name: array_sha256(value) for name, value in data["scaler"].items()
            },
        },
    )
    write_json(
        run_root / "dataset_manifest.json",
        {
            "phase": "6.5D",
            "task": TASK,
            "walk": config.walk,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": sha256_file(dataset_path),
            "feature_path": str(feature_path.resolve()),
            "feature_sha256": sha256_file(feature_path),
            "configuration": config.configuration,
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
            "branch_order": list(CONFIG_BRANCHES[config.configuration]),
            "input_dim": CONFIG_DIMS[config.configuration],
        },
    )
    write_json(
        run_root / "training_contract.json",
        {
            "optimizer": "Adam",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "loss": "mean squared error in raw future-probability units",
            "feature_scaler_fit_population": "task training rows only",
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
                raise FloatingPointError("non-finite Phase 6.5D downstream loss")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError("non-finite Phase 6.5D downstream gradient")
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
                    "phase": "6.5D",
                    "purpose": "residual_cnn_future_price_downstream",
                    "task": TASK,
                    "configuration": config.configuration,
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
        evaluated = build_head(config).to(device)
        evaluated.load_state_dict(checkpoint["model_state_dict"], strict=True)
        inference_started = time.perf_counter()
        output = _predict(evaluated, data["X_test"], config.batch_size, device)
        inference_seconds = time.perf_counter() - inference_started
        metrics, arrays = _evaluate(config, output, data)
        snapshot = checkpoint_path.parent
        _atomic_savez(snapshot / "predictions.npz", arrays)
        row = {
            "epoch": int(checkpoint["completed_epoch"]),
            "train_loss": losses[int(checkpoint["completed_epoch"]) - 1],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(snapshot / "predictions.npz"),
            "elapsed_training_seconds": float(
                sum(seconds[: int(checkpoint["completed_epoch"])])
            ),
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


def validate_residual_cnn_downstream(
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
        raise ValueError(f"Phase 6.5D downstream run is missing artifacts: {missing}")
    payload = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
    config = Phase65ResidualCNNDownstreamConfig(**{**payload, "device": "cpu"})
    data = _load_price_data(dataset_path, feature_path, config)
    source = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if source.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("Phase 6.5D downstream dataset hash mismatch")
    if source.get("feature_sha256") != sha256_file(feature_path):
        raise ValueError("Phase 6.5D downstream feature hash mismatch")
    if source.get("branch_order") != list(CONFIG_BRANCHES[config.configuration]):
        raise ValueError("Phase 6.5D downstream branch-order mismatch")
    for split in ("train", "test"):
        identity_hash = sha256_arrays(
            *(data[f"{split}_{field}"] for field in IDENTITY_FIELDS)
        )
        if source.get(f"{split}_identity_hash") != identity_hash:
            raise ValueError(f"Phase 6.5D downstream {split} identity hash mismatch")
        if source.get(f"{split}_target_hash") != array_sha256(data[f"y_{split}"]):
            raise ValueError(f"Phase 6.5D downstream {split} target hash mismatch")
    architecture = json.loads(
        (run_root / "architecture_manifest.json").read_text(encoding="utf-8")
    )
    if (
        architecture.get("phase") != "6.5D"
        or architecture.get("branches") != list(CONFIG_BRANCHES[config.configuration])
        or architecture.get("input_dim") != CONFIG_DIMS[config.configuration]
        or architecture.get("hidden_layers") != [128, 64]
        or architecture.get("task_head") != "AbsolutePriceRegressor"
        or architecture.get("output_contract") != "scalar sigmoid probability"
    ):
        raise ValueError("Phase 6.5D downstream architecture mismatch")
    training_contract = json.loads(
        (run_root / "training_contract.json").read_text(encoding="utf-8")
    )
    if (
        training_contract.get("optimizer") != "Adam"
        or training_contract.get("learning_rate") != config.learning_rate
        or training_contract.get("weight_decay") != 0.0
        or training_contract.get("batch_size") != 512
        or training_contract.get("loss") != "mean squared error in raw future-probability units"
        or training_contract.get("evaluation_used_for_fitting_or_selection") is not False
        or training_contract.get("principal_epoch") != 50
    ):
        raise ValueError("Phase 6.5D downstream training contract mismatch")
    scaler_path = run_root / "feature_standardizer.npz"
    scaler_manifest = json.loads(
        (run_root / "feature_standardizer.manifest.json").read_text(encoding="utf-8")
    )
    if (
        scaler_manifest.get("sha256") != sha256_file(scaler_path)
        or scaler_manifest.get("fit_population") != "task training rows only"
        or scaler_manifest.get("evaluation_used") is not False
    ):
        raise ValueError("Phase 6.5D downstream scaler manifest mismatch")
    with np.load(scaler_path, allow_pickle=False) as saved:
        for name, value in data["scaler"].items():
            if not np.array_equal(saved[name], value):
                raise ValueError(f"Phase 6.5D scaler replay mismatch: {name}")
    rows = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))["snapshots"]
    if [int(row["epoch"]) for row in rows] != list(SNAPSHOT_EPOCHS):
        raise ValueError("Phase 6.5D downstream snapshot matrix is incomplete")
    for row in rows:
        epoch = int(row["epoch"])
        snapshot = run_root / f"e{epoch}"
        checkpoint_path = snapshot / "checkpoint.pth"
        prediction_path = snapshot / "predictions.npz"
        history_path = snapshot / "history.npz"
        metrics_path = snapshot / "metrics.json"
        if not all(
            path.is_file()
            for path in (checkpoint_path, prediction_path, history_path, metrics_path)
        ):
            raise ValueError(f"Phase 6.5D downstream e{epoch} artifacts are incomplete")
        if sha256_file(checkpoint_path) != row["checkpoint_sha256"] or sha256_file(
            prediction_path
        ) != row["predictions_sha256"]:
            raise ValueError(f"Phase 6.5D downstream e{epoch} artifact hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("phase") != "6.5D"
            or checkpoint.get("purpose") != "residual_cnn_future_price_downstream"
            or checkpoint.get("task") != TASK
            or checkpoint.get("configuration") != config.configuration
            or checkpoint.get("walk") != config.walk
            or checkpoint.get("completed_epoch") != epoch
            or checkpoint.get("dataset_sha256") != sha256_file(dataset_path)
            or checkpoint.get("feature_sha256") != sha256_file(feature_path)
            or checkpoint.get("scaler_sha256") != sha256_file(scaler_path)
        ):
            raise ValueError(f"Phase 6.5D downstream e{epoch} checkpoint mismatch")
        with np.load(history_path, allow_pickle=False) as history:
            if set(history.files) != {"epochs", "train_loss", "epoch_seconds"}:
                raise ValueError(f"Phase 6.5D downstream e{epoch} history fields mismatch")
            if len(history["epochs"]) != epoch or int(history["epochs"][-1]) != epoch:
                raise ValueError(f"Phase 6.5D downstream e{epoch} history mismatch")
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            int(metrics_payload.get("epoch", -1)) != epoch
            or metrics_payload.get("checkpoint_sha256") != row["checkpoint_sha256"]
            or metrics_payload.get("predictions_sha256") != row["predictions_sha256"]
        ):
            raise ValueError(f"Phase 6.5D downstream e{epoch} metrics contract mismatch")
        model = build_head(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        replay = _predict(model, data["X_test"], config.batch_size, torch.device("cpu"))
        with np.load(prediction_path, allow_pickle=False) as saved:
            saved_output = np.asarray(saved["prediction_future_price"])
            if not np.allclose(saved_output, replay.reshape(-1), rtol=2e-5, atol=2e-5):
                raise ValueError("Phase 6.5D future-price prediction replay mismatch")
            replay_metrics, expected = _evaluate(config, saved_output, data)
            _assert_metric_replay(metrics_payload["metrics"], replay_metrics)
            if set(saved.files) != set(expected):
                raise ValueError("Phase 6.5D prediction fields mismatch")
            exact_fields = set(IDENTITY_FIELDS) | {
                "target_future_price",
                "current_close",
                "target_delta_h8",
                "last_hour_reversal",
            }
            for field in exact_fields:
                if not np.array_equal(saved[field], expected[field]):
                    raise ValueError(f"Phase 6.5D prediction identity/reference mismatch: {field}")
    complete = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("principal_epoch") != 50:
        raise ValueError("Phase 6.5D downstream completion marker is invalid")
    return {
        "valid": True,
        "task": TASK,
        "configuration": config.configuration,
        "walk": config.walk,
        "snapshots": list(SNAPSHOT_EPOCHS),
    }
