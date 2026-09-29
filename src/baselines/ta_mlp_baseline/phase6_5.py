"""Causal feature preparation and training for the Phase 6.5C TA-MLP matrix."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baselines.ta_mlp_baseline.ta_features import FEATURE_NAMES, N_FEATURES, compute_ta_features
from baselines.ta_mlp_baseline.ta_mlp_model import TAMLPClassifier
from data_processing.phase5_walks import CLASS_NAMES, sha256_arrays, sha256_file, validate_phase5_bundle_files
from features.phase5_features import BRANCH_ORDER
from tasks.phase2_classification.metrics import probabilistic_classification_metrics
from tasks.phase2_classification.protocols import (
    LogitAdjustedCrossEntropy,
    class_priors,
    protocol_sample_indices,
)
from tasks.trend_classification import TrendClassifier
from training.phase5_baselines import Phase5BaselineConfig, Phase5RawBaseline
from training.phase5_downstream import environment_manifest, resolve_device, set_seed
from training.phase5_encoder import write_json


TA_MODELS = ("h0", "raw_mlp", "raw_lstm", "ta_mlp")
TA_PROTOCOLS = ("P2", "P1U")
SNAPSHOT_EPOCHS = (5, 15, 50)
SCALER_FLOOR = 1e-8
SCALER_CLIP = 10.0
HOUR = pd.Timedelta(hours=1)
IDENTITY_FIELDS = (
    "condition_ids",
    "window_start_ns",
    "decision_date_ns",
    "decision_availability_ns",
)
REPORTING_FIELDS = (
    "context_imputed_rows",
    "lifecycle_fraction",
    "lifecycle_stage",
    "categories",
    "event_families",
    "selection_ranks",
    "activity_change_count_24h",
)


@dataclass(frozen=True)
class Phase65TAConfig:
    model: str
    protocol: str
    walk: int
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    logit_adjustment_strength: float = 1.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.model not in TA_MODELS:
            raise ValueError(f"model must be one of {TA_MODELS}")
        if self.protocol not in TA_PROTOCOLS:
            raise ValueError(f"protocol must be one of {TA_PROTOCOLS}")
        if self.protocol == "P1U" and self.model != "ta_mlp":
            raise ValueError("P1U is a TA-MLP-only sensitivity in Phase 6.5C")
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("Phase 6.5C requires 50 epochs and 5/15/50 snapshots")
        if self.seed != 0 or self.logit_adjustment_strength != 1.0:
            raise ValueError("Phase 6.5C is frozen to seed0 and P2 lambda=1")

    @property
    def batch_size(self) -> int:
        return 64 if self.model == "ta_mlp" else 512

    @property
    def learning_rate(self) -> float:
        return 1e-3 if self.model == "ta_mlp" else 1e-4

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["snapshot_epochs"] = list(self.snapshot_epochs)
        result["batch_size"] = self.batch_size
        result["learning_rate"] = self.learning_rate
        result["weight_decay"] = 0.0
        return result


def strict_matrix_entries() -> tuple[tuple[str, str], ...]:
    return (
        ("h0", "P2"),
        ("raw_mlp", "P2"),
        ("raw_lstm", "P2"),
        ("ta_mlp", "P2"),
        ("ta_mlp", "P1U"),
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _segment_ids(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_values("date", kind="stable")
    breaks = ordered["date"].diff().ne(HOUR)
    if len(breaks):
        breaks.iloc[0] = True
    return breaks.cumsum().astype(np.int64)


def compute_walk_ta_features(
    candles: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute all 36 features per contract/gap segment with train-only volume fit."""

    required = {"condition_id", "date", "open", "high", "low", "close", "volume"}
    missing = required.difference(candles.columns)
    if missing:
        raise ValueError(f"candle source is missing columns: {sorted(missing)}")
    frame = candles.copy()
    frame["condition_id"] = frame["condition_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.sort_values(["condition_id", "date"], kind="stable").reset_index(drop=True)
    duplicated = frame.duplicated(["condition_id", "date"], keep=False)
    if duplicated.any():
        raise ValueError("TA source contains duplicate condition/date rows")
    availability = frame["date"] + HOUR
    fit_mask = (frame["date"] >= train_start) & (availability < cutoff)
    fit_volume = frame.loc[fit_mask, "volume"].to_numpy(np.float64)
    if not len(fit_volume) or not np.all(np.isfinite(fit_volume)):
        raise ValueError("TA volume fit population is empty or non-finite")
    volume_mean = float(np.mean(fit_volume))
    volume_std = float(np.std(fit_volume, ddof=0))
    if volume_std < SCALER_FLOOR:
        volume_std = 1.0

    parts: list[pd.DataFrame] = []
    segment_count = 0
    for condition_id, contract in frame.groupby("condition_id", sort=True):
        contract = contract.sort_values("date", kind="stable").copy()
        contract["_segment"] = _segment_ids(contract).to_numpy()
        for segment_id, segment in contract.groupby("_segment", sort=True):
            segment_count += 1
            features = compute_ta_features(
                segment,
                volume_mean=volume_mean,
                volume_std=volume_std,
            )
            if features.empty:
                continue
            features = features.copy()
            features.insert(0, "decision_date_ns", segment.loc[features.index, "date"].astype("int64"))
            features.insert(0, "condition_id", str(condition_id))
            features.insert(2, "gap_segment_id", int(segment_id))
            parts.append(features.reset_index(drop=True))
    if not parts:
        raise ValueError("no finite TA features were produced")
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["condition_id", "decision_date_ns"]).any():
        raise ValueError("TA feature identities are not unique")
    manifest = {
        "feature_names": list(FEATURE_NAMES),
        "feature_count": N_FEATURES,
        "rolling_resets_at_gap_segment": True,
        "volume_fit": {
            "mean": volume_mean,
            "std": volume_std,
            "fit_row_count": int(len(fit_volume)),
            "fit_start_inclusive": train_start.isoformat(),
            "fit_availability_cutoff_exclusive": cutoff.isoformat(),
            "uses_evaluation_rows": False,
        },
        "segment_count": segment_count,
        "available_feature_rows": int(len(result)),
    }
    return result, manifest


def _concat_h0(feature_path: Path, split: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(feature_path, allow_pickle=False) as stored:
        identities = {field: np.asarray(stored[f"{split}_{field}"]) for field in IDENTITY_FIELDS}
        values = np.concatenate(
            [np.asarray(stored[f"{split}_{branch}"], dtype=np.float32) for branch in BRANCH_ORDER],
            axis=1,
        )
    if values.shape[1] != 445 or not np.isfinite(values).all():
        raise ValueError("canonical H0 must be a finite 445-dimensional matrix")
    return values, identities


def build_phase65_ta_store(
    dataset_path: Path,
    canonical_feature_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists() or Path(f"{output_path}.manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite Phase 6.5C TA store: {output_path}")
    validation = validate_phase5_bundle_files(dataset_path)
    dataset_manifest = _load_json(Path(f"{dataset_path}.manifest.json"))
    source = dataset_manifest["source_provenance"]["candles_1h_clean_ffill1"]
    candle_path = Path(source["path"])
    if sha256_file(candle_path) != source["sha256"]:
        raise ValueError("Phase 5 candle source hash mismatch")
    intervals = dataset_manifest["intervals"]
    train_start = pd.Timestamp(intervals["training"]["start_inclusive"])
    cutoff = pd.Timestamp(intervals["training"]["cutoff_exclusive"])
    features, feature_manifest = compute_walk_ta_features(
        pd.read_parquet(candle_path), train_start=train_start, cutoff=cutoff
    )
    lookup = {
        (str(row.condition_id), int(row.decision_date_ns)): row
        for row in features.itertuples(index=False)
    }

    arrays: dict[str, np.ndarray] = {
        "feature_names": np.asarray(FEATURE_NAMES, dtype=np.str_),
    }
    intersection: dict[str, Any] = {}
    with np.load(dataset_path, allow_pickle=False) as dataset:
        for split in ("train", "test"):
            h0, h0_identity = _concat_h0(canonical_feature_path, split)
            for field in IDENTITY_FIELDS:
                source_values = np.asarray(dataset[f"{split}_{field}"])
                if not np.array_equal(source_values, h0_identity[field]):
                    raise ValueError(f"canonical H0 identity mismatch: {split}_{field}")
            contracts = np.asarray(dataset[f"{split}_condition_ids"]).astype(str)
            decisions = np.asarray(dataset[f"{split}_decision_date_ns"], dtype=np.int64)
            available = np.asarray(
                [(condition_id, int(decision)) in lookup for condition_id, decision in zip(contracts, decisions)],
                dtype=bool,
            )
            positions = np.flatnonzero(available).astype(np.int64)
            if not len(positions):
                raise ValueError(f"Phase 6.5C {split} intersection is empty")
            ta = np.stack(
                [
                    np.asarray(
                        [getattr(lookup[(contracts[position], int(decisions[position]))], name) for name in FEATURE_NAMES],
                        dtype=np.float32,
                    )
                    for position in positions
                ]
            )
            if ta.shape[1] != N_FEATURES or not np.isfinite(ta).all():
                raise ValueError(f"invalid Phase 6.5C {split} TA matrix")
            arrays[f"{split}_source_positions"] = positions
            arrays[f"{split}_feature_available"] = available
            arrays[f"{split}_exclusion_reason"] = np.where(
                available, "included", "rolling_warmup_or_nonfinite"
            ).astype(np.str_)
            arrays[f"{split}_ta_features"] = ta
            arrays[f"{split}_h0_features"] = h0[positions]
            arrays[f"{split}_raw_sequences"] = np.asarray(dataset[f"{split}_sequences"], dtype=np.float32)[positions]
            arrays[f"{split}_labels"] = np.asarray(dataset[f"{split}_classification_labels"], dtype=np.int64)[positions]
            for field in (*IDENTITY_FIELDS, *REPORTING_FIELDS):
                arrays[f"{split}_{field}"] = np.asarray(dataset[f"{split}_{field}"])[positions]
            intersection[split] = {
                "source_rows": int(len(available)),
                "included_rows": int(len(positions)),
                "excluded_rows": int(np.count_nonzero(~available)),
                "source_positions_sha256": sha256_arrays(positions),
                "identity_sha256": sha256_arrays(
                    *(arrays[f"{split}_{field}"] for field in IDENTITY_FIELDS)
                ),
                "labels_sha256": sha256_arrays(arrays[f"{split}_labels"]),
            }
    manifest = {
        "schema_version": "phase6-5c-ta-store-v1",
        "phase": "6.5C",
        "walk": int(validation["walk"]),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "canonical_feature_path": str(canonical_feature_path.resolve()),
        "canonical_feature_sha256": sha256_file(canonical_feature_path),
        "candle_path": str(candle_path.resolve()),
        "candle_sha256": sha256_file(candle_path),
        "feature_contract": feature_manifest,
        "intersection": intersection,
        "intersection_uses_labels": False,
        "target": {"horizon_hours": 2, "tau": 0.001, "class_names": list(CLASS_NAMES)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(output_path, arrays)
    manifest["artifact"] = {
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
    }
    write_json(Path(f"{output_path}.manifest.json"), manifest)
    validate_phase65_ta_store(output_path)
    return manifest


def validate_phase65_ta_store(path: Path) -> dict[str, Any]:
    manifest_path = Path(f"{path}.manifest.json")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != "phase6-5c-ta-store-v1":
        raise ValueError("unexpected Phase 6.5C TA-store schema")
    if manifest["artifact"]["sha256"] != sha256_file(path):
        raise ValueError("Phase 6.5C TA-store artifact hash mismatch")
    with np.load(path, allow_pickle=False) as stored:
        if not np.array_equal(stored["feature_names"], np.asarray(FEATURE_NAMES)):
            raise ValueError("TA feature names/order changed")
        for split in ("train", "test"):
            positions = np.asarray(stored[f"{split}_source_positions"], dtype=np.int64)
            available = np.asarray(stored[f"{split}_feature_available"], dtype=bool)
            if not np.array_equal(positions, np.flatnonzero(available)):
                raise ValueError(f"{split} TA availability intersection mismatch")
            count = len(positions)
            expected_shapes = {
                f"{split}_ta_features": (count, 36),
                f"{split}_h0_features": (count, 445),
                f"{split}_raw_sequences": (count, 64, 5),
                f"{split}_labels": (count,),
            }
            for key, shape in expected_shapes.items():
                if stored[key].shape != shape or not np.isfinite(stored[key]).all():
                    raise ValueError(f"invalid Phase 6.5C array: {key}")
            replay_identity = sha256_arrays(
                *(np.asarray(stored[f"{split}_{field}"]) for field in IDENTITY_FIELDS)
            )
            if replay_identity != manifest["intersection"][split]["identity_sha256"]:
                raise ValueError(f"{split} TA identity hash mismatch")
    return {
        "valid": True,
        "walk": int(manifest["walk"]),
        "row_counts": {
            split: int(manifest["intersection"][split]["included_rows"])
            for split in ("train", "test")
        },
        "feature_count": 36,
    }


def _fit_standardizer(values: np.ndarray) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    mean = np.mean(array, axis=0)
    raw_std = np.std(array, axis=0, ddof=0)
    scale = np.where(raw_std < SCALER_FLOOR, 1.0, raw_std)
    return {"mean": mean, "raw_std": raw_std, "scale": scale}


def _apply_standardizer(values: np.ndarray, scaler: Mapping[str, np.ndarray]) -> np.ndarray:
    result = (np.asarray(values, dtype=np.float64) - scaler["mean"]) / scaler["scale"]
    result = np.clip(result, -SCALER_CLIP, SCALER_CLIP).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Phase 6.5C standardization produced non-finite values")
    return result


def build_phase65_ta_model(config: Phase65TAConfig) -> nn.Module:
    if config.model == "h0":
        return TrendClassifier(445, hidden_dim=128, n_classes=3)
    if config.model in ("raw_mlp", "raw_lstm"):
        baseline = "raw_ohlcv_mlp" if config.model == "raw_mlp" else "raw_ohlcv_lstm"
        baseline_config = Phase5BaselineConfig(
            baseline=baseline,
            task="classification_h2",
            walk=config.walk,
            device=config.device,
        )
        return Phase5RawBaseline(baseline_config)
    return TAMLPClassifier(in_dim=36, n_classes=3)


def _load_training_data(store_path: Path, config: Phase65TAConfig) -> dict[str, Any]:
    validation = validate_phase65_ta_store(store_path)
    if validation["walk"] != config.walk:
        raise ValueError("TA-store/config walk mismatch")
    with np.load(store_path, allow_pickle=False) as stored:
        if config.model == "h0":
            raw_train = np.asarray(stored["train_h0_features"], dtype=np.float32)
            raw_test = np.asarray(stored["test_h0_features"], dtype=np.float32)
            scaler = _fit_standardizer(raw_train)
            X_train = _apply_standardizer(raw_train, scaler)
            X_test = _apply_standardizer(raw_test, scaler)
        elif config.model == "ta_mlp":
            raw_train = np.asarray(stored["train_ta_features"], dtype=np.float32)
            raw_test = np.asarray(stored["test_ta_features"], dtype=np.float32)
            scaler = _fit_standardizer(raw_train)
            X_train = _apply_standardizer(raw_train, scaler)
            X_test = _apply_standardizer(raw_test, scaler)
        else:
            scaler = None
            X_train = np.asarray(stored["train_raw_sequences"], dtype=np.float32)
            X_test = np.asarray(stored["test_raw_sequences"], dtype=np.float32)
        result: dict[str, Any] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": np.asarray(stored["train_labels"], dtype=np.int64),
            "y_test": np.asarray(stored["test_labels"], dtype=np.int64),
            "scaler": scaler,
        }
        for split in ("train", "test"):
            for field in (*IDENTITY_FIELDS, *REPORTING_FIELDS):
                result[f"{split}_{field}"] = np.asarray(stored[f"{split}_{field}"])
    return result


def smoke_test_phase65_ta_models() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_name, protocol in strict_matrix_entries():
        config = Phase65TAConfig(model_name, protocol, walk=1, device="cpu")
        model = build_phase65_ta_model(config)
        width = 445 if model_name == "h0" else 36 if model_name == "ta_mlp" else None
        values = torch.randn(4, width) if width else torch.randn(4, 64, 5)
        output = model(values)
        target = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        criterion: nn.Module
        if protocol == "P2":
            criterion = LogitAdjustedCrossEntropy(np.asarray([0.25, 0.5, 0.25]), strength=1.0)
        else:
            criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        if output.shape != (4, 3) or not torch.isfinite(loss):
            raise RuntimeError(f"Phase 6.5C smoke failed for {model_name}/{protocol}")
        result[f"{model_name}/{protocol}"] = {
            "output_shape": list(output.shape),
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        }
    sample, audit = protocol_sample_indices(
        np.asarray([0, 0, 1, 1, 1, 1, 1, 2, 2]), "P1U", seed=0
    )
    if len(np.unique(sample)) != len(sample) or audit["duplicate_count"] != 0:
        raise RuntimeError("Phase 6.5C P1U smoke failed")
    return {"valid": True, "models": result, "p1u_audit": audit}


@torch.no_grad()
def _predict(model: nn.Module, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    return np.concatenate(
        [model(batch.to(device)).cpu().numpy() for batch in tensor.split(batch_size)]
    ).astype(np.float32)


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def run_phase65_ta_training(
    store_path: Path,
    run_root: Path,
    config: Phase65TAConfig,
) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite Phase 6.5C run: {run_root}")
    data = _load_training_data(store_path, config)
    device = resolve_device(config.device)
    set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_phase65_ta_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    priors = class_priors(data["y_train"], n_classes=3)
    if config.protocol == "P2":
        selected, sampling_audit = protocol_sample_indices(data["y_train"], "P2", seed=0)
        criterion: nn.Module = LogitAdjustedCrossEntropy(
            priors, strength=config.logit_adjustment_strength
        )
    else:
        selected, sampling_audit = protocol_sample_indices(data["y_train"], "P1U", seed=0)
        criterion = nn.CrossEntropyLoss()
    # P2 stores its log-prior adjustment as a module buffer.  Keep that
    # buffer on the same runtime device as the logits and targets.
    criterion = criterion.to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data["X_train"][selected]),
            torch.from_numpy(data["y_train"][selected]),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed),
    )
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "config.json", config.to_dict())
    write_json(run_root / "environment.json", environment_manifest(device))
    store_manifest = _load_json(Path(f"{store_path}.manifest.json"))
    write_json(
        run_root / "dataset_manifest.json",
        {
            "phase": "6.5C",
            "walk": config.walk,
            "model": config.model,
            "protocol": config.protocol,
            "store_path": str(store_path.resolve()),
            "store_sha256": sha256_file(store_path),
            "train_identity_sha256": store_manifest["intersection"]["train"]["identity_sha256"],
            "test_identity_sha256": store_manifest["intersection"]["test"]["identity_sha256"],
            "identical_p2_intersection": config.protocol == "P2",
            "evaluation_resampled": False,
        },
    )
    write_json(
        run_root / "architecture_manifest.json",
        {
            "model": config.model,
            "protocol": config.protocol,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "ta_architecture": [36, 128, 64, 32, 3] if config.model == "ta_mlp" else None,
        },
    )
    write_json(
        run_root / "imbalance_manifest.json",
        {
            "class_names": list(CLASS_NAMES),
            "training_priors": priors.tolist(),
            "sampling": sampling_audit,
            "selected_source_positions_sha256": sha256_arrays(selected),
            "evaluation_distribution": "natural and unchanged",
        },
    )
    if data["scaler"] is not None:
        _atomic_savez(run_root / "feature_standardizer.npz", data["scaler"])
        write_json(
            run_root / "feature_standardizer.manifest.json",
            {
                "fit_population": "exact Phase 6.5C training intersection only",
                "evaluation_used_for_fit": False,
                "clip": [-SCALER_CLIP, SCALER_CLIP],
            },
        )
    losses: list[float] = []
    seconds: list[float] = []
    checkpoints: list[Path] = []
    for epoch in range(1, config.epochs + 1):
        tick = time.perf_counter()
        model.train()
        total = 0.0
        count = 0
        for values, target in loader:
            values = values.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Phase 6.5C training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(values)
            count += len(values)
        losses.append(total / max(count, 1))
        seconds.append(time.perf_counter() - tick)
        if epoch in config.snapshot_epochs:
            snapshot = run_root / f"e{epoch}"
            snapshot.mkdir()
            checkpoint = snapshot / "checkpoint.pth"
            torch.save(
                {
                    "schema_version": "phase6-5c-classification-checkpoint-v1",
                    "completed_epoch": epoch,
                    "walk": config.walk,
                    "model": config.model,
                    "protocol": config.protocol,
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                },
                checkpoint,
            )
            checkpoints.append(checkpoint)
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        evaluated = build_phase65_ta_model(config).to(device)
        evaluated.load_state_dict(checkpoint["model_state_dict"], strict=True)
        tick = time.perf_counter()
        logits = _predict(evaluated, data["X_test"], config.batch_size, device)
        inference_seconds = time.perf_counter() - tick
        metrics, prediction_arrays = probabilistic_classification_metrics(
            logits, data["y_test"], CLASS_NAMES
        )
        arrays = dict(prediction_arrays)
        for field in (*IDENTITY_FIELDS, *REPORTING_FIELDS):
            arrays[field] = data[f"test_{field}"]
        snapshot = checkpoint_path.parent
        _atomic_savez(snapshot / "predictions.npz", arrays)
        payload = {
            "phase": "6.5C",
            "walk": config.walk,
            "model": config.model,
            "protocol": config.protocol,
            "epoch": int(checkpoint["completed_epoch"]),
            "train_loss": losses[int(checkpoint["completed_epoch"]) - 1],
            "classification": metrics,
            "elapsed_training_seconds": float(sum(seconds[: int(checkpoint["completed_epoch"])])),
            "inference_seconds": inference_seconds,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(snapshot / "predictions.npz"),
        }
        write_json(snapshot / "metrics.json", payload)
        _atomic_savez(
            snapshot / "history.npz",
            {
                "epochs": np.arange(1, int(checkpoint["completed_epoch"]) + 1, dtype=np.int16),
                "train_loss": np.asarray(losses[: int(checkpoint["completed_epoch"])], dtype=np.float64),
                "epoch_seconds": np.asarray(seconds[: int(checkpoint["completed_epoch"])], dtype=np.float64),
            },
        )
    snapshots = [
        _load_json(run_root / f"e{epoch}" / "metrics.json") for epoch in config.snapshot_epochs
    ]
    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    complete = {
        "complete": True,
        "phase": "6.5C",
        "walk": config.walk,
        "model": config.model,
        "protocol": config.protocol,
        "snapshots": list(config.snapshot_epochs),
        "principal_epoch": 50,
        "evaluation_selected_checkpoint": False,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    write_json(run_root / "training_complete.json", complete)
    return complete


def validate_phase65_ta_run(store_path: Path, run_root: Path) -> dict[str, Any]:
    payload = _load_json(run_root / "config.json")
    payload.pop("batch_size", None)
    payload.pop("learning_rate", None)
    payload.pop("weight_decay", None)
    payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
    config = Phase65TAConfig(**payload)
    data = _load_training_data(store_path, config)
    for epoch in config.snapshot_epochs:
        snapshot = run_root / f"e{epoch}"
        checkpoint = torch.load(snapshot / "checkpoint.pth", map_location="cpu", weights_only=True)
        model = build_phase65_ta_model(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        replay = _predict(model, data["X_test"], config.batch_size, torch.device("cpu"))
        with np.load(snapshot / "predictions.npz", allow_pickle=False) as stored:
            saved = np.asarray(stored["logits"], dtype=np.float32)
            # cuDNN and the CPU backend accumulate small recurrent reduction
            # differences through the three-layer Raw LSTM.  The immutable
            # saved CUDA logits drive metrics; this bound only validates that
            # the CPU checkpoint replay remains numerically equivalent.
            tolerance = 5e-3 if config.model == "raw_lstm" else 2e-5
            if not np.allclose(replay, saved, atol=tolerance, rtol=0.0):
                raise ValueError(f"Phase 6.5C e{epoch} CPU prediction replay mismatch")
            if not np.array_equal(stored["targets"], data["y_test"]):
                raise ValueError(f"Phase 6.5C e{epoch} target mismatch")
    return {
        "valid": True,
        "walk": config.walk,
        "model": config.model,
        "protocol": config.protocol,
        "snapshots": list(config.snapshot_epochs),
    }
