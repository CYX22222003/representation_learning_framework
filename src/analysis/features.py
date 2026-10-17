"""Exact branch views and training-only coordinate standardization."""

from pathlib import Path

import numpy as np

from .artifacts import hash_array, identity_hash, write_arrays_atomic, write_json_atomic
from .contracts import BRANCHES
from .types import AblationSpec, FeatureStandardizer, RowIdentity, TaskSplit


def select_features(split: TaskSplit, ablation: AblationSpec) -> np.ndarray:
    expected = tuple(b.key for b in BRANCHES if b.key in ablation.branches)
    if expected != ablation.branches or len(set(expected)) != len(ablation.branches):
        raise ValueError("Invalid canonical branch selection/order")
    values = np.concatenate([split.branches[k] for k in expected], axis=1)
    if values.shape != (len(split.targets), ablation.width) or not np.isfinite(values).all():
        raise ValueError("Selected feature shape/finiteness mismatch")
    return values


def fit_standardizer(training_features: np.ndarray, identity: RowIdentity) -> FeatureStandardizer:
    values = np.asarray(training_features, dtype=np.float64)
    if values.ndim != 2 or not len(values) or len(values) != len(identity.condition_ids) or not np.isfinite(values).all():
        raise ValueError("Scaler requires a finite nonempty aligned training matrix")
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=0)
    return FeatureStandardizer(mean, std, np.where(std < 1e-8, 1.0, std), identity_hash(identity))


def apply_standardizer(features: np.ndarray, scaler: FeatureStandardizer) -> np.ndarray:
    a = np.asarray(features, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != len(scaler.mean) or np.any(scaler.scale <= 0):
        raise ValueError("Scaler dimension/scale mismatch")
    result = np.clip((a - scaler.mean) / scaler.scale, *scaler.clip_limits).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Standardized features contain non-finite values")
    return result


def validate_scaler_parity(scaler: FeatureStandardizer, canonical: FeatureStandardizer,
                          ablation: AblationSpec) -> None:
    positions = np.concatenate([np.arange(*b.h0_interval) for b in BRANCHES if b.key in ablation.branches])
    for name in ("mean", "raw_std", "scale"):
        if not np.allclose(getattr(scaler, name), getattr(canonical, name)[positions], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Canonical scaler parity failed: {name}")
    if scaler.clip_limits != canonical.clip_limits:
        raise ValueError("Canonical scaler clip mismatch")


def save_standardizer(path: Path, scaler: FeatureStandardizer) -> None:
    arrays = {name: np.asarray(getattr(scaler, name)) for name in ("mean", "raw_std", "scale")}
    arrays.update(clip_low=np.asarray(scaler.clip_limits[0]), clip_high=np.asarray(scaler.clip_limits[1]))
    write_arrays_atomic(path, arrays)
    write_json_atomic(Path(f"{path}.manifest.json"), {
        "training_identity_hash": scaler.training_identity_hash,
        "array_hashes": {k: hash_array(v) for k, v in arrays.items()}, "fit_split": "train",
    })


def load_standardizer(path: Path, training_identity_hash: str = "legacy") -> FeatureStandardizer:
    with np.load(path, allow_pickle=False) as a:
        result = FeatureStandardizer(a["mean"], a["raw_std"], a["scale"], training_identity_hash,
                                     (float(a["clip_low"]) if "clip_low" in a else -10.,
                                      float(a["clip_high"]) if "clip_high" in a else 10.))
    return result
