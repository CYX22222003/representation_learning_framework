from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from features.feature_store import NpzFeatureStore
from tasks.phase2_classification.labels import load_label_bundle, validate_label_bundle


IDENTITY_FIELDS = ("indices", "contract_ids", "window_starts", "timestamps_ns")


def load_processed(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "train" not in data or "test" not in data:
            raise ValueError("processed NPZ must contain train and test")
        train = np.asarray(data["train"], dtype=np.float32)
        test = np.asarray(data["test"], dtype=np.float32)
    if train.ndim != 3 or test.ndim != 3 or train.shape[1:] != test.shape[1:]:
        raise ValueError(f"invalid processed shapes: {train.shape}, {test.shape}")
    return train, test


def alignment_positions(
    label_bundle: Mapping[str, np.ndarray], alignment_npz: str | Path | None
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if alignment_npz is None:
        return (
            np.arange(len(label_bundle["train_labels"]), dtype=np.int64),
            np.arange(len(label_bundle["test_labels"]), dtype=np.int64),
            {"alignment_npz": None},
        )
    path = Path(alignment_npz)
    with np.load(path, allow_pickle=False) as data:
        required = {"train_label_positions", "test_label_positions", "train_indices", "test_indices"}
        if not required.issubset(data.files):
            raise ValueError(f"alignment bundle missing {sorted(required.difference(data.files))}")
        train_positions = np.asarray(data["train_label_positions"], dtype=np.int64)
        test_positions = np.asarray(data["test_label_positions"], dtype=np.int64)
        expected_train = np.asarray(data["train_indices"], dtype=np.int64)
        expected_test = np.asarray(data["test_indices"], dtype=np.int64)
    for split, positions, expected in (
        ("train", train_positions, expected_train), ("test", test_positions, expected_test)
    ):
        if len(positions) == 0 or positions.min() < 0 or positions.max() >= len(label_bundle[f"{split}_labels"]):
            raise ValueError(f"{split} alignment positions are invalid")
        actual = np.asarray(label_bundle[f"{split}_indices"])[positions]
        if not np.array_equal(actual, expected):
            raise ValueError(f"{split} alignment identities do not match the label bundle")
    manifest_path = Path(f"{path}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return train_positions, test_positions, {"alignment_npz": str(path), **manifest}


def load_labels_for_rows(
    labels_npz: str | Path, alignment_npz: str | Path | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    bundle, label_manifest = load_label_bundle(labels_npz)
    train_positions, test_positions, alignment_manifest = alignment_positions(bundle, alignment_npz)
    selected: dict[str, np.ndarray] = {"class_names": bundle["class_names"].copy()}
    for split, positions in (("train", train_positions), ("test", test_positions)):
        for name in ("labels", *IDENTITY_FIELDS, "current_close", "future_close", "delta"):
            selected[f"{split}_{name}"] = np.asarray(bundle[f"{split}_{name}"])[positions]
    validate_label_bundle(selected)
    return selected, {"label_manifest": label_manifest, "alignment_manifest": alignment_manifest}


def load_raw_inputs(
    processed_npz: str | Path, labels: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    train, test = load_processed(processed_npz)
    return train[np.asarray(labels["train_indices"])], test[np.asarray(labels["test_indices"])]


def _parse_branches(branches: str | None, available: list[str]) -> list[str]:
    if branches is None or branches.strip() == "":
        return available
    selected = [item.strip() for item in branches.split(",") if item.strip()]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("--branches must contain unique, non-empty names")
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"unknown branches: {unknown}; available={available}")
    return selected


def load_framework_inputs(
    features_npz: str | Path,
    labels: Mapping[str, np.ndarray],
    branches: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, np.ndarray]]:
    feature_path = Path(features_npz)
    index_path = Path(f"{feature_path}.index.npz")
    if not index_path.exists():
        raise FileNotFoundError(f"missing feature index: {index_path}")
    with np.load(index_path, allow_pickle=False) as data:
        train_size, test_size = int(data["train_size"]), int(data["test_size"])
    all_branches = NpzFeatureStore(str(feature_path)).load().as_branch_dict()
    names = _parse_branches(branches, list(all_branches))
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    dims: dict[str, int] = {}
    scaler: dict[str, np.ndarray] = {}
    for name in names:
        values = np.asarray(all_branches[name], dtype=np.float32)
        if len(values) != train_size + test_size:
            raise ValueError(f"feature branch {name} has incompatible row count")
        train = values[:train_size][np.asarray(labels["train_indices"])]
        test = values[train_size:][np.asarray(labels["test_indices"])]
        mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = train.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        train = np.clip((train - mean) / std, -10.0, 10.0).astype(np.float32)
        test = np.clip((test - mean) / std, -10.0, 10.0).astype(np.float32)
        train_parts.append(train)
        test_parts.append(test)
        dims[name] = int(train.shape[1])
        scaler[f"{name}__mean"] = mean
        scaler[f"{name}__std"] = std
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1), dims, scaler


def load_ta_inputs(
    ta_features_npz: str | Path, labels: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    path = Path(ta_features_npz)
    with np.load(path, allow_pickle=False) as data:
        required = {"train_features", "test_features", "train_indices", "test_indices"}
        if not required.issubset(data.files):
            raise ValueError(f"TA feature bundle missing {sorted(required.difference(data.files))}")
        train = np.asarray(data["train_features"], dtype=np.float32)
        test = np.asarray(data["test_features"], dtype=np.float32)
        if not np.array_equal(data["train_indices"], labels["train_indices"]):
            raise ValueError("TA train rows do not match selected label rows")
        if not np.array_equal(data["test_indices"], labels["test_indices"]):
            raise ValueError("TA test rows do not match selected label rows")
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    scaler = {"ta__mean": mean, "ta__std": std}
    return (
        np.clip((train - mean) / std, -10.0, 10.0).astype(np.float32),
        np.clip((test - mean) / std, -10.0, 10.0).astype(np.float32),
        scaler,
    )
