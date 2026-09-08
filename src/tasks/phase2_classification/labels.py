from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DOWN = 0
STABLE = 1
UP = 2
CLASS_NAMES = np.asarray(["DOWN", "STABLE", "UP"])
LABEL_MODE = "absolute_probability_movement"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        arr = np.ascontiguousarray(values)
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def movement_classes(delta: np.ndarray, threshold: float) -> np.ndarray:
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    values = np.asarray(delta)
    if not np.all(np.isfinite(values)):
        raise ValueError("delta contains non-finite values")
    labels = np.full(values.shape, STABLE, dtype=np.int64)
    labels[values < -threshold] = DOWN
    labels[values > threshold] = UP
    return labels


def _concat(values: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=dtype)
    return np.concatenate(values).astype(dtype, copy=False)


def build_movement_label_bundle(
    contract_sequences: Iterable[np.ndarray],
    *,
    horizon: int = 2,
    threshold: float = 0.005,
    price_index: int = 3,
    train_ratio: float = 0.8,
    contract_ids: Iterable[int] | None = None,
    sequence_timestamps_ns: Sequence[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Build split-local movement labels without crossing split/contract boundaries."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    sequences = [np.asarray(item, dtype=np.float32) for item in contract_sequences]
    ids = list(contract_ids) if contract_ids is not None else list(range(len(sequences)))
    if len(ids) != len(sequences):
        raise ValueError("contract_ids length must match contract_sequences")
    if sequence_timestamps_ns is not None and len(sequence_timestamps_ns) != len(sequences):
        raise ValueError("sequence_timestamps_ns length must match contract_sequences")

    fields: dict[str, list[np.ndarray]] = {
        f"{split}_{name}": []
        for split in ("train", "test")
        for name in (
            "indices",
            "labels",
            "contract_ids",
            "window_starts",
            "timestamps_ns",
            "current_close",
            "future_close",
            "delta",
        )
    }
    train_offset = 0
    test_offset = 0

    for position, (contract_id, seq) in enumerate(zip(ids, sequences)):
        if seq.ndim != 3:
            raise ValueError(f"Expected [N, seq_len, features], got {seq.shape}")
        if not 0 <= price_index < seq.shape[2]:
            raise ValueError(f"price_index {price_index} out of range for {seq.shape}")
        n_sequences = int(seq.shape[0])
        split_idx = int(n_sequences * train_ratio)
        test_count = n_sequences - split_idx
        timestamps = (
            np.asarray(sequence_timestamps_ns[position], dtype=np.int64)
            if sequence_timestamps_ns is not None
            else np.full(n_sequences, -1, dtype=np.int64)
        )
        if timestamps.shape != (n_sequences,):
            raise ValueError("each timestamp array must have one entry per sequence")

        for split, start, count, offset in (
            ("train", 0, split_idx, train_offset),
            ("test", split_idx, test_count, test_offset),
        ):
            eligible = max(0, count - horizon)
            if eligible == 0:
                continue
            local = np.arange(eligible, dtype=np.int64)
            current_position = start + local
            future_position = current_position + horizon
            current = seq[current_position, -1, price_index]
            future = seq[future_position, -1, price_index]
            finite = np.isfinite(current) & np.isfinite(future)
            local = local[finite]
            current_position = current_position[finite]
            current = current[finite]
            future = future[finite]
            delta = future - current

            fields[f"{split}_indices"].append(offset + local)
            fields[f"{split}_labels"].append(movement_classes(delta, threshold))
            fields[f"{split}_contract_ids"].append(
                np.full(len(local), int(contract_id), dtype=np.int32)
            )
            fields[f"{split}_window_starts"].append(current_position)
            fields[f"{split}_timestamps_ns"].append(timestamps[current_position])
            fields[f"{split}_current_close"].append(current.astype(np.float32))
            fields[f"{split}_future_close"].append(future.astype(np.float32))
            fields[f"{split}_delta"].append(delta.astype(np.float32))

        train_offset += split_idx
        test_offset += test_count

    bundle: dict[str, np.ndarray] = {}
    dtypes = {
        "indices": np.dtype(np.int64),
        "labels": np.dtype(np.int64),
        "contract_ids": np.dtype(np.int32),
        "window_starts": np.dtype(np.int64),
        "timestamps_ns": np.dtype(np.int64),
        "current_close": np.dtype(np.float32),
        "future_close": np.dtype(np.float32),
        "delta": np.dtype(np.float32),
    }
    for split in ("train", "test"):
        for name, dtype in dtypes.items():
            bundle[f"{split}_{name}"] = _concat(fields[f"{split}_{name}"], dtype)
    bundle.update(
        {
            "class_names": CLASS_NAMES.copy(),
            "horizon": np.asarray(horizon, dtype=np.int32),
            "threshold": np.asarray(threshold, dtype=np.float32),
            "price_index": np.asarray(price_index, dtype=np.int32),
            "train_ratio": np.asarray(train_ratio, dtype=np.float32),
        }
    )
    return bundle


def validate_label_bundle(
    bundle: dict[str, np.ndarray],
    *,
    train_size: int | None = None,
    test_size: int | None = None,
) -> dict[str, object]:
    row_fields = (
        "indices", "labels", "contract_ids", "window_starts", "timestamps_ns",
        "current_close", "future_close", "delta",
    )
    for split, limit in (("train", train_size), ("test", test_size)):
        missing = [f"{split}_{name}" for name in row_fields if f"{split}_{name}" not in bundle]
        if missing:
            raise ValueError(f"Missing movement-label arrays: {missing}")
        n_rows = len(bundle[f"{split}_labels"])
        if n_rows == 0:
            raise ValueError(f"{split} contains no eligible labels")
        for name in row_fields:
            values = np.asarray(bundle[f"{split}_{name}"])
            if values.ndim != 1 or len(values) != n_rows:
                raise ValueError(f"{split}_{name} must be 1-D with {n_rows} rows")
        labels = np.asarray(bundle[f"{split}_labels"], dtype=np.int64)
        if np.any((labels < DOWN) | (labels > UP)):
            raise ValueError(f"{split}_labels contains an invalid class")
        indices = np.asarray(bundle[f"{split}_indices"], dtype=np.int64)
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{split}_indices contains duplicates")
        if limit is not None and (indices.min() < 0 or indices.max() >= limit):
            raise ValueError(f"{split}_indices outside processed split size {limit}")
        for name in ("current_close", "future_close", "delta"):
            if not np.all(np.isfinite(bundle[f"{split}_{name}"])):
                raise ValueError(f"{split}_{name} contains non-finite values")
        expected = np.asarray(bundle[f"{split}_future_close"]) - np.asarray(bundle[f"{split}_current_close"])
        if not np.allclose(expected, bundle[f"{split}_delta"], atol=1e-7):
            raise ValueError(f"{split}_delta does not match future-current")
        contracts = np.asarray(bundle[f"{split}_contract_ids"])
        starts = np.asarray(bundle[f"{split}_window_starts"])
        for contract_id in np.unique(contracts):
            if np.any(np.diff(starts[contracts == contract_id]) <= 0):
                raise ValueError(f"{split} window starts are not increasing within contract")

    class_names = [str(item) for item in np.asarray(bundle.get("class_names", [])).tolist()]
    if class_names != CLASS_NAMES.tolist():
        raise ValueError(f"class_names must be {CLASS_NAMES.tolist()}")
    train_counts = np.bincount(bundle["train_labels"].astype(np.int64), minlength=3)
    if np.any(train_counts == 0):
        raise ValueError("every class must occur in training data")
    return {
        "train_count": int(len(bundle["train_labels"])),
        "test_count": int(len(bundle["test_labels"])),
        "train_class_counts": train_counts.tolist(),
        "test_class_counts": np.bincount(bundle["test_labels"].astype(np.int64), minlength=3).tolist(),
        "train_identity_hash": identity_hash(bundle["train_indices"], bundle["train_contract_ids"], bundle["train_window_starts"]),
        "test_identity_hash": identity_hash(bundle["test_indices"], bundle["test_contract_ids"], bundle["test_window_starts"]),
    }


def save_label_bundle(bundle: dict[str, np.ndarray], manifest: dict[str, object], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    validation = validate_label_bundle(bundle)
    np.savez_compressed(out, **bundle)
    payload = {**manifest, **validation, "label_mode": LABEL_MODE, "class_names": CLASS_NAMES.tolist()}
    Path(f"{out}.manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_label_bundle(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    label_path = Path(path)
    with np.load(label_path, allow_pickle=False) as data:
        bundle = {key: data[key].copy() for key in data.files}
    manifest_path = Path(f"{label_path}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    validate_label_bundle(bundle)
    if manifest.get("label_mode") not in (None, LABEL_MODE):
        raise ValueError("manifest label_mode is incompatible with Phase 2 movement labels")
    return bundle, manifest
