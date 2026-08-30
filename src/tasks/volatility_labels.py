from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


TARGET_DEFINITION = "next_stride_one_window_realized_volatility"
TARGET_FORMULA = "sqrt(mean(diff(log(clip(sequence[j+1,:,close], eps, None))) ** 2))"
OVERLAP_NOTE = "next stored stride-one sequence; 63 of 64 timesteps overlap when seq_len=64"


@dataclass(frozen=True)
class VolatilityLabelMetadata:
    task: str = "volatility_prediction"
    label_mode: str = TARGET_DEFINITION
    target_formula: str = TARGET_FORMULA
    timeframe: str = "4h"
    seq_len: int = 64
    top_k: int = 50
    price_index: int = 3
    horizon: int = 1
    train_ratio: float = 0.8
    dataset_id: str = ""
    processed_npz_sha256: str = ""
    overlap_note: str = OVERLAP_NOTE

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ContractAlignment:
    contract_id: int
    filename: str
    raw_rows: int
    sequence_count: int
    train_sequence_count: int
    test_sequence_count: int
    train_label_count: int
    test_label_count: int
    train_global_offset: int
    test_global_offset: int
    file_size: int | None = None
    file_sha256: str | None = None
    status: str = "included"
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_dataset_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def realized_volatility(prices: np.ndarray, eps: float = 1e-8) -> np.float32:
    p = np.asarray(prices, dtype=np.float32)
    if p.ndim != 1:
        raise ValueError(f"Expected 1-D prices, got {p.shape}")
    if p.shape[0] < 2:
        return np.float32(0.0)
    returns = np.diff(np.log(np.clip(p, eps, None)))
    return np.float32(np.sqrt(np.mean(returns**2)))


def _concat_or_empty(values: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=dtype)
    return np.concatenate(values).astype(dtype, copy=False)


def build_aligned_volatility_labels(
    contract_sequences: Iterable[np.ndarray],
    *,
    price_index: int = 3,
    horizon: int = 1,
    train_ratio: float = 0.8,
    contract_ids: Iterable[int] | None = None,
    window_start_offset: int = 0,
) -> dict[str, np.ndarray]:
    if horizon != 1:
        raise ValueError("Only horizon=1 is supported by the shared volatility MVP target")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    sequences_list = [np.asarray(seq, dtype=np.float32) for seq in contract_sequences]
    ids = list(contract_ids) if contract_ids is not None else list(range(len(sequences_list)))
    if len(ids) != len(sequences_list):
        raise ValueError("contract_ids length must match contract_sequences")

    train_labels: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    train_contract_ids: list[np.ndarray] = []
    test_contract_ids: list[np.ndarray] = []
    train_starts: list[np.ndarray] = []
    test_starts: list[np.ndarray] = []

    train_offset = 0
    test_offset = 0
    for contract_id, seq in zip(ids, sequences_list):
        if seq.ndim != 3:
            raise ValueError(f"Expected contract sequences [N, seq_len, features], got {seq.shape}")
        if not 0 <= price_index < seq.shape[2]:
            raise ValueError(f"price_index {price_index} out of range for shape {seq.shape}")

        n_seq = int(seq.shape[0])
        split_idx = int(n_seq * train_ratio)
        test_count = n_seq - split_idx

        train_count = max(0, split_idx - horizon)
        test_label_count = max(0, test_count - horizon)

        if train_count:
            local = np.arange(train_count, dtype=np.int64)
            future = seq[local + horizon, :, price_index]
            train_labels.append(np.asarray([realized_volatility(row) for row in future], dtype=np.float32))
            train_indices.append(train_offset + local)
            train_contract_ids.append(np.full(train_count, int(contract_id), dtype=np.int32))
            train_starts.append(window_start_offset + local)

        if test_label_count:
            local = np.arange(test_label_count, dtype=np.int64)
            future = seq[split_idx + local + horizon, :, price_index]
            test_labels.append(np.asarray([realized_volatility(row) for row in future], dtype=np.float32))
            test_indices.append(test_offset + local)
            test_contract_ids.append(np.full(test_label_count, int(contract_id), dtype=np.int32))
            test_starts.append(window_start_offset + split_idx + local)

        train_offset += split_idx
        test_offset += test_count

    return {
        "train_labels": _concat_or_empty(train_labels, np.dtype(np.float32)),
        "test_labels": _concat_or_empty(test_labels, np.dtype(np.float32)),
        "train_row_indices": _concat_or_empty(train_indices, np.dtype(np.int64)),
        "test_row_indices": _concat_or_empty(test_indices, np.dtype(np.int64)),
        "train_contract_ids": _concat_or_empty(train_contract_ids, np.dtype(np.int32)),
        "test_contract_ids": _concat_or_empty(test_contract_ids, np.dtype(np.int32)),
        "train_window_starts": _concat_or_empty(train_starts, np.dtype(np.int64)),
        "test_window_starts": _concat_or_empty(test_starts, np.dtype(np.int64)),
    }


def validate_volatility_label_bundle(
    bundle: dict[str, np.ndarray],
    *,
    train_size: int | None = None,
    test_size: int | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    required = {
        "train_labels": np.float32,
        "test_labels": np.float32,
        "train_row_indices": np.int64,
        "test_row_indices": np.int64,
        "train_contract_ids": np.int32,
        "test_contract_ids": np.int32,
        "train_window_starts": np.int64,
        "test_window_starts": np.int64,
    }
    for key, dtype in required.items():
        if key not in bundle:
            raise ValueError(f"Missing volatility label array: {key}")
        arr = np.asarray(bundle[key])
        if arr.ndim != 1:
            raise ValueError(f"{key} must be 1-D, got {arr.shape}")
        if arr.dtype != dtype:
            raise ValueError(f"{key} must have dtype {dtype}, got {arr.dtype}")

    for split in ("train", "test"):
        n = int(bundle[f"{split}_labels"].shape[0])
        for suffix in ("row_indices", "contract_ids", "window_starts"):
            if int(bundle[f"{split}_{suffix}"].shape[0]) != n:
                raise ValueError(f"{split}_{suffix} row count does not match {split}_labels")
        labels = bundle[f"{split}_labels"]
        if not np.all(np.isfinite(labels)):
            raise ValueError(f"{split}_labels contains non-finite values")
        if np.any(labels < 0):
            raise ValueError(f"{split}_labels contains negative realized volatility")
        row_indices = bundle[f"{split}_row_indices"]
        limit = train_size if split == "train" else test_size
        if limit is not None and n:
            if int(row_indices.min()) < 0 or int(row_indices.max()) >= int(limit):
                raise ValueError(f"{split}_row_indices out of range for processed {split} size {limit}")
        starts = bundle[f"{split}_window_starts"]
        contracts = bundle[f"{split}_contract_ids"]
        for contract_id in np.unique(contracts):
            mask = contracts == contract_id
            if np.any(np.diff(starts[mask]) <= 0):
                raise ValueError(f"{split}_window_starts are not strictly increasing for contract {contract_id}")
            if np.any(np.diff(row_indices[mask]) <= 0):
                raise ValueError(f"{split}_row_indices are not strictly increasing for contract {contract_id}")

    if metadata is not None:
        if metadata.get("target_definition") not in (None, TARGET_DEFINITION):
            raise ValueError("manifest target_definition does not match the volatility label contract")
        if metadata.get("label_mode") not in (None, TARGET_DEFINITION):
            raise ValueError("manifest label_mode does not match the volatility label contract")
        if metadata.get("horizon") not in (None, 1):
            raise ValueError("volatility label bundle only supports horizon=1")

    return {
        "train_label_count": int(bundle["train_labels"].shape[0]),
        "test_label_count": int(bundle["test_labels"].shape[0]),
        "train_contract_count": int(len(np.unique(bundle["train_contract_ids"]))),
        "test_contract_count": int(len(np.unique(bundle["test_contract_ids"]))),
        "overlap_note": OVERLAP_NOTE,
    }


def save_volatility_label_bundle(
    bundle: dict[str, np.ndarray],
    manifest: dict[str, object],
    out_path: str | Path,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    validate_volatility_label_bundle(bundle, metadata=manifest)
    np.savez_compressed(out, **bundle)
    Path(f"{out}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_volatility_label_bundle(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    bundle_path = Path(path)
    with np.load(bundle_path) as data:
        bundle = {key: data[key].copy() for key in data.files}
    manifest_path = Path(f"{bundle_path}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    validate_volatility_label_bundle(bundle, metadata=manifest)
    return bundle, manifest
