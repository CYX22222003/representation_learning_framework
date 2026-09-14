from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from features.feature_store import NpzFeatureStore


BRANCH_ORDER = ("statistical", "transformed", "vae", "contrastive", "byol")


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
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def build_temporal_index(
    contract_split_sizes: Sequence[tuple[int, int]],
    *,
    context_length: int,
    contract_timestamps_ns: Sequence[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    if context_length < 1:
        raise ValueError("context_length must be positive")
    if contract_timestamps_ns is not None and len(contract_timestamps_ns) != len(contract_split_sizes):
        raise ValueError("timestamp and contract counts differ")
    output: dict[str, list[np.ndarray]] = {
        f"{split}_{field}": []
        for split in ("train", "test")
        for field in ("context_row_indices", "context_window_starts", "context_timestamps_ns",
                      "final_row_indices", "contract_ids", "final_window_starts", "final_timestamps_ns")
    }
    train_offset = test_offset = 0
    for contract_id, (train_count, test_count) in enumerate(contract_split_sizes):
        total = train_count + test_count
        timestamps = (
            np.asarray(contract_timestamps_ns[contract_id], dtype=np.int64)
            if contract_timestamps_ns is not None
            else np.full(total, -1, dtype=np.int64)
        )
        if timestamps.shape != (total,):
            raise ValueError("each timestamp array must match the contract sequence count")
        for split, count, global_offset, local_offset in (
            ("train", train_count, train_offset, 0),
            ("test", test_count, test_offset, train_count),
        ):
            n_contexts = max(0, count - context_length + 1)
            if n_contexts:
                starts = np.arange(n_contexts, dtype=np.int64)[:, None]
                steps = np.arange(context_length, dtype=np.int64)[None, :]
                local_context = starts + steps
                rows = global_offset + local_context
                window_starts = local_offset + local_context
                context_times = timestamps[window_starts]
                output[f"{split}_context_row_indices"].append(rows)
                output[f"{split}_context_window_starts"].append(window_starts)
                output[f"{split}_context_timestamps_ns"].append(context_times)
                output[f"{split}_final_row_indices"].append(rows[:, -1])
                output[f"{split}_contract_ids"].append(np.full(n_contexts, contract_id, dtype=np.int32))
                output[f"{split}_final_window_starts"].append(window_starts[:, -1])
                output[f"{split}_final_timestamps_ns"].append(context_times[:, -1])
        train_offset += train_count
        test_offset += test_count
    bundle: dict[str, np.ndarray] = {}
    for key, parts in output.items():
        if parts:
            bundle[key] = np.concatenate(parts)
        elif "context_" in key:
            bundle[key] = np.empty((0, context_length), dtype=np.int64)
        else:
            bundle[key] = np.empty(0, dtype=np.int32 if key.endswith("contract_ids") else np.int64)
    bundle["context_length"] = np.asarray(context_length, dtype=np.int32)
    return bundle


def validate_temporal_index(
    bundle: Mapping[str, np.ndarray], *, train_size: int | None = None, test_size: int | None = None
) -> dict[str, object]:
    if "context_length" not in bundle:
        raise ValueError("temporal index is missing context_length")
    k = int(np.asarray(bundle["context_length"]))
    if k < 1:
        raise ValueError("invalid context_length")
    for split, limit in (("train", train_size), ("test", test_size)):
        context = np.asarray(bundle[f"{split}_context_row_indices"], dtype=np.int64)
        starts = np.asarray(bundle[f"{split}_context_window_starts"], dtype=np.int64)
        times = np.asarray(bundle[f"{split}_context_timestamps_ns"], dtype=np.int64)
        final = np.asarray(bundle[f"{split}_final_row_indices"], dtype=np.int64)
        contracts = np.asarray(bundle[f"{split}_contract_ids"], dtype=np.int32)
        final_starts = np.asarray(bundle[f"{split}_final_window_starts"], dtype=np.int64)
        final_times = np.asarray(bundle[f"{split}_final_timestamps_ns"], dtype=np.int64)
        if context.ndim != 2 or context.shape[1] != k:
            raise ValueError(f"{split} contexts must have shape [N,{k}]")
        if starts.shape != context.shape or times.shape != context.shape:
            raise ValueError(f"{split} context identity shapes differ")
        n = len(context)
        if any(len(value) != n for value in (final, contracts, final_starts, final_times)):
            raise ValueError(f"{split} final identity lengths differ")
        if n and (not np.all(np.diff(context, axis=1) == 1) or not np.all(np.diff(starts, axis=1) == 1)):
            raise ValueError(f"{split} contains non-contiguous contexts")
        if n and (not np.array_equal(final, context[:, -1]) or not np.array_equal(final_starts, starts[:, -1])):
            raise ValueError(f"{split} final rows do not match contexts")
        if n and not np.array_equal(final_times, times[:, -1]):
            raise ValueError(f"{split} final timestamps do not match contexts")
        known_pairs = (times[:, 1:] >= 0) & (times[:, :-1] >= 0)
        if np.any(known_pairs & (np.diff(times, axis=1) <= 0)):
            raise ValueError(f"{split} timestamps are not increasing")
        if limit is not None and n and (context.min() < 0 or context.max() >= limit):
            raise ValueError(f"{split} row indices exceed split size")
        for contract_id in np.unique(contracts):
            mask = contracts == contract_id
            if np.any(np.diff(final[mask]) <= 0):
                raise ValueError(f"{split} final rows are not increasing within contract")
    return {
        "context_length": k,
        "train_context_count": int(len(bundle["train_final_row_indices"])),
        "test_context_count": int(len(bundle["test_final_row_indices"])),
        "train_identity_hash": identity_hash(bundle["train_final_row_indices"], bundle["train_contract_ids"], bundle["train_final_window_starts"]),
        "test_identity_hash": identity_hash(bundle["test_final_row_indices"], bundle["test_contract_ids"], bundle["test_final_window_starts"]),
    }


def build_price_label_bundle(
    contract_sequences: Sequence[np.ndarray], *, horizon: int = 1, price_index: int = 3,
    train_ratio: float = 0.8, timestamps_ns: Sequence[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    fields = {f"{s}_{n}": [] for s in ("train", "test") for n in
              ("labels", "row_indices", "contract_ids", "window_starts", "timestamps_ns")}
    train_offset = test_offset = 0
    for cid, raw in enumerate(contract_sequences):
        seq = np.asarray(raw, dtype=np.float32)
        if seq.ndim != 3 or not 0 <= price_index < seq.shape[2]:
            raise ValueError("invalid contract sequence shape or price index")
        split_at = int(len(seq) * train_ratio)
        times = np.asarray(timestamps_ns[cid], dtype=np.int64) if timestamps_ns is not None else np.full(len(seq), -1, np.int64)
        for split, start, count, offset in (("train", 0, split_at, train_offset), ("test", split_at, len(seq)-split_at, test_offset)):
            eligible = max(0, count - horizon)
            local = np.arange(eligible, dtype=np.int64)
            current = start + local
            labels = seq[current + horizon, -1, price_index]
            finite = np.isfinite(labels)
            local, current, labels = local[finite], current[finite], labels[finite]
            fields[f"{split}_labels"].append(labels.astype(np.float32))
            fields[f"{split}_row_indices"].append((offset + local).astype(np.int64))
            fields[f"{split}_contract_ids"].append(np.full(len(local), cid, np.int32))
            fields[f"{split}_window_starts"].append(current.astype(np.int64))
            fields[f"{split}_timestamps_ns"].append(times[current].astype(np.int64))
        train_offset += split_at
        test_offset += len(seq) - split_at
    result = {key: np.concatenate(parts) if parts else np.empty(0) for key, parts in fields.items()}
    result.update(horizon=np.asarray(horizon, np.int32), price_index=np.asarray(price_index, np.int32))
    return result


def validate_regression_labels(bundle: Mapping[str, np.ndarray], *, train_size: int, test_size: int) -> dict[str, object]:
    for split, limit in (("train", train_size), ("test", test_size)):
        names = ("labels", "row_indices", "contract_ids", "window_starts", "timestamps_ns")
        arrays = [np.asarray(bundle[f"{split}_{name}"]) for name in names]
        if not arrays or len({len(a) for a in arrays}) != 1 or len(arrays[0]) == 0:
            raise ValueError(f"{split} regression-label arrays are empty or misaligned")
        if not np.all(np.isfinite(arrays[0])):
            raise ValueError(f"{split} labels contain non-finite values")
        rows = arrays[1].astype(np.int64)
        if len(np.unique(rows)) != len(rows) or rows.min() < 0 or rows.max() >= limit:
            raise ValueError(f"{split} row indices are invalid")
        for cid in np.unique(arrays[2]):
            mask = arrays[2] == cid
            if np.any(np.diff(arrays[3][mask]) <= 0):
                raise ValueError(f"{split} window starts are not increasing")
    return {
        "train_count": int(len(bundle["train_labels"])), "test_count": int(len(bundle["test_labels"])),
        "train_identity_hash": identity_hash(bundle["train_row_indices"], bundle["train_contract_ids"], bundle["train_window_starts"]),
        "test_identity_hash": identity_hash(bundle["test_row_indices"], bundle["test_contract_ids"], bundle["test_window_starts"]),
    }


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def load_scaled_features(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, np.ndarray]]:
    feature_path = Path(path)
    with np.load(f"{feature_path}.index.npz", allow_pickle=False) as index:
        train_size, test_size = int(index["train_size"]), int(index["test_size"])
    branches = NpzFeatureStore(str(feature_path)).load().as_branch_dict()
    if tuple(branches) != BRANCH_ORDER:
        raise ValueError(f"expected canonical branch order {BRANCH_ORDER}, got {tuple(branches)}")
    train_parts, test_parts, dims, scaler = [], [], {}, {}
    for name in BRANCH_ORDER:
        values = np.asarray(branches[name], np.float32)
        if len(values) != train_size + test_size:
            raise ValueError(f"branch {name} row count differs from feature index")
        train, test = values[:train_size], values[train_size:]
        mean = train.mean(0, dtype=np.float64).astype(np.float32)
        std = train.std(0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        train_parts.append(np.clip((train-mean)/std, -10, 10).astype(np.float32))
        test_parts.append(np.clip((test-mean)/std, -10, 10).astype(np.float32))
        dims[name] = int(values.shape[1])
        scaler[f"{name}__mean"], scaler[f"{name}__std"] = mean, std
    return np.concatenate(train_parts, 1), np.concatenate(test_parts, 1), dims, scaler


def align_contexts_to_labels(
    temporal: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    label_rows = np.asarray(labels[f"{split}_row_indices"], np.int64)
    positions = {int(row): pos for pos, row in enumerate(label_rows)}
    final = np.asarray(temporal[f"{split}_final_row_indices"], np.int64)
    keep = np.asarray([i for i, row in enumerate(final) if int(row) in positions], np.int64)
    label_pos = np.asarray([positions[int(final[i])] for i in keep], np.int64)
    for field, temporal_name, label_name in (
        ("contract", "contract_ids", "contract_ids"), ("window", "final_window_starts", "window_starts")
    ):
        left = np.asarray(temporal[f"{split}_{temporal_name}"])[keep]
        right = np.asarray(labels[f"{split}_{label_name}"])[label_pos]
        if not np.array_equal(left, right):
            raise ValueError(f"{split} {field} identities differ between temporal index and labels")
    identities = {
        "row_indices": final[keep],
        "contract_ids": np.asarray(temporal[f"{split}_contract_ids"], np.int32)[keep],
        "window_starts": np.asarray(temporal[f"{split}_final_window_starts"], np.int64)[keep],
        "timestamps_ns": np.asarray(temporal[f"{split}_final_timestamps_ns"], np.int64)[keep],
    }
    return np.asarray(temporal[f"{split}_context_row_indices"], np.int64)[keep], np.asarray(labels[f"{split}_labels"])[label_pos], identities


def read_manifest(path: str | Path) -> dict[str, object]:
    manifest = Path(f"{path}.manifest.json")
    return json.loads(manifest.read_text()) if manifest.exists() else {}
