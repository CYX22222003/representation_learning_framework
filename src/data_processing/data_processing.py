from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_processing.file_list import DATA_DIR, four_hour_file_list

FEATURE_COLUMNS = ("open", "high", "low", "close", "volume")
CONTEXT_POLICY = "isolated_test_windows"


class MarketDataset(Dataset):
    def __init__(self, sequences: np.ndarray):
        self.data = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


@dataclass(frozen=True)
class PreparedContract:
    train: np.ndarray
    test: np.ndarray
    train_window_starts: np.ndarray
    test_window_starts: np.ndarray
    train_window_ends: np.ndarray
    test_window_ends: np.ndarray
    train_timestamps_ns: np.ndarray
    test_timestamps_ns: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProcessedBundle:
    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]


def _validate_ratio(train_ratio: float) -> None:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1")


def _timestamps_ns(df: pd.DataFrame) -> np.ndarray:
    if "date" not in df.columns:
        return np.arange(len(df), dtype=np.int64)
    timestamps = pd.to_datetime(df["date"], errors="raise")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("date column must be in chronological order")
    return timestamps.astype("int64").to_numpy(dtype=np.int64)


def _training_fill_values(df: pd.DataFrame, fit_end: int) -> np.ndarray:
    values = df.loc[:, FEATURE_COLUMNS].iloc[:fit_end].to_numpy(dtype=np.float64)
    fill_values = np.nanmedian(values, axis=0)
    if not np.isfinite(fill_values).all():
        missing = [FEATURE_COLUMNS[i] for i, value in enumerate(fill_values) if not np.isfinite(value)]
        raise ValueError(f"training prefix has no finite value for columns: {missing}")
    return fill_values


def preprocess_market(
    df: pd.DataFrame, *, fit_end: int | None = None, return_parameters: bool = False
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Causally impute OHLCV and fit volume scaling on ``df[:fit_end]`` only."""
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if fit_end is None:
        fit_end = len(df)
    if not 0 < fit_end <= len(df):
        raise ValueError("fit_end must identify a non-empty prefix")

    fill_values = _training_fill_values(df, fit_end)
    cleaned = df.loc[:, FEATURE_COLUMNS].astype(np.float64).ffill().fillna(
        dict(zip(FEATURE_COLUMNS, fill_values, strict=True))
    )
    values = cleaned.to_numpy(dtype=np.float64)
    volume_mean = float(values[:fit_end, 4].mean())
    volume_std = float(values[:fit_end, 4].std())
    if not np.isfinite(volume_mean) or not np.isfinite(volume_std):
        raise ValueError("training-prefix volume statistics are not finite")
    safe_std = volume_std if volume_std > 1e-8 else 1.0
    values[:, 4] = (values[:, 4] - volume_mean) / safe_std
    features = values.astype(np.float32)
    parameters = {
        "fit_start": 0,
        "fit_end_exclusive": int(fit_end),
        "imputation": "causal_forward_fill_then_training_median",
        "fill_values": {name: float(value) for name, value in zip(FEATURE_COLUMNS, fill_values, strict=True)},
        "volume_mean": volume_mean,
        "volume_std": volume_std,
        "volume_scale_denominator": safe_std,
    }
    return (features, parameters) if return_parameters else features


def create_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    data = np.asarray(data, dtype=np.float32)
    if len(data) < seq_len:
        return np.empty((0, seq_len, data.shape[1]), dtype=np.float32)
    return np.stack([data[start : start + seq_len] for start in range(len(data) - seq_len + 1)])


def split_sequences(sequences: np.ndarray, train_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Legacy helper; the production pipeline splits raw rows before this stage."""
    _validate_ratio(train_ratio)
    split_idx = int(len(sequences) * train_ratio)
    return sequences[:split_idx], sequences[split_idx:]


def prepare_contract(
    df: pd.DataFrame,
    seq_len: int,
    *,
    train_ratio: float = 0.8,
    contract_id: int = 0,
    filename: str = "<memory>",
    source_sha256: str | None = None,
) -> PreparedContract:
    _validate_ratio(train_ratio)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    split_idx = int(len(df) * train_ratio)
    if split_idx < seq_len or len(df) - split_idx < seq_len:
        raise ValueError(
            f"contract needs at least seq_len={seq_len} raw rows on both sides of boundary; "
            f"got train={split_idx}, test={len(df) - split_idx}"
        )

    timestamps = _timestamps_ns(df)
    features, preprocessing = preprocess_market(df, fit_end=split_idx, return_parameters=True)
    train = create_sequences(features[:split_idx], seq_len)
    test = create_sequences(features[split_idx:], seq_len)
    train_starts = np.arange(len(train), dtype=np.int64)
    test_starts = split_idx + np.arange(len(test), dtype=np.int64)
    train_ends = train_starts + seq_len - 1
    test_ends = test_starts + seq_len - 1
    metadata = {
        "contract_id": int(contract_id), "filename": filename, "source_sha256": source_sha256,
        "raw_row_count": int(len(df)), "raw_boundary_index": split_idx,
        "raw_boundary_timestamp_ns": int(timestamps[split_idx]),
        "last_train_timestamp_ns": int(timestamps[split_idx - 1]),
        "train_window_count": int(len(train)), "test_window_count": int(len(test)),
        "preprocessing": preprocessing,
    }
    return PreparedContract(
        train, test, train_starts, test_starts, train_ends, test_ends,
        timestamps[train_ends], timestamps[test_ends], metadata
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_processed_bundle(
    file_list: Sequence[tuple[str, int]], seq_len: int, *, train_ratio: float = 0.8,
    data_dir: str | os.PathLike[str] = DATA_DIR,
) -> ProcessedBundle:
    prepared: list[PreparedContract] = []
    skipped: list[dict[str, Any]] = []
    for contract_id, (filename, _) in enumerate(file_list):
        path = Path(data_dir) / filename
        try:
            prepared.append(prepare_contract(
                pd.read_feather(path), seq_len, train_ratio=train_ratio, contract_id=contract_id,
                filename=filename, source_sha256=_sha256(path)
            ))
        except Exception as exc:
            skipped.append({"contract_id": contract_id, "filename": filename, "reason": str(exc)})
    if not prepared:
        raise ValueError("No valid contracts produced both train and test windows")

    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "test"):
        arrays[split] = np.concatenate([getattr(item, split) for item in prepared]).astype(np.float32)
        arrays[f"{split}_contract_ids"] = np.concatenate([
            np.full(len(getattr(item, split)), item.metadata["contract_id"], dtype=np.int32)
            for item in prepared
        ])
        for field in ("window_starts", "window_ends", "timestamps_ns"):
            arrays[f"{split}_{field}"] = np.concatenate([getattr(item, f"{split}_{field}") for item in prepared])

    manifest = {
        "schema_version": 2, "pipeline": "raw_time_first", "train_ratio": train_ratio,
        "seq_len": seq_len, "feature_columns": list(FEATURE_COLUMNS),
        "context_policy": CONTEXT_POLICY,
        "contract_order": [item.metadata["filename"] for item in prepared],
        "train_shape": list(arrays["train"].shape), "test_shape": list(arrays["test"].shape),
        "identity_hashes": {
            split: _array_sha256(
                arrays[f"{split}_contract_ids"], arrays[f"{split}_window_starts"],
                arrays[f"{split}_window_ends"], arrays[f"{split}_timestamps_ns"],
            ) for split in ("train", "test")
        },
        "sequence_hashes": {split: _array_sha256(arrays[split]) for split in ("train", "test")},
        "contracts": [item.metadata for item in prepared], "skipped_contracts": skipped,
    }
    return ProcessedBundle(arrays, manifest)


def build_from_file_list(file_list, seq_len, train_ratio=0.8):
    bundle = build_processed_bundle(file_list, seq_len, train_ratio=train_ratio)
    return bundle.arrays["train"], bundle.arrays["test"]


if __name__ == "__main__":
    train_4h, test_4h = build_from_file_list(four_hour_file_list, 64)
    print("Train size:", len(MarketDataset(train_4h)))
    print("Test size:", len(MarketDataset(test_4h)))
