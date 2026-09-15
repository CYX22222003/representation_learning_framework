from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.data_processing import create_sequences, preprocess_market, split_sequences
from data_processing.file_list import list_top_k
from tasks.phase2_decoders.data import sha256_file


def timestamp_ns(frame: pd.DataFrame, seq_len: int, count: int) -> np.ndarray:
    if "date" not in frame:
        return np.full(count, -1, dtype=np.int64)
    values = pd.to_datetime(frame["date"], utc=True, errors="coerce").astype("int64").to_numpy()
    return np.asarray(values[seq_len-1:seq_len-1+count], dtype=np.int64)


def reconstruct_contract_sequences(
    processed_npz: str | Path, *, timeframe: str, top_k: int, seq_len: int,
    data_dir: str | Path = "data", train_ratio: float = 0.8,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, object]], np.ndarray, np.ndarray]:
    with np.load(processed_npz, allow_pickle=False) as data:
        processed_train = np.asarray(data["train"], np.float32)
        processed_test = np.asarray(data["test"], np.float32)
    sequences_out, timestamps_out, contracts = [], [], []
    train_offset = test_offset = 0
    for filename, file_size in list_top_k(timeframe, top_k, str(data_dir)):
        source = Path(data_dir) / filename
        try:
            frame = pd.read_feather(source)
            sequences = create_sequences(preprocess_market(frame), seq_len)
        except Exception as exc:
            contracts.append({"filename": filename, "status": "skipped", "reason": str(exc)})
            continue
        if len(sequences) == 0:
            contracts.append({"filename": filename, "status": "skipped", "reason": "no sequences"})
            continue
        train, test = split_sequences(sequences, train_ratio)
        if not np.allclose(train, processed_train[train_offset:train_offset+len(train)], equal_nan=True):
            raise ValueError(f"reconstructed training sequences differ for {filename}")
        if not np.allclose(test, processed_test[test_offset:test_offset+len(test)], equal_nan=True):
            raise ValueError(f"reconstructed test sequences differ for {filename}")
        contract_id = len(sequences_out)
        times = timestamp_ns(frame, seq_len, len(sequences))
        sequences_out.append(sequences)
        timestamps_out.append(times)
        contracts.append({
            "contract_id": contract_id, "filename": filename, "status": "included",
            "raw_rows": int(len(frame)), "sequence_count": int(len(sequences)),
            "train_sequence_count": int(len(train)), "test_sequence_count": int(len(test)),
            "train_global_offset": train_offset, "test_global_offset": test_offset,
            "file_size": int(file_size), "file_sha256": sha256_file(source),
        })
        train_offset += len(train)
        test_offset += len(test)
    if (train_offset, test_offset) != (len(processed_train), len(processed_test)):
        raise ValueError("reconstructed split counts differ from processed NPZ")
    return sequences_out, timestamps_out, contracts, processed_train, processed_test
