# Stage 1 of the data pipeline.
# Reads raw Polymarket OHLCV feather files, selects the top-K most active
# contracts per timeframe (by file size as a proxy for trading activity),
# preprocesses each contract (ffill, volume z-score, sliding windows, 80/20
# chronological split), and merges across all contracts into two arrays:
#   train: [N_train, seq_len, 5]  float32
#   test:  [N_test,  seq_len, 5]  float32
# Both arrays are saved to a single compressed .npz file under data/processed/.
#
# Usage:
#   python scripts/prepare_sequences.py --timeframes 4h --seq-len 64 --top-k 50
#   python scripts/prepare_sequences.py --timeframes 1h,4h,1d --seq-len 64 --top-k 50

import argparse
import os
import sys
from typing import Iterable

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from data_processing.data_processing import build_from_file_list
from data_processing.file_list import list_top_k


def save_sequences_for_timeframe(
    timeframe: str, seq_len: int, top_k: int, out_dir: str
) -> str:
    # list_top_k returns [(filename, file_size), ...] sorted by size descending;
    # file size is used as a proxy for the number of candles / trading activity.
    file_list = list_top_k(timeframe, top_k)
    # build_from_file_list preprocesses each contract and concatenates sequences
    # across contracts; the 80/20 split is applied per contract before merging
    # to prevent data leakage (later contracts can't bleed into the test set).
    train, test = build_from_file_list(file_list, seq_len)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"market_{timeframe}_seq{seq_len}_top{top_k}.npz")
    # Keys "train" and "test" are the canonical contract used by all downstream scripts.
    np.savez_compressed(out_path, train=train, test=test)
    return out_path


def _parse_timeframes(raw: str) -> list[str]:
    timeframes = [t.strip() for t in raw.split(",") if t.strip()]
    allowed = {"1h", "4h", "1d"}
    invalid = [t for t in timeframes if t not in allowed]
    if invalid:
        raise ValueError(f"Invalid timeframe(s): {invalid}. Allowed: {sorted(allowed)}")
    return timeframes


def run_batch(
    timeframes: Iterable[str], seq_len: int, top_k: int, out_dir: str
) -> list[str]:
    outputs = []
    for timeframe in timeframes:
        out_path = save_sequences_for_timeframe(
            timeframe=timeframe, seq_len=seq_len, top_k=top_k, out_dir=out_dir
        )
        print(f"Saved sequences: {out_path}")
        outputs.append(out_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/test sequence tensors from raw market feather files."
    )
    parser.add_argument("--timeframes", type=str, default="4h")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="data/processed")
    args = parser.parse_args()

    timeframes = _parse_timeframes(args.timeframes)
    run_batch(timeframes, seq_len=args.seq_len, top_k=args.top_k, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
