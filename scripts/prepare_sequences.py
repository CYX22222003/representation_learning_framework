# Stage 1 of the data pipeline.
# Reads raw Polymarket OHLCV feather files, selects the top-K most active
# contracts per timeframe (by file size as a proxy for trading activity),
# fixes a raw-time 80/20 boundary, causally imputes, fits volume scaling on the
# training prefix, then constructs non-overlapping train/test window sets.
#   train: [N_train, seq_len, 5]  float32
#   test:  [N_test,  seq_len, 5]  float32
# Both arrays are saved to a single compressed .npz file under data/processed/.
#
# Usage:
#   python scripts/prepare_sequences.py --timeframes 4h --seq-len 64 --top-k 50
#   python scripts/prepare_sequences.py --timeframes 1h,4h,1d --seq-len 64 --top-k 50

import argparse
import json
import os
import sys
from typing import Iterable

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from data_processing.data_processing import build_processed_bundle
from data_processing.file_list import list_top_k


def save_sequences_for_timeframe(
    timeframe: str, seq_len: int, top_k: int, out_dir: str,
    *, train_ratio: float = 0.8, output_suffix: str = "split_safe", overwrite: bool = False,
) -> str:
    # list_top_k returns [(filename, file_size), ...] sorted by size descending;
    # file size is used as a proxy for the number of candles / trading activity.
    file_list = list_top_k(timeframe, top_k)
    bundle = build_processed_bundle(file_list, seq_len, train_ratio=train_ratio)
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""
    out_path = os.path.join(out_dir, f"market_{timeframe}_seq{seq_len}_top{top_k}{suffix}.npz")
    manifest_path = f"{out_path}.manifest.json"
    existing = [path for path in (out_path, manifest_path) if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifacts: {existing}")
    np.savez_compressed(out_path, **bundle.arrays)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({**bundle.manifest, "processed_npz": out_path}, handle, indent=2, sort_keys=True)
    return out_path


def _parse_timeframes(raw: str) -> list[str]:
    timeframes = [t.strip() for t in raw.split(",") if t.strip()]
    allowed = {"1h", "4h", "1d"}
    invalid = [t for t in timeframes if t not in allowed]
    if invalid:
        raise ValueError(f"Invalid timeframe(s): {invalid}. Allowed: {sorted(allowed)}")
    return timeframes


def run_batch(
    timeframes: Iterable[str], seq_len: int, top_k: int, out_dir: str, **kwargs
) -> list[str]:
    outputs = []
    for timeframe in timeframes:
        out_path = save_sequences_for_timeframe(
            timeframe=timeframe, seq_len=seq_len, top_k=top_k, out_dir=out_dir, **kwargs
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
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--output-suffix", default="split_safe")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    timeframes = _parse_timeframes(args.timeframes)
    run_batch(
        timeframes, seq_len=args.seq_len, top_k=args.top_k, out_dir=args.out_dir,
        train_ratio=args.train_ratio, output_suffix=args.output_suffix, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
