from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_processing.data_processing import create_sequences, preprocess_market, split_sequences
from data_processing.file_list import list_top_k
from tasks.phase2_classification.labels import (
    LABEL_MODE,
    build_movement_label_bundle,
    save_label_bundle,
    sha256_file,
    validate_label_bundle,
)


def _timestamp_ns(df: pd.DataFrame, seq_len: int, n_sequences: int) -> np.ndarray:
    if "date" not in df:
        return np.full(n_sequences, -1, dtype=np.int64)
    dates = pd.to_datetime(df["date"], utc=True, errors="coerce")
    values = dates.astype("int64").to_numpy()
    result = values[seq_len - 1 : seq_len - 1 + n_sequences]
    return np.asarray(result, dtype=np.int64)


def build_bundle(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    processed_path = Path(args.processed_npz)
    with np.load(processed_path, allow_pickle=False) as data:
        processed_train = np.asarray(data["train"], dtype=np.float32)
        processed_test = np.asarray(data["test"], dtype=np.float32)
    contract_sequences = []
    timestamps = []
    contracts = []
    source_files = []
    train_offset = test_offset = 0
    for filename, file_size in list_top_k(args.timeframe, args.top_k, args.data_dir):
        path = Path(args.data_dir) / filename
        try:
            frame = pd.read_feather(path)
            sequences = create_sequences(preprocess_market(frame), args.seq_len)
        except Exception as exc:
            contracts.append({"filename": filename, "status": "skipped", "reason": str(exc)})
            continue
        if len(sequences) == 0:
            contracts.append({"filename": filename, "status": "skipped", "reason": "no sequences"})
            continue
        train, test = split_sequences(sequences, args.train_ratio)
        expected_train = processed_train[train_offset : train_offset + len(train)]
        expected_test = processed_test[test_offset : test_offset + len(test)]
        if not np.allclose(train, expected_train, equal_nan=True) or not np.allclose(
            test, expected_test, equal_nan=True
        ):
            raise ValueError(f"reconstructed sequences differ from processed NPZ for {filename}")
        contract_id = len(contract_sequences)
        contract_sequences.append(sequences)
        timestamps.append(_timestamp_ns(frame, args.seq_len, len(sequences)))
        file_sha = sha256_file(path)
        contracts.append(
            {
                "contract_id": contract_id, "filename": filename, "status": "included",
                "raw_rows": int(len(frame)), "sequence_count": int(len(sequences)),
                "train_sequence_count": int(len(train)), "test_sequence_count": int(len(test)),
                "train_global_offset": train_offset, "test_global_offset": test_offset,
                "file_size": int(file_size), "file_sha256": file_sha,
            }
        )
        source_files.append({"filename": filename, "size": int(file_size), "sha256": file_sha})
        train_offset += len(train)
        test_offset += len(test)
    if (train_offset, test_offset) != (len(processed_train), len(processed_test)):
        raise ValueError(
            "reconstructed sequence counts do not match processed NPZ: "
            f"{train_offset}/{test_offset} != {len(processed_train)}/{len(processed_test)}"
        )
    bundle = build_movement_label_bundle(
        contract_sequences, horizon=args.horizon, threshold=args.threshold,
        price_index=args.price_index, train_ratio=args.train_ratio,
        contract_ids=range(len(contract_sequences)), sequence_timestamps_ns=timestamps,
    )
    validation = validate_label_bundle(bundle, train_size=len(processed_train), test_size=len(processed_test))
    manifest = {
        "task": "trend_classification", "label_mode": LABEL_MODE,
        "formula": "delta = close(sequence[i+h], -1) - close(sequence[i], -1)",
        "boundary_rule": "DOWN if delta < -tau; UP if delta > tau; equality is STABLE",
        "horizon": args.horizon, "threshold": args.threshold, "price_index": args.price_index,
        "timeframe": args.timeframe, "seq_len": args.seq_len, "top_k": args.top_k,
        "train_ratio": args.train_ratio, "data_dir": args.data_dir,
        "processed_npz": str(processed_path),
        "processed_npz_sha256": sha256_file(processed_path),
        "processed_shapes": {"train": list(processed_train.shape), "test": list(processed_test.shape)},
        "source_files": source_files, "contracts": contracts,
        "split_safety": "train and test labels built independently within each contract split",
        **validation,
    }
    return bundle, manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build Phase 2 absolute probability-movement labels.")
    result.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    result.add_argument("--out-path", default="data/task_labels/trend_classification/probability_movement_4h_h2_tau005_seq64_top50.npz")
    result.add_argument("--timeframe", choices=("1h", "4h", "1d"), default="4h")
    result.add_argument("--seq-len", type=int, default=64)
    result.add_argument("--top-k", type=int, default=50)
    result.add_argument("--data-dir", default="data")
    result.add_argument("--train-ratio", type=float, default=0.8)
    result.add_argument("--horizon", type=int, default=2)
    result.add_argument("--threshold", type=float, default=0.005)
    result.add_argument("--price-index", type=int, default=3)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--verify", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    out = Path(args.out_path)
    try:
        if args.verify:
            with np.load(out, allow_pickle=False) as data:
                bundle = {key: data[key].copy() for key in data.files}
            print(json.dumps(validate_label_bundle(bundle), indent=2))
            return 0
        if (out.exists() or Path(f"{out}.manifest.json").exists()) and not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {out}")
        bundle, manifest = build_bundle(args)
        save_label_bundle(bundle, manifest, out)
        print(json.dumps({"wrote": str(out), **validate_label_bundle(bundle)}, indent=2))
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"movement label preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
