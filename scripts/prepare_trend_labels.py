# Build TA-MLP-style tri-class trend labels aligned to processed sequence rows.
#
# The output stores labels plus row indices into the existing train/test feature
# arrays, so framework training can use only labelable rows without rebuilding
# feature bundles.

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

from baselines.ta_mlp_baseline.ta_labels import BUY, HOLD, SELL, assign_labels, compute_thresholds
from data_processing.file_list import list_top_k


CLASS_NAMES = np.array(["BUY", "HOLD", "SELL"])


def _class_counts(labels: np.ndarray, n_classes: int = 3) -> dict[str, int]:
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes)
    return {str(idx): int(count) for idx, count in enumerate(counts.tolist())}


def _load_processed_shapes(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    with np.load(path) as data:
        if "train" not in data or "test" not in data:
            raise ValueError("Processed .npz must contain train and test arrays")
        return tuple(data["train"].shape), tuple(data["test"].shape)


def build_trend_label_bundle(
    processed_npz: Path,
    timeframe: str,
    seq_len: int,
    top_k: int,
    data_dir: Path,
    train_ratio: float,
    b_window: int,
    f_window: int,
    hold_q: float,
    buy_sell_q: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if b_window <= 0:
        raise ValueError("b_window must be positive")
    if f_window <= 0:
        raise ValueError("f_window must be positive")

    processed_train_shape, processed_test_shape = _load_processed_shapes(processed_npz)
    file_list = list_top_k(timeframe=timeframe, top_k=top_k, data_dir=str(data_dir))

    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []
    contracts: list[dict[str, object]] = []
    train_offset = 0
    test_offset = 0

    for filename, file_size in file_list:
        path = data_dir / filename
        try:
            df = pd.read_feather(path).ffill().interpolate()
        except Exception as exc:
            contracts.append({"filename": filename, "status": "skipped", "reason": str(exc)})
            continue

        if "close" not in df:
            contracts.append({"filename": filename, "status": "skipped", "reason": "missing close"})
            continue

        close = df["close"].to_numpy(dtype=np.float32)
        n_sequences = len(close) - seq_len
        if n_sequences <= 0:
            contracts.append(
                {
                    "filename": filename,
                    "status": "skipped",
                    "reason": "not enough rows",
                    "raw_rows": int(len(close)),
                }
            )
            continue

        split_idx = int(n_sequences * train_ratio)
        end_positions = np.arange(seq_len - 1, seq_len - 1 + n_sequences, dtype=np.int64)

        train_end_positions = end_positions[:split_idx]
        alpha, beta = compute_thresholds(
            pd.Series(close[train_end_positions]),
            hold_q=hold_q,
            buy_sell_q=buy_sell_q,
        )
        row_labels = assign_labels(
            pd.Series(close),
            b_window=b_window,
            f_window=f_window,
            alpha=alpha,
            beta=beta,
        )

        # Keep labels inside each split. The final f_window sequence ends in a
        # split cannot be labeled without looking beyond that split's endpoint.
        train_labelable = max(0, split_idx - f_window)
        test_total = n_sequences - split_idx
        test_labelable = max(0, test_total - f_window)

        if train_labelable:
            local = np.arange(train_labelable, dtype=np.int64)
            train_indices.append(train_offset + local)
            train_labels.append(row_labels[end_positions[local]].astype(np.int64))
        if test_labelable:
            local = split_idx + np.arange(test_labelable, dtype=np.int64)
            test_indices.append(test_offset + np.arange(test_labelable, dtype=np.int64))
            test_labels.append(row_labels[end_positions[local]].astype(np.int64))

        contracts.append(
            {
                "filename": filename,
                "file_size": int(file_size),
                "status": "included",
                "raw_rows": int(len(close)),
                "sequence_count": int(n_sequences),
                "train_sequence_count": int(split_idx),
                "test_sequence_count": int(test_total),
                "train_label_count": int(train_labelable),
                "test_label_count": int(test_labelable),
                "alpha": float(alpha),
                "beta": float(beta),
            }
        )
        train_offset += split_idx
        test_offset += test_total

    if train_offset != processed_train_shape[0] or test_offset != processed_test_shape[0]:
        raise ValueError(
            "Reconstructed sequence counts do not match processed .npz: "
            f"reconstructed train/test={train_offset}/{test_offset}, "
            f"processed train/test={processed_train_shape[0]}/{processed_test_shape[0]}"
        )
    if not train_indices or not test_indices:
        raise RuntimeError("No labelable trend rows were produced")

    train_idx = np.concatenate(train_indices).astype(np.int64)
    test_idx = np.concatenate(test_indices).astype(np.int64)
    y_train = np.concatenate(train_labels).astype(np.int64)
    y_test = np.concatenate(test_labels).astype(np.int64)

    payload = {
        "train_indices": train_idx,
        "train_labels": y_train,
        "test_indices": test_idx,
        "test_labels": y_test,
        "class_names": CLASS_NAMES,
        "b_window": np.int32(b_window),
        "f_window": np.int32(f_window),
        "hold_q": np.float32(hold_q),
        "buy_sell_q": np.float32(buy_sell_q),
        "seq_len": np.int32(seq_len),
        "top_k": np.int32(top_k),
        "train_ratio": np.float32(train_ratio),
    }
    manifest = {
        "task": "trend_classification",
        "label_mode": "ta_triclass",
        "class_mapping": {"BUY": BUY, "HOLD": HOLD, "SELL": SELL},
        "class_names": CLASS_NAMES.tolist(),
        "processed_npz": str(processed_npz),
        "timeframe": timeframe,
        "seq_len": seq_len,
        "top_k": top_k,
        "data_dir": str(data_dir),
        "train_ratio": train_ratio,
        "label_params": {
            "b_window": b_window,
            "f_window": f_window,
            "hold_q": hold_q,
            "buy_sell_q": buy_sell_q,
        },
        "threshold_fit": "per contract using training sequence end prices only",
        "split_boundary": "labels are kept inside each processed train/test split",
        "processed_shapes": {
            "train": list(processed_train_shape),
            "test": list(processed_test_shape),
        },
        "label_counts": {
            "train": int(y_train.shape[0]),
            "test": int(y_test.shape[0]),
        },
        "class_counts": {
            "train": _class_counts(y_train),
            "test": _class_counts(y_test),
        },
        "contracts": contracts,
    }
    return payload, manifest


def save_trend_label_bundle(payload: dict[str, np.ndarray], manifest: dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    manifest_path = Path(f"{out_path}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TA-MLP-style tri-class trend labels.")
    parser.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    parser.add_argument(
        "--out-path",
        default="data/task_labels/trend_classification/triclass_4h_seq64_top50.npz",
    )
    parser.add_argument("--timeframe", choices=("1h", "4h", "1d"), default="4h")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--b-window", type=int, default=5)
    parser.add_argument("--f-window", type=int, default=2)
    parser.add_argument("--hold-q", type=float, default=0.85)
    parser.add_argument("--buy-sell-q", type=float, default=0.997)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    out_path = Path(args.out_path)
    if out_path.exists() and not args.overwrite:
        print(f"output exists; pass --overwrite: {out_path}", file=sys.stderr)
        return 2
    try:
        payload, manifest = build_trend_label_bundle(
            processed_npz=Path(args.processed_npz),
            timeframe=args.timeframe,
            seq_len=args.seq_len,
            top_k=args.top_k,
            data_dir=Path(args.data_dir),
            train_ratio=args.train_ratio,
            b_window=args.b_window,
            f_window=args.f_window,
            hold_q=args.hold_q,
            buy_sell_q=args.buy_sell_q,
        )
        save_trend_label_bundle(payload, manifest, out_path)
        print(f"saved trend labels: {out_path}")
        print(f"saved manifest: {out_path}.manifest.json")
        print(
            "label counts: "
            f"train={manifest['label_counts']['train']} test={manifest['label_counts']['test']}"
        )
        print(
            "class counts: "
            f"train={manifest['class_counts']['train']} test={manifest['class_counts']['test']}"
        )
        return 0
    except Exception as exc:
        print(f"trend label preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
