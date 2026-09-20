"""Build leakage-safe Phase 3 encoder sequences from raw 4-hour feather files."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_processing.data_processing import build_processed_bundle  # noqa: E402
from phase3_encoder_common import (  # noqa: E402
    DEFAULT_PROCESSED_NPZ,
    PHASE3_DATA_ROOT,
    refuse_occupied,
    require_phase3_path,
    sha256_file,
    write_json,
)
from validate_phase3_encoder_data import validate_bundle  # noqa: E402


SELECTION_PREFIX_ROWS = 256
MIN_RAW_ROWS = 320


def _score_candidate(path: Path, train_ratio: float, seq_len: int) -> dict[str, Any]:
    record: dict[str, Any] = {"filename": path.name, "source_sha256": sha256_file(path)}
    try:
        frame = pd.read_feather(path)
        required = {"open", "high", "low", "close", "volume"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing columns {missing}")
        if "date" in frame.columns:
            timestamps = pd.to_datetime(frame["date"], errors="raise")
            if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
                raise ValueError("timestamps must be strictly ordered and unique")
        row_count = len(frame)
        boundary = int(row_count * train_ratio)
        minimum = max(MIN_RAW_ROWS, math.ceil(SELECTION_PREFIX_ROWS / train_ratio), seq_len * 2)
        if row_count < minimum or boundary < SELECTION_PREFIX_ROWS:
            raise ValueError(f"requires at least {minimum} rows with selection prefix in train")
        volume = pd.to_numeric(frame["volume"].iloc[:SELECTION_PREFIX_ROWS], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite_positive = np.isfinite(volume) & (volume > 0.0)
        record.update({
            "raw_row_count": row_count,
            "raw_boundary_index": boundary,
            "selection_prefix_rows": SELECTION_PREFIX_ROWS,
            "positive_volume_count": int(finite_positive.sum()),
            "log_volume_score": float(np.log1p(np.maximum(volume[finite_positive], 0.0)).sum()),
            "eligible": True,
            "reason": None,
        })
    except Exception as exc:
        record.update({"eligible": False, "reason": str(exc)})
    return record


def select_universe(
    data_dir: Path, timeframe: str, top_k: int, train_ratio: float, seq_len: int
) -> tuple[list[tuple[str, int]], list[dict[str, Any]]]:
    suffix = f"-{timeframe}.feather"
    paths = sorted(path for path in data_dir.iterdir() if path.is_file() and path.name.endswith(suffix))
    if not paths:
        raise FileNotFoundError(f"no raw {timeframe} feather files found under {data_dir}")
    records = [_score_candidate(path, train_ratio, seq_len) for path in paths]
    eligible = [record for record in records if record["eligible"]]
    eligible.sort(
        key=lambda row: (-row["positive_volume_count"], -row["log_volume_score"], row["filename"])
    )
    if len(eligible) < top_k:
        raise ValueError(f"only {len(eligible)} eligible contracts; top_k={top_k}")
    rank_by_name = {row["filename"]: rank for rank, row in enumerate(eligible, start=1)}
    selected_names = {row["filename"] for row in eligible[:top_k]}
    for record in records:
        record["rank"] = rank_by_name.get(record["filename"])
        record["selected"] = record["filename"] in selected_names
    selected = [(row["filename"], int(row["positive_volume_count"])) for row in eligible[:top_k]]
    return selected, sorted(records, key=lambda row: row["filename"])


def prepare(
    data_dir: Path,
    out_path: Path,
    *,
    timeframe: str = "4h",
    seq_len: int = 64,
    top_k: int = 50,
    train_ratio: float = 0.8,
) -> dict[str, Any]:
    if timeframe != "4h" or seq_len != 64 or top_k != 50 or train_ratio != 0.8:
        raise ValueError("initial Phase 3 encoder contract is fixed to 4h/seq64/top50/80-20")
    out_path = require_phase3_path(out_path, PHASE3_DATA_ROOT / "processed")
    manifest_path = Path(f"{out_path}.manifest.json")
    refuse_occupied((out_path, manifest_path))
    selected, universe_records = select_universe(data_dir, timeframe, top_k, train_ratio, seq_len)
    bundle = build_processed_bundle(selected, seq_len, train_ratio=train_ratio, data_dir=data_dir)
    if bundle.manifest["skipped_contracts"]:
        raise ValueError(f"selected contracts failed preprocessing: {bundle.manifest['skipped_contracts']}")
    if len(bundle.manifest["contracts"]) != top_k:
        raise ValueError("processed contract count differs from frozen top_k")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **bundle.arrays)
    manifest = {
        **bundle.manifest,
        "phase": 3,
        "purpose": "encoder_pretraining",
        "timeframe": timeframe,
        "top_k": top_k,
        "processed_npz": str(out_path),
        "universe_selection": {
            "policy": "fixed_early_prefix_positive_volume_then_log_volume",
            "selection_prefix_rows": SELECTION_PREFIX_ROWS,
            "minimum_raw_rows": MIN_RAW_ROWS,
            "uses_test_values": False,
            "candidate_records": universe_records,
        },
    }
    write_json(manifest_path, manifest)
    # The file hash is added only after the NPZ is closed and stable.
    manifest["processed_npz_sha256"] = sha256_file(out_path)
    write_json(manifest_path, manifest)
    return validate_bundle(out_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-path", type=Path, default=DEFAULT_PROCESSED_NPZ)
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args(argv)
    try:
        result = prepare(
            args.data_dir, args.out_path, timeframe=args.timeframe, seq_len=args.seq_len,
            top_k=args.top_k, train_ratio=args.train_ratio,
        )
        print(f"Phase 3 encoder data prepared and validated: {result}")
        return 0
    except Exception as exc:
        print(f"Phase 3 encoder data preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
