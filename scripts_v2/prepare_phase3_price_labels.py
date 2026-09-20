"""Build contract-local horizon-1 price labels from the Phase 3 processed bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_encoder_common import (  # noqa: E402
    DEFAULT_PRICE_LABELS, DEFAULT_PROCESSED_NPZ, PHASE3_DATA_ROOT,
    refuse_occupied, require_phase3_path, sha256_arrays, sha256_file, write_json,
)
from validate_phase3_encoder_data import validate_bundle  # noqa: E402


def _split_labels(bundle, split: str) -> dict[str, np.ndarray]:
    sequences = np.asarray(bundle[split], dtype=np.float32)
    contract_ids = np.asarray(bundle[f"{split}_contract_ids"], dtype=np.int32)
    starts = np.asarray(bundle[f"{split}_window_starts"], dtype=np.int64)
    timestamps = np.asarray(bundle[f"{split}_timestamps_ns"], dtype=np.int64)
    eligible = []
    for contract_id in np.unique(contract_ids):
        rows = np.flatnonzero(contract_ids == contract_id)
        if len(rows) < 2:
            continue
        if not np.all(np.diff(starts[rows]) == 1):
            raise ValueError(f"{split} contract {contract_id} windows are not consecutive")
        eligible.extend(rows[:-1].tolist())
    indices = np.asarray(eligible, dtype=np.int64)
    future = indices + 1
    if not np.array_equal(contract_ids[indices], contract_ids[future]):
        raise ValueError(f"{split} price horizon crosses a contract boundary")
    if not np.array_equal(starts[future], starts[indices] + 1):
        raise ValueError(f"{split} price horizon is not the next raw-time window")
    current = sequences[indices, -1, 3].astype(np.float32)
    targets = sequences[future, -1, 3].astype(np.float32)
    if not np.isfinite(current).all() or not np.isfinite(targets).all():
        raise ValueError(f"{split} price labels contain non-finite values")
    return {
        f"{split}_row_indices": indices,
        f"{split}_labels": targets,
        f"{split}_contract_ids": contract_ids[indices],
        f"{split}_window_starts": starts[indices],
        f"{split}_timestamps_ns": timestamps[indices],
        f"{split}_current_close": current,
        f"{split}_future_close": targets.copy(),
    }


def validate_labels(path: Path, processed_npz: Path) -> dict:
    with np.load(processed_npz, allow_pickle=False) as processed, np.load(path, allow_pickle=False) as labels:
        result = {}
        for split in ("train", "test"):
            expected = _split_labels(processed, split)
            for name, values in expected.items():
                if name not in labels.files or not np.array_equal(labels[name], values):
                    raise ValueError(f"price label replay mismatch: {name}")
            result[f"{split}_rows"] = len(expected[f"{split}_labels"])
            result[f"{split}_identity_hash"] = sha256_arrays(
                expected[f"{split}_row_indices"], expected[f"{split}_contract_ids"],
                expected[f"{split}_window_starts"], expected[f"{split}_timestamps_ns"],
            )
    return result


def prepare(processed_npz: Path, out_path: Path) -> dict:
    validate_bundle(processed_npz)
    out_path = require_phase3_path(out_path, PHASE3_DATA_ROOT / "task_labels")
    manifest_path = Path(f"{out_path}.manifest.json")
    refuse_occupied((out_path, manifest_path))
    with np.load(processed_npz, allow_pickle=False) as bundle:
        payload = {**_split_labels(bundle, "train"), **_split_labels(bundle, "test")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    validation = validate_labels(out_path, processed_npz)
    write_json(manifest_path, {
        "phase": 3, "task": "price_prediction", "horizon": 1, "price_index": 3,
        "construction": "next consecutive window close within each contract and stored split",
        "terminal_policy": "drop final row of every contract independently in each split",
        "processed_npz": str(processed_npz.resolve()),
        "processed_npz_sha256": sha256_file(processed_npz),
        "labels_npz": str(out_path.resolve()), "labels_npz_sha256": sha256_file(out_path),
        **validation,
    })
    return validation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-npz", type=Path, default=DEFAULT_PROCESSED_NPZ)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_PRICE_LABELS)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate_labels(args.out_path, args.processed_npz) if args.verify else prepare(args.processed_npz, args.out_path)
        print(f"Phase 3 price labels valid: {result}")
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr); return 2
    except Exception as exc:
        print(f"Phase 3 price label preparation failed: {exc}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
