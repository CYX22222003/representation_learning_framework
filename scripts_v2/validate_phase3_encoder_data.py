"""Validate the Phase 3 encoder input bundle and its raw-time provenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_processing.data_processing import CONTEXT_POLICY, FEATURE_COLUMNS  # noqa: E402
from phase3_encoder_common import (  # noqa: E402
    DEFAULT_PROCESSED_NPZ,
    PHASE3_DATA_ROOT,
    read_json,
    require_phase3_path,
    sha256_arrays,
    sha256_file,
)


REQUIRED_ARRAYS = tuple(
    name
    for split in ("train", "test")
    for name in (
        split,
        f"{split}_contract_ids",
        f"{split}_window_starts",
        f"{split}_window_ends",
        f"{split}_timestamps_ns",
    )
)


def validate_bundle(npz_path: Path) -> dict[str, Any]:
    npz_path = require_phase3_path(npz_path, PHASE3_DATA_ROOT / "processed")
    manifest_path = Path(f"{npz_path}.manifest.json")
    if not npz_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("processed NPZ and companion manifest are both required")
    manifest = read_json(manifest_path)

    if manifest.get("phase") != 3 or manifest.get("purpose") != "encoder_pretraining":
        raise ValueError("manifest is not a Phase 3 encoder-pretraining artifact")
    if manifest.get("pipeline") != "raw_time_first":
        raise ValueError("pipeline must be raw_time_first")
    if manifest.get("context_policy") != CONTEXT_POLICY:
        raise ValueError(f"context_policy must be {CONTEXT_POLICY}")
    if manifest.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("unexpected OHLCV feature schema")
    if manifest.get("universe_selection", {}).get("uses_test_values") is not False:
        raise ValueError("universe selection must explicitly exclude test values")
    if manifest.get("universe_selection", {}).get("selection_prefix_rows") != 256:
        raise ValueError("Phase 3 requires the fixed 256-row universe-selection prefix")

    with np.load(npz_path, allow_pickle=False) as bundle:
        missing = [name for name in REQUIRED_ARRAYS if name not in bundle.files]
        if missing:
            raise ValueError(f"processed bundle is missing arrays: {missing}")
        arrays = {name: np.asarray(bundle[name]) for name in REQUIRED_ARRAYS}

    for split in ("train", "test"):
        sequences = arrays[split]
        if sequences.dtype != np.float32 or sequences.ndim != 3:
            raise ValueError(f"{split} must be float32 [N, sequence, feature]")
        if sequences.shape[1:] != (int(manifest["seq_len"]), len(FEATURE_COLUMNS)):
            raise ValueError(f"{split} shape disagrees with manifest")
        if not np.isfinite(sequences).all():
            raise ValueError(f"{split} contains non-finite values")
        count = len(sequences)
        for suffix in ("contract_ids", "window_starts", "window_ends", "timestamps_ns"):
            if len(arrays[f"{split}_{suffix}"]) != count:
                raise ValueError(f"{split}_{suffix} length mismatch")
        actual_identity = sha256_arrays(
            arrays[f"{split}_contract_ids"], arrays[f"{split}_window_starts"],
            arrays[f"{split}_window_ends"], arrays[f"{split}_timestamps_ns"],
        )
        if actual_identity != manifest["identity_hashes"][split]:
            raise ValueError(f"{split} identity hash mismatch")
        if sha256_arrays(sequences) != manifest["sequence_hashes"][split]:
            raise ValueError(f"{split} sequence hash mismatch")

    contract_records = manifest.get("contracts", [])
    if len(contract_records) != manifest.get("top_k"):
        raise ValueError("selected contract count does not equal top_k")
    train_ids = arrays["train_contract_ids"]
    test_ids = arrays["test_contract_ids"]
    for record in contract_records:
        contract_id = int(record["contract_id"])
        boundary = int(record["raw_boundary_index"])
        fit_end = int(record["preprocessing"]["fit_end_exclusive"])
        if fit_end != boundary:
            raise ValueError(f"contract {contract_id}: preprocessing was not fit to the raw boundary")
        train_mask = train_ids == contract_id
        test_mask = test_ids == contract_id
        if not train_mask.any() or not test_mask.any():
            raise ValueError(f"contract {contract_id}: both splits must have windows")
        if int(arrays["train_window_ends"][train_mask].max()) >= boundary:
            raise ValueError(f"contract {contract_id}: train window reaches test raw rows")
        if int(arrays["test_window_starts"][test_mask].min()) < boundary:
            raise ValueError(f"contract {contract_id}: test window reaches train raw rows")
        train_rows = set()
        for start, end in zip(
            arrays["train_window_starts"][train_mask], arrays["train_window_ends"][train_mask]
        ):
            train_rows.update(range(int(start), int(end) + 1))
        for start, end in zip(
            arrays["test_window_starts"][test_mask], arrays["test_window_ends"][test_mask]
        ):
            if any(row in train_rows for row in range(int(start), int(end) + 1)):
                raise ValueError(f"contract {contract_id}: train/test raw-row overlap")

    expected_sha = manifest.get("processed_npz_sha256")
    actual_sha = sha256_file(npz_path)
    if expected_sha and expected_sha != actual_sha:
        raise ValueError("processed NPZ file hash mismatch")
    return {
        "valid": True,
        "processed_npz": str(npz_path),
        "processed_npz_sha256": actual_sha,
        "train_shape": list(arrays["train"].shape),
        "test_shape": list(arrays["test"].shape),
        "contract_count": len(contract_records),
        "identity_hashes": manifest["identity_hashes"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-npz", type=Path, default=DEFAULT_PROCESSED_NPZ)
    args = parser.parse_args(argv)
    try:
        result = validate_bundle(args.processed_npz)
        print(f"Phase 3 encoder data valid: {result}")
        return 0
    except Exception as exc:
        print(f"Phase 3 encoder data validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
