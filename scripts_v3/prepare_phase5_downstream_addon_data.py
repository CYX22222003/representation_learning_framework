#!/usr/bin/env python3
"""Build leakage-safe eight-hour data for additional Phase 5 downstream tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts_v3") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts_v3"))

from data_processing.phase5_walks import (
    build_phase5_walk_bundle,
    sha256_file,
    validate_phase5_bundle_files,
)
from prepare_phase5_data import canonical_walk_inputs


OUTPUT_ROOT = (
    ROOT / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8" / "data"
)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_walk(walk: int) -> dict[str, object]:
    walk_input = canonical_walk_inputs()[walk]
    input_dir = walk_input.input_dir.resolve()
    candles_path = input_dir / "candles_1h_clean_ffill1.parquet"
    metadata_path = input_dir / "market_metadata.parquet"
    fill_manifest_path = input_dir / "fill_manifest.json"
    for path in (candles_path, metadata_path, fill_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    walk_dir = OUTPUT_ROOT / f"walk{walk}"
    output = walk_dir / "market_1h_seq64_h8.npz"
    manifest_path = Path(f"{output}.manifest.json")
    if output.exists() or manifest_path.exists():
        return validate_phase5_bundle_files(output)
    walk_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "candles_1h_clean_ffill1": {
            "path": str(candles_path),
            "sha256": sha256_file(candles_path),
        },
        "market_metadata": {
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
        },
        "fill_manifest": {
            "path": str(fill_manifest_path),
            "sha256": sha256_file(fill_manifest_path),
        },
    }
    bundle = build_phase5_walk_bundle(
        pd.read_parquet(candles_path),
        pd.read_parquet(metadata_path),
        walk_input.spec,
        horizon=8,
        task_role="exploratory_raw_delta_h8",
        source_provenance=provenance,
    )
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **bundle.arrays)
    os.replace(temporary, output)
    manifest = {
        **bundle.manifest,
        "artifact": {
            "path": str(output),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
        },
        "exploratory_status": (
            "additional post-primary regression task; not selected from evaluation performance"
        ),
    }
    _write_json(manifest_path, manifest)
    result = validate_phase5_bundle_files(output)

    primary = ROOT / f"experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz"
    with np.load(primary, allow_pickle=False) as h2, np.load(output, allow_pickle=False) as h8:
        for key in (
            "encoder_train_condition_ids",
            "encoder_train_window_start_ns",
            "encoder_train_decision_date_ns",
            "encoder_train_decision_availability_ns",
            "encoder_train_sequences",
            "encoder_train_raw_sequences",
        ):
            if not np.array_equal(h2[key], h8[key]):
                raise ValueError(f"h8 add-on changed target-free encoder array: {key}")
    result["encoder_population_matches_primary_h2"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", choices=("1", "2", "all"), default="all")
    args = parser.parse_args()
    selected = (1, 2) if args.walk == "all" else (int(args.walk),)
    results = [prepare_walk(walk) for walk in selected]
    print(json.dumps({"valid": True, "walks": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
