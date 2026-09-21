#!/usr/bin/env python3
"""Replay-validate the Phase 5 eight-hour raw-change sensitivity bundles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_processing.phase5_walks import validate_phase5_bundle_files


def main() -> None:
    results = []
    for walk in (1, 2):
        sensitivity = Path(
            f"experiments/phase5/data_sensitivities/raw_delta_h8/walk{walk}/market_1h_seq64_h8.npz"
        )
        result = validate_phase5_bundle_files(sensitivity)
        primary = Path(
            f"experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz"
        )
        with np.load(primary, allow_pickle=False) as h2, np.load(
            sensitivity, allow_pickle=False
        ) as h8:
            for key in (
                "encoder_train_condition_ids",
                "encoder_train_window_start_ns",
                "encoder_train_decision_date_ns",
                "encoder_train_decision_availability_ns",
                "encoder_train_sequences",
                "encoder_train_raw_sequences",
            ):
                if not np.array_equal(h2[key], h8[key]):
                    raise ValueError(f"walk {walk} h8 encoder mismatch: {key}")
        result["encoder_population_matches_primary_h2"] = True
        results.append(result)
    print(json.dumps({"valid": True, "walks": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
