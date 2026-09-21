#!/usr/bin/env python3
"""Replay-validate canonical Phase 5 feature stores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features.phase5_features import validate_phase5_feature_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", type=int, choices=(1, 2), action="append")
    args = parser.parse_args()
    walks = args.walk or [1, 2]
    results = []
    for walk in walks:
        results.append(
            validate_phase5_feature_bundle(
                Path(f"experiments/phase5/features/walk{walk}/five_branch_epoch50.npz"),
                dataset_path=Path(
                    f"experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz"
                ),
            )
        )
    print(json.dumps({"valid": True, "walks": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
