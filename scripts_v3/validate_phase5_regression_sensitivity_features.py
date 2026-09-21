#!/usr/bin/env python3
"""Replay-validate the Phase 5 eight-hour sensitivity feature stores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features.phase5_features import validate_phase5_feature_bundle


def main() -> None:
    results = []
    for walk in (1, 2):
        feature = Path(
            f"experiments/phase5/features_sensitivities/raw_delta_h8/walk{walk}/five_branch_epoch50.npz"
        )
        dataset = Path(
            f"experiments/phase5/data_sensitivities/raw_delta_h8/walk{walk}/market_1h_seq64_h8.npz"
        )
        results.append(validate_phase5_feature_bundle(feature, dataset_path=dataset))
    print(json.dumps({"valid": True, "walks": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
