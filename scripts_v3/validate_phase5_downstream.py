#!/usr/bin/env python3
"""Replay-validate all Phase 5 feature scalers and framework downstream runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.phase5_downstream import validate_downstream_run, validate_feature_standardizer


def main() -> None:
    scalers = []
    runs = []
    for walk in (1, 2):
        dataset = Path(f"experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz")
        feature = Path(f"experiments/phase5/features/walk{walk}/five_branch_epoch50.npz")
        scaler = Path(f"experiments/phase5/downstream/walk{walk}/feature_standardizer.npz")
        scalers.append(validate_feature_standardizer(feature, scaler, dataset_path=dataset))
        for task in ("regression", "classification"):
            runs.append(
                validate_downstream_run(
                    dataset,
                    feature,
                    scaler,
                    Path(f"experiments/phase5/downstream/walk{walk}/{task}/seed0"),
                )
            )
    print(json.dumps({"valid": True, "scalers": scalers, "runs": runs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
