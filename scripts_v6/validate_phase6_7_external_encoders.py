#!/usr/bin/env python3
"""Replay both SaURL Phase 6.7 encoder trajectories on CPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.phase6_7_external_encoders import METHODS, validate_external_encoder  # noqa: E402


def main() -> int:
    try:
        results = []
        for walk in (1, 2):
            dataset = (
                ROOT
                / "experiments"
                / "phase5"
                / "data_preparation"
                / f"walk{walk}"
                / "market_1h_seq64_h2.npz"
            )
            for method in METHODS:
                run_root = (
                    ROOT
                    / "experiments"
                    / "phase6_7"
                    / "encoder_pretraining"
                    / method
                    / f"walk{walk}"
                    / "seed0"
                )
                results.append(validate_external_encoder(dataset, run_root))
        print(json.dumps({"valid": True, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.7 encoder validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
