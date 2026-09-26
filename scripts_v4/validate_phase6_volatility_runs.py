#!/usr/bin/env python3
"""Replay completed ready-now Phase 6 volatility neural trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.phase6_volatility import MODELS, validate_volatility_run  # noqa: E402


def main() -> int:
    root = ROOT / "experiments" / "phase6" / "volatility_prediction"
    try:
        results = []
        for walk in (1, 2):
            label = root / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz"
            features = root / "features" / f"walk{walk}" / "five_branch_epoch50_h8.npz"
            for model in MODELS:
                results.append(
                    validate_volatility_run(
                        label,
                        features if model == "framework_h0" else None,
                        root / "runs" / model / f"walk{walk}" / "seed0",
                    )
                )
        print(json.dumps({"valid": True, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 volatility run validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
