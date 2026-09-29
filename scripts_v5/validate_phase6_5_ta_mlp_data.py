#!/usr/bin/env python3
"""Replay-validate both Phase 6.5C causal TA feature stores."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.ta_mlp_baseline.phase6_5 import validate_phase65_ta_store  # noqa: E402


def main() -> int:
    try:
        results = []
        for walk in (1, 2):
            path = ROOT / "experiments" / "phase6_5" / "ta_mlp" / "data_preparation" / f"walk{walk}" / "classification_h2_ta36.npz"
            results.append(validate_phase65_ta_store(path))
        print(json.dumps({"valid": True, "stores": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5C TA validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
