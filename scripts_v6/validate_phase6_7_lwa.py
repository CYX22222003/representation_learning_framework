#!/usr/bin/env python3
"""Replay both walk-specific LWA caches and two-stage trajectories on CPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.phase6_7_lwa import METHOD, validate_lwa_pretraining  # noqa: E402


def main() -> int:
    try:
        results = []
        for walk in (1, 2):
            dataset = ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
            cache = ROOT / "experiments" / "phase6_7" / "view_cache" / METHOD / f"walk{walk}"
            run = ROOT / "experiments" / "phase6_7" / "encoder_pretraining" / METHOD / f"walk{walk}" / "seed0"
            results.append(validate_lwa_pretraining(dataset, cache, run))
        print(json.dumps({"valid": True, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.7 LWA validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
