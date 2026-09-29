#!/usr/bin/env python3
"""Replay-validate completed Phase 6.5B GARCH--LSTM runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.garch_lstm_stacking.phase6_5 import validate_phase65_garch_lstm_run  # noqa: E402


def main() -> int:
    try:
        results = []
        for walk in (1, 2):
            label = ROOT / "experiments" / "phase6" / "volatility_prediction" / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz"
            raw = ROOT / "experiments" / "phase6" / "volatility_prediction" / "runs" / "raw_lstm" / f"walk{walk}" / "seed0"
            run = ROOT / "experiments" / "phase6_5" / "garch_lstm" / f"walk{walk}" / "seed0"
            results.append(validate_phase65_garch_lstm_run(label, raw, run))
        print(json.dumps({"valid": True, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5B validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
