#!/usr/bin/env python3
"""Replay-validate both Phase 5 eight-hour absolute-price runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts_v3") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts_v3"))

from launch_phase5_absolute_price_h8 import paths
from training.phase5_absolute_price import validate_absolute_price_run


def main() -> None:
    results = []
    for walk in (1, 2):
        dataset, feature, scaler, run_root = paths(walk)
        results.append(validate_absolute_price_run(dataset, feature, scaler, run_root))
    print(json.dumps({"valid": True, "runs": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
