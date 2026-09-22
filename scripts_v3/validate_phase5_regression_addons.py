#!/usr/bin/env python3
"""Replay-validate all Phase 5 exploratory regression regression add-on runs."""

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

from launch_phase5_regression_addons import paths
from training.phase5_regression_addons import validate_regression_addon_run


def main() -> None:
    results = []
    for task in ("raw_delta_h8", "log_return_h2"):
        for walk in (1, 2):
            dataset, feature, scaler, run_root = paths(task, walk)
            results.append(validate_regression_addon_run(dataset, feature, scaler, run_root))
    print(json.dumps({"valid": True, "runs": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
