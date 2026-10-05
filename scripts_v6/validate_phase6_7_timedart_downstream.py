#!/usr/bin/env python3
"""Advisory replay of all six TimeDART downstream trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase6_7_timedart_data import METHOD, TASKS, timedart_dataset_paths  # noqa: E402
from training.phase6_7_downstream import validate_external_downstream  # noqa: E402


def main() -> int:
    paths = timedart_dataset_paths(ROOT)
    results = []
    for task in TASKS:
        for walk in (1, 2):
            feature = ROOT / "experiments" / "phase6_7" / "features" / METHOD / f"walk{walk}" / "representations.npz"
            run_root = ROOT / "experiments" / "phase6_7" / "downstream" / task / METHOD / f"walk{walk}" / "seed0"
            try:
                results.append(validate_external_downstream(paths[walk][task], feature, run_root))
            except Exception as exc:
                results.append({"valid": False, "status": "warning", "continued": True, "task": task, "walk": walk, "message": str(exc)})
    print(json.dumps({"status": "warning" if any(not row.get("valid") for row in results) else "pass", "non_blocking": True, "runs": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

