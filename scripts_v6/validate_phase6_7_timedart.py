#!/usr/bin/env python3
"""Advisory validation of both TimeDART encoder trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase6_7_timedart_data import METHOD, encoder_dataset_path  # noqa: E402
from training.phase6_7_timedart import validate_timedart_pretraining  # noqa: E402


def main() -> int:
    results = []
    for walk in (1, 2):
        try:
            results.append(validate_timedart_pretraining(
                encoder_dataset_path(ROOT, walk),
                ROOT / "experiments" / "phase6_7" / "encoder_pretraining" / METHOD / f"walk{walk}" / "seed0",
            ))
        except Exception as exc:
            results.append({"valid": False, "status": "warning", "continued": True, "walk": walk, "message": str(exc)})
    print(json.dumps({"status": "warning" if any(not row.get("valid") for row in results) else "pass", "non_blocking": True, "runs": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

