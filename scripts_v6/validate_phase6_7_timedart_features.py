#!/usr/bin/env python3
"""Advisory replay of both TimeDART master feature stores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase6_7_timedart_data import METHOD  # noqa: E402
from features.phase6_7_timedart_features import validate_timedart_feature_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    results = []
    for walk in (1, 2):
        path = ROOT / "experiments" / "phase6_7" / "features" / METHOD / f"walk{walk}" / "representations.npz"
        try:
            results.append(validate_timedart_feature_store(path, replay=args.replay, device=args.device, batch_size=args.batch_size))
        except Exception as exc:
            results.append({"valid": False, "status": "warning", "continued": True, "walk": walk, "message": str(exc)})
    print(json.dumps({"status": "warning" if any(not row.get("valid") for row in results) else "pass", "non_blocking": True, "stores": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

