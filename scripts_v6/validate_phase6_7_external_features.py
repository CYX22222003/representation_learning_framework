#!/usr/bin/env python3
"""Replay the two SaURL Phase 6.7 master representation stores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.phase6_7_external_features import METHOD, validate_external_feature_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    try:
        results = []
        for walk in (1, 2):
            store = (
                ROOT
                / "experiments"
                / "phase6_7"
                / "features"
                / METHOD
                / f"walk{walk}"
                / "representations.npz"
            )
            results.append(
                validate_external_feature_store(
                    store, replay=True, device=args.device, batch_size=args.batch_size
                )
            )
        print(json.dumps({"valid": True, "stores": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.7 feature validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
