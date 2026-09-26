#!/usr/bin/env python3
"""Validate the six Phase 6 task/walk variant feature stores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.phase6_encoder_variant_features import TASKS, validate_variant_feature_store  # noqa: E402


def main() -> int:
    try:
        base = ROOT / "experiments" / "phase6" / "encoder_variants" / "features"
        results = [
            validate_variant_feature_store(base / f"walk{walk}" / f"{task}.npz")
            for walk in (1, 2) for task in TASKS
        ]
        print(json.dumps({"valid": True, "stores": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 variant feature validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
