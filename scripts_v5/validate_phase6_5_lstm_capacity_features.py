#!/usr/bin/env python3
"""Validate all six Phase 6.5A task/walk feature stores."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.phase6_5_lstm_capacity_features import (  # noqa: E402
    TASKS,
    validate_lstm_capacity_feature_store,
)


def main() -> int:
    try:
        root = ROOT / "experiments" / "phase6_5" / "lstm_capacity" / "features"
        results = [
            validate_lstm_capacity_feature_store(root / f"walk{walk}" / f"{task}.npz")
            for walk in (1, 2)
            for task in TASKS
        ]
        print(json.dumps({"valid": True, "stores": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5A feature validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
