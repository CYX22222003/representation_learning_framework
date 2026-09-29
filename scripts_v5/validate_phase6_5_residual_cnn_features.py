#!/usr/bin/env python3
"""Validate both Phase 6.5D future-price feature stores."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.phase6_5_residual_cnn_features import (  # noqa: E402
    validate_residual_cnn_feature_store,
)


def main() -> int:
    try:
        root = ROOT / "experiments" / "phase6_5" / "residual_cnn" / "features"
        results = [
            validate_residual_cnn_feature_store(
                root / f"walk{walk}" / "absolute_price_h8" / "features.npz"
            )
            for walk in (1, 2)
        ]
        print(json.dumps({"valid": True, "stores": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5D feature validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
