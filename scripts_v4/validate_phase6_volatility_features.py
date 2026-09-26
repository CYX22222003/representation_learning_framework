#!/usr/bin/env python3
"""Replay Phase 6 volatility identities against frozen canonical features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.phase6_volatility_features import (  # noqa: E402
    validate_phase6_volatility_feature_bundle,
)


DEFAULT_OUTPUT_ROOT = (
    ROOT / "experiments" / "phase6" / "volatility_prediction" / "features"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", choices=("1", "2", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        selected = (1, 2) if args.walk == "all" else (int(args.walk),)
        results = [
            validate_phase6_volatility_feature_bundle(
                args.output_root / f"walk{walk}" / "five_branch_epoch50_h8.npz"
            )
            for walk in selected
        ]
        print(json.dumps({"valid": True, "walks": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 volatility feature validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
