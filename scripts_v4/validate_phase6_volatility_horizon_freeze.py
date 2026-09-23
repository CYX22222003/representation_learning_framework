#!/usr/bin/env python3
"""Replay the approved Phase 6 H=8 volatility horizon freeze."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase6_volatility import validate_horizon_freeze_manifest  # noqa: E402


DEFAULT_FREEZE_PATH = (
    ROOT
    / "experiments"
    / "phase6"
    / "volatility_prediction"
    / "manifests"
    / "horizon_freeze_h8.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-path", type=Path, default=DEFAULT_FREEZE_PATH)
    args = parser.parse_args(argv)
    try:
        result = validate_horizon_freeze_manifest(args.freeze_path, ROOT)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 volatility horizon-freeze validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
