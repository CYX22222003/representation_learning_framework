#!/usr/bin/env python3
"""Replay-validate both completed SGN-C walk trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.phase6_9_sgn import SGNTrainingConfig, validate_sgn_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        results = [validate_sgn_training(ROOT, SGNTrainingConfig(walk, device=args.device)) for walk in (1, 2)]
        print(json.dumps({"valid": True, "results": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"SGN-C validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
