#!/usr/bin/env python3
"""Build or replay the two frozen observed eight-hour OHLCV path bundles."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_processing.phase6_9_xlstm_mixer import prepare_future_path_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", choices=("1", "2", "all"), default="all")
    args = parser.parse_args()
    try:
        walks = (1, 2) if args.walk == "all" else (int(args.walk),)
        print(json.dumps({"valid": True, "walks": [prepare_future_path_data(ROOT, w) for w in walks]}, indent=2))
        return 0
    except Exception as exc:
        print(f"XM-MV8 data preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
