#!/usr/bin/env python3
"""Replay hashes, source closes, and invariants of Phase 6 H=8 labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase6_volatility_labels import (  # noqa: E402
    validate_volatility_label_bundle_files,
)


DEFAULT_OUTPUT_ROOT = (
    ROOT / "experiments" / "phase6" / "volatility_prediction" / "data_preparation"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", choices=("1", "2", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-source-replay", action="store_true")
    args = parser.parse_args(argv)
    try:
        selected = (1, 2) if args.walk == "all" else (int(args.walk),)
        results = []
        for walk in selected:
            path = args.output_root / f"walk{walk}" / "volatility_1h_seq64_h8.npz"
            results.append(
                validate_volatility_label_bundle_files(
                    path, replay_source=not args.skip_source_replay
                )
            )
        print(json.dumps({"valid": True, "walks": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 volatility label validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
