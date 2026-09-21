"""Validate saved Phase 5 walk bundles and their source lineage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import validate_phase5_bundle_files  # noqa: E402


DEFAULT_ROOT = ROOT / "experiments" / "phase5" / "data_preparation"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--walk", choices=("1", "2", "all"), default="all")
    args = parser.parse_args(argv)
    walks = (1, 2) if args.walk == "all" else (int(args.walk),)
    try:
        for walk in walks:
            npz_path = args.root / f"walk{walk}" / "market_1h_seq64_h2.npz"
            result = validate_phase5_bundle_files(npz_path)
            print(f"Phase 5 walk {walk} data valid: {result}", flush=True)
        return 0
    except Exception as exc:
        print(f"Phase 5 data validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
