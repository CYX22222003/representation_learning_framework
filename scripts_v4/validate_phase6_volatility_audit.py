#!/usr/bin/env python3
"""Validate hashes and the training-only contract of the Phase 6 audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase6_volatility import validate_audit_artifacts  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "phase6" / "volatility_prediction" / "data_exploration"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        result = validate_audit_artifacts(args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 volatility audit validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
