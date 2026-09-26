#!/usr/bin/env python3
"""Replay and report the complete Phase 6 temporal encoder matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.phase6_encoder_variant_reporting import build_complete_report  # noqa: E402


def main() -> int:
    try:
        print(json.dumps(build_complete_report(ROOT), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 encoder-variant reporting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
