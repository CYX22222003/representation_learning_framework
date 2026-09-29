#!/usr/bin/env python3
"""Replay and report the complete Phase 6.5A LSTM-capacity matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.phase6_5_lstm_capacity_reporting import (  # noqa: E402
    build_lstm_capacity_report,
)


def main() -> int:
    try:
        print(json.dumps(build_lstm_capacity_report(ROOT), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5A reporting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
