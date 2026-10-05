#!/usr/bin/env python3
"""Freeze the minimal six-file TimeDART data dependency manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase6_7_timedart_data import prepare_timedart_data_manifest  # noqa: E402


OUTPUT = ROOT / "experiments" / "phase6_7" / "manifests" / "timedart_data.json"


def main() -> int:
    try:
        payload = prepare_timedart_data_manifest(ROOT, OUTPUT)
        print(json.dumps({"valid": True, "manifest": str(OUTPUT.resolve()), "data": payload}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"TimeDART data preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

