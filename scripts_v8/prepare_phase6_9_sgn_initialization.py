#!/usr/bin/env python3
"""Build the two train-only SGN-C BDC/group/period initializations."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.phase6_9_sgn import prepare_initialization  # noqa: E402


def main() -> int:
    try:
        results = [prepare_initialization(ROOT, walk) for walk in (1, 2)]
        print(json.dumps({"valid": True, "results": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"SGN-C initialization failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
