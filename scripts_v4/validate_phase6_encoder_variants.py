#!/usr/bin/env python3
"""Replay all completed Phase 6 temporal encoder artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.encoder_variants import BYOL_VARIANTS, CONTRASTIVE_VARIANTS  # noqa: E402
from training.phase6_encoder_variants import validate_encoder_variant  # noqa: E402


def main() -> int:
    try:
        results = []
        for walk in (1, 2):
            dataset = ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
            for variant in (*CONTRASTIVE_VARIANTS, *BYOL_VARIANTS):
                family, backbone = variant.split("_", 1)
                run_root = ROOT / "experiments" / "phase6" / "encoder_variants" / "pretraining" / f"walk{walk}" / family / backbone / "seed0"
                results.append(validate_encoder_variant(dataset, run_root))
        print(json.dumps({"valid": True, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 temporal encoder validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
