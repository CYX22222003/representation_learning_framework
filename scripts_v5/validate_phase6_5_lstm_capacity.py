#!/usr/bin/env python3
"""Replay-validate all four frozen Phase 6.5A encoder trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.phase6_5_lstm_capacity import (  # noqa: E402
    VARIANTS,
    validate_lstm_capacity_encoder,
)


def main() -> int:
    try:
        results = []
        capacity_root = ROOT / "experiments" / "phase6_5" / "lstm_capacity"
        for walk in (1, 2):
            dataset = (
                ROOT
                / "experiments"
                / "phase5"
                / "data_preparation"
                / f"walk{walk}"
                / "market_1h_seq64_h2.npz"
            )
            for variant in VARIANTS:
                family = "contrastive" if variant.startswith("contrastive") else "byol"
                run_root = (
                    capacity_root
                    / "pretraining"
                    / f"walk{walk}"
                    / family
                    / "lstm2"
                    / "seed0"
                )
                results.append(validate_lstm_capacity_encoder(dataset, run_root))
        print(json.dumps({"valid": True, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5A validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
