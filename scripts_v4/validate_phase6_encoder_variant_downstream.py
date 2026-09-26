#!/usr/bin/env python3
"""Replay all Phase 6 temporal/control downstream trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.phase6_encoder_variant_features import CONFIG_BRANCHES, TASKS  # noqa: E402
from evaluation.phase6_encoder_variant_reporting import validate_h0_reference  # noqa: E402
from training.phase6_encoder_variant_downstream import validate_variant_downstream  # noqa: E402


def task_dataset(task: str, walk: int) -> Path:
    if task == "classification_h2":
        return ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
    if task == "absolute_price_h8":
        return ROOT / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8" / "data" / f"walk{walk}" / "market_1h_seq64_h8.npz"
    return ROOT / "experiments" / "phase6" / "volatility_prediction" / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz"


def main() -> int:
    try:
        base = ROOT / "experiments" / "phase6" / "encoder_variants"
        results = []
        for task in TASKS:
            for walk in (1, 2):
                dataset = task_dataset(task, walk)
                feature = base / "features" / f"walk{walk}" / f"{task}.npz"
                for configuration in CONFIG_BRANCHES:
                    if configuration == "H0":
                        validation = validate_h0_reference(ROOT, task, walk)
                        results.append(
                            {
                                **validation,
                                "task": task,
                                "walk": walk,
                                "configuration": "H0",
                                "immutable_reference": True,
                            }
                        )
                    else:
                        run = base / "downstream" / task / configuration.lower() / f"walk{walk}" / "seed0"
                        results.append(validate_variant_downstream(dataset, feature, run))
        print(json.dumps({"valid": True, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 variant downstream validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
