#!/usr/bin/env python3
"""Replay six LWA probes and the matching immutable H0 references."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.phase6_encoder_variant_reporting import validate_h0_reference  # noqa: E402
from features.phase6_7_lwa_features import METHOD, TASKS  # noqa: E402
from training.phase6_7_downstream import validate_external_downstream  # noqa: E402


def task_dataset(task: str, walk: int) -> Path:
    if task == "classification_h2":
        return ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
    if task == "absolute_price_h8":
        return ROOT / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8" / "data" / f"walk{walk}" / "market_1h_seq64_h8.npz"
    return ROOT / "experiments" / "phase6" / "volatility_prediction" / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz"


def main() -> int:
    try:
        results = []
        phase = ROOT / "experiments" / "phase6_7"
        for task in TASKS:
            for walk in (1, 2):
                dataset = task_dataset(task, walk)
                feature = phase / "features" / METHOD / f"walk{walk}" / "representations.npz"
                run = phase / "downstream" / task / METHOD / f"walk{walk}" / "seed0"
                results.append(validate_external_downstream(dataset, feature, run))
                results.append({**validate_h0_reference(ROOT, task, walk), "task": task, "walk": walk, "method": "H0", "immutable_reference": True})
        print(json.dumps({"valid": True, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.7 LWA downstream validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
