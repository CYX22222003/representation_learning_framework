#!/usr/bin/env python3
"""Replay-validate all 24 Phase 6.5A downstream trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.phase6_5_lstm_capacity_features import CONFIG_BRANCHES, TASKS  # noqa: E402
from training.phase6_5_lstm_capacity_downstream import (  # noqa: E402
    validate_lstm_capacity_downstream,
)


def _task_paths(task: str, walk: int) -> tuple[Path, Path]:
    root = ROOT / "experiments" / "phase6_5" / "lstm_capacity"
    if task == "classification_h2":
        dataset = ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
    elif task == "absolute_price_h8":
        dataset = ROOT / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8" / "data" / f"walk{walk}" / "market_1h_seq64_h8.npz"
    else:
        dataset = ROOT / "experiments" / "phase6" / "volatility_prediction" / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz"
    return dataset, root / "features" / f"walk{walk}" / f"{task}.npz"


def main() -> int:
    try:
        results = []
        root = ROOT / "experiments" / "phase6_5" / "lstm_capacity"
        for task in TASKS:
            for walk in (1, 2):
                dataset, feature = _task_paths(task, walk)
                for configuration in CONFIG_BRANCHES:
                    if configuration == "H0":
                        continue
                    run_root = root / "downstream" / task / configuration.lower() / f"walk{walk}" / "seed0"
                    results.append(
                        validate_lstm_capacity_downstream(dataset, feature, run_root)
                    )
        print(json.dumps({"valid": True, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5A downstream validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
