#!/usr/bin/env python3
"""Extract task-aligned Phase 6.5A two-layer LSTM feature stores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.phase6_5_lstm_capacity_features import (  # noqa: E402
    TASKS,
    build_lstm_capacity_feature_store,
    validate_lstm_capacity_feature_store,
)


def task_paths(task: str, walk: int) -> tuple[Path, Path]:
    if task == "classification_h2":
        return (
            ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz",
            ROOT / "experiments" / "phase5" / "features" / f"walk{walk}" / "five_branch_epoch50.npz",
        )
    if task == "absolute_price_h8":
        base = ROOT / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8"
        return (
            base / "data" / f"walk{walk}" / "market_1h_seq64_h8.npz",
            base / "features" / f"walk{walk}" / "five_branch_epoch50.npz",
        )
    base = ROOT / "experiments" / "phase6" / "volatility_prediction"
    return (
        base / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz",
        base / "features" / f"walk{walk}" / "five_branch_epoch50_h8.npz",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--walks", default="1,2")
    parser.add_argument("--tasks", default=",".join(TASKS))
    args = parser.parse_args()
    try:
        walks = tuple(int(value.strip()) for value in args.walks.split(",") if value.strip())
        tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
        if not walks or any(walk not in (1, 2) for walk in walks):
            raise ValueError("--walks must contain 1 and/or 2")
        if not tasks or any(task not in TASKS for task in tasks):
            raise ValueError(f"--tasks must contain only {TASKS}")
        results = []
        root = ROOT / "experiments" / "phase6_5" / "lstm_capacity"
        for walk in walks:
            encoder_dataset = ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
            for task in tasks:
                dataset, h0 = task_paths(task, walk)
                output = root / "features" / f"walk{walk}" / f"{task}.npz"
                if output.is_file() and Path(f"{output}.manifest.json").is_file():
                    result = validate_lstm_capacity_feature_store(output)
                    result["action"] = "validated_existing"
                else:
                    result = build_lstm_capacity_feature_store(
                        dataset,
                        h0,
                        encoder_dataset,
                        root,
                        output,
                        task=task,
                        walk=walk,
                        device=args.device,
                        batch_size=args.batch_size,
                    )
                    result["action"] = "extracted"
                results.append(result)
        print(json.dumps({"valid": True, "stores": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5A feature preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
