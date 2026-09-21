#!/usr/bin/env python3
"""Train one frozen Phase 5 framework downstream trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.phase5_downstream import (
    Phase5DownstreamConfig,
    prepare_feature_standardizer,
    run_downstream_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", type=int, choices=(1, 2), required=True)
    parser.add_argument("--task", choices=("regression", "classification"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dataset = Path(f"experiments/phase5/data_preparation/walk{args.walk}/market_1h_seq64_h2.npz")
    feature = Path(f"experiments/phase5/features/walk{args.walk}/five_branch_epoch50.npz")
    scaler = Path(f"experiments/phase5/downstream/walk{args.walk}/feature_standardizer.npz")
    prepare_feature_standardizer(feature, scaler, dataset_path=dataset)
    run_root = Path(f"experiments/phase5/downstream/walk{args.walk}/{args.task}/seed0")
    snapshots = run_downstream_training(
        dataset,
        feature,
        scaler,
        run_root,
        Phase5DownstreamConfig(task=args.task, walk=args.walk, device=args.device),
    )
    print(json.dumps({"complete": True, "snapshots": snapshots}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
