#!/usr/bin/env python3
"""Freeze, execute, and replay-check the four-run Phase 5 downstream matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features.phase5_features import validate_phase5_feature_bundle
from training.phase5_downstream import (
    Phase5DownstreamConfig,
    prepare_feature_standardizer,
    run_downstream_training,
    smoke_test_heads,
    validate_downstream_run,
)
from training.phase5_encoder import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    prerequisites = []
    scalers = {}
    for walk in (1, 2):
        dataset = Path(f"experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz")
        feature = Path(f"experiments/phase5/features/walk{walk}/five_branch_epoch50.npz")
        feature_validation = validate_phase5_feature_bundle(feature, dataset_path=dataset)
        scaler = Path(f"experiments/phase5/downstream/walk{walk}/feature_standardizer.npz")
        scaler_validation = prepare_feature_standardizer(feature, scaler, dataset_path=dataset)
        prerequisites.append({"walk": walk, "features": feature_validation, "scaler": scaler_validation})
        scalers[walk] = scaler
    smoke = smoke_test_heads()
    matrix_path = Path("experiments/phase5/manifests/framework_downstream_seed0.json")
    matrix = {
        "phase": 5,
        "purpose": "frozen_framework_downstream_seed0_matrix",
        "frozen_before_training": True,
        "runs": [],
        "prerequisites": prerequisites,
        "cpu_smoke_test": smoke,
    }
    for walk in (1, 2):
        for task in ("regression", "classification"):
            config = Phase5DownstreamConfig(task=task, walk=walk, device=args.device)
            matrix["runs"].append(
                {
                    "walk": walk,
                    "task": task,
                    "seed": 0,
                    "config": config.to_dict(),
                    "dataset_sha256": sha256_file(Path(f"experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz")),
                    "feature_sha256": sha256_file(Path(f"experiments/phase5/features/walk{walk}/five_branch_epoch50.npz")),
                    "scaler_sha256": sha256_file(scalers[walk]),
                }
            )
    if matrix_path.is_file():
        existing = json.loads(matrix_path.read_text(encoding="utf-8"))
        if existing != matrix:
            raise ValueError("existing frozen downstream matrix differs from requested execution")
    else:
        write_json(matrix_path, matrix)

    results = []
    for row in matrix["runs"]:
        walk = int(row["walk"])
        task = str(row["task"])
        dataset = Path(f"experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz")
        feature = Path(f"experiments/phase5/features/walk{walk}/five_branch_epoch50.npz")
        run_root = Path(f"experiments/phase5/downstream/walk{walk}/{task}/seed0")
        if (run_root / "training_complete.json").is_file():
            result = validate_downstream_run(dataset, feature, scalers[walk], run_root)
            result["action"] = "validated_existing"
        else:
            run_downstream_training(
                dataset,
                feature,
                scalers[walk],
                run_root,
                Phase5DownstreamConfig(task=task, walk=walk, device=args.device),
            )
            result = validate_downstream_run(dataset, feature, scalers[walk], run_root)
            result["action"] = "trained"
        results.append(result)
    print(json.dumps({"complete": True, "runs": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
