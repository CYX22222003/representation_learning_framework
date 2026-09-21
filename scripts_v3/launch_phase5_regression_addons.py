#!/usr/bin/env python3
"""Freeze and execute the four-run Phase 5 additional regression-task matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.phase5_downstream import prepare_feature_standardizer
from training.phase5_encoder import sha256_file, write_json
from training.phase5_regression_addons import (
    RegressionAddonConfig,
    run_regression_addon,
    validate_regression_addon_run,
)


def paths(task: str, walk: int) -> tuple[Path, Path, Path, Path]:
    if task == "raw_delta_h8":
        dataset = Path(
            f"experiments/phase5/downstream_addons/shared/h8/data/walk{walk}/market_1h_seq64_h8.npz"
        )
        feature = Path(
            f"experiments/phase5/downstream_addons/shared/h8/features/walk{walk}/five_branch_epoch50.npz"
        )
        scaler = Path(
            f"experiments/phase5/downstream_addons/shared/h8/feature_scalers/walk{walk}/feature_standardizer.npz"
        )
    else:
        dataset = Path(
            f"experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz"
        )
        feature = Path(
            f"experiments/phase5/features/walk{walk}/five_branch_epoch50.npz"
        )
        scaler = Path(f"experiments/phase5/downstream/walk{walk}/feature_standardizer.npz")
    run_root = Path(
        f"experiments/phase5/downstream_addons/tasks/{task}/walk{walk}/seed0"
    )
    return dataset, feature, scaler, run_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    matrix_path = Path("experiments/phase5/downstream_addons/manifests/regression_tasks_seed0.json")
    matrix: dict[str, object] = {
        "phase": 5,
        "purpose": "frozen_additional_regression_task_matrix",
        "exploratory_status": "additional post-primary regression tasks",
        "frozen_before_training": True,
        "runs": [],
    }
    for task in ("raw_delta_h8", "log_return_h2"):
        for walk in (1, 2):
            dataset, feature, scaler, run_root = paths(task, walk)
            if task == "raw_delta_h8":
                prepare_feature_standardizer(feature, scaler, dataset_path=dataset)
            config = RegressionAddonConfig(task=task, walk=walk, device=args.device)
            matrix["runs"].append(
                {
                    "task": task,
                    "walk": walk,
                    "config": config.to_dict(),
                    "dataset_path": str(dataset),
                    "dataset_sha256": sha256_file(dataset),
                    "feature_path": str(feature),
                    "feature_sha256": sha256_file(feature),
                    "feature_scaler_path": str(scaler),
                    "feature_scaler_sha256": sha256_file(scaler),
                    "run_root": str(run_root),
                }
            )
    if matrix_path.is_file():
        existing = json.loads(matrix_path.read_text(encoding="utf-8"))
        if existing != matrix:
            raise ValueError("existing regression add-on matrix differs")
    else:
        write_json(matrix_path, matrix)

    results = []
    for row in matrix["runs"]:
        task, walk = str(row["task"]), int(row["walk"])
        dataset, feature, scaler, run_root = paths(task, walk)
        if (run_root / "training_complete.json").is_file():
            result = validate_regression_addon_run(dataset, feature, scaler, run_root)
            result["action"] = "validated_existing"
        else:
            run_regression_addon(
                dataset,
                feature,
                scaler,
                run_root,
                RegressionAddonConfig(task=task, walk=walk, device=args.device),
            )
            result = validate_regression_addon_run(dataset, feature, scaler, run_root)
            result["action"] = "trained"
        results.append(result)
    print(json.dumps({"complete": True, "runs": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
