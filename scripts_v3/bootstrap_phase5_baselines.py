#!/usr/bin/env python3
"""Freeze the Phase 5 raw-baseline matrix and optionally execute it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.phase5_baselines import (
    BASELINES,
    TASKS,
    Phase5BaselineConfig,
    run_baseline_training,
    smoke_test_baselines,
    validate_baseline_run,
)
from training.phase5_encoder import sha256_file, write_json


MATRIX_PATH = Path("experiments/phase5/baselines/manifests/baseline_matrix_seed0.json")


def dataset_path(walk: int, task: str) -> Path:
    if task == "absolute_price_h8":
        return Path(
            f"experiments/phase5/downstream_addons/shared/h8/data/walk{walk}/"
            "market_1h_seq64_h8.npz"
        )
    return Path(f"experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz")


def run_root(walk: int, task: str, baseline: str) -> Path:
    return Path(
        f"experiments/phase5/baselines/tasks/{task}/{baseline}/walk{walk}/seed0"
    )


def build_matrix(device: str) -> dict[str, object]:
    runs = []
    for task in TASKS:
        for baseline in BASELINES:
            for walk in (1, 2):
                source = dataset_path(walk, task)
                config = Phase5BaselineConfig(
                    baseline=baseline, task=task, walk=walk, device=device
                )
                runs.append(
                    {
                        "baseline": baseline,
                        "task": task,
                        "walk": walk,
                        "config": config.to_dict(),
                        "dataset_path": str(source),
                        "dataset_sha256": sha256_file(source),
                        "run_root": str(run_root(walk, task, baseline)),
                    }
                )
    return {
        "phase": 5,
        "purpose": "frozen_matched_raw_sequence_baseline_matrix",
        "frozen_before_training": True,
        "selection_rule": "epoch 50 predeclared; no evaluation-driven selection",
        "run_count": 12,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--execute", action="store_true", help="Train missing runs after freezing the matrix."
    )
    args = parser.parse_args()
    smoke = smoke_test_baselines()
    matrix = build_matrix(args.device)
    if MATRIX_PATH.is_file():
        if json.loads(MATRIX_PATH.read_text(encoding="utf-8")) != matrix:
            raise ValueError("existing Phase 5 baseline matrix differs from the frozen contract")
    else:
        write_json(MATRIX_PATH, matrix)
    results = []
    if args.execute:
        for row in matrix["runs"]:
            source = Path(row["dataset_path"])
            root = Path(row["run_root"])
            config_payload = dict(row["config"])
            config_payload["snapshot_epochs"] = tuple(config_payload["snapshot_epochs"])
            config = Phase5BaselineConfig(**config_payload)
            if (root / "training_complete.json").is_file():
                result = validate_baseline_run(source, root)
                result["action"] = "validated_existing"
            else:
                run_baseline_training(source, root, config)
                result = validate_baseline_run(source, root)
                result["action"] = "trained"
            results.append(result)
    print(
        json.dumps(
            {
                "matrix_frozen": True,
                "execution_requested": args.execute,
                "matrix_path": str(MATRIX_PATH),
                "smoke_test": smoke,
                "runs": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
