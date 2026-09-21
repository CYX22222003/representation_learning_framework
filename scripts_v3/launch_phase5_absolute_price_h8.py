#!/usr/bin/env python3
"""Freeze and execute the two-walk Phase 5 eight-hour absolute-price probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.phase5_absolute_price import (
    AbsolutePriceConfig,
    run_absolute_price,
    validate_absolute_price_run,
)
from training.phase5_encoder import sha256_file, write_json


def paths(walk: int) -> tuple[Path, Path, Path, Path]:
    return (
        Path(
            f"experiments/phase5/data_sensitivities/raw_delta_h8/walk{walk}/market_1h_seq64_h8.npz"
        ),
        Path(
            f"experiments/phase5/features_sensitivities/raw_delta_h8/walk{walk}/five_branch_epoch50.npz"
        ),
        Path(
            f"experiments/phase5/regression_sensitivities/raw_delta_h8/walk{walk}/feature_standardizer.npz"
        ),
        Path(f"experiments/phase5/absolute_price_h8/walk{walk}/seed0"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    matrix_path = Path("experiments/phase5/manifests/absolute_price_h8_seed0.json")
    matrix = {
        "phase": 5,
        "purpose": "frozen_exploratory_absolute_price_h8_matrix",
        "frozen_before_training": True,
        "runs": [],
    }
    for walk in (1, 2):
        dataset, feature, scaler, run_root = paths(walk)
        config = AbsolutePriceConfig(walk=walk, device=args.device)
        matrix["runs"].append(
            {
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
        if json.loads(matrix_path.read_text(encoding="utf-8")) != matrix:
            raise ValueError("existing absolute-price matrix differs")
    else:
        write_json(matrix_path, matrix)
    results = []
    for walk in (1, 2):
        dataset, feature, scaler, run_root = paths(walk)
        if (run_root / "training_complete.json").is_file():
            result = validate_absolute_price_run(dataset, feature, scaler, run_root)
            result["action"] = "validated_existing"
        else:
            run_absolute_price(
                dataset,
                feature,
                scaler,
                run_root,
                AbsolutePriceConfig(walk=walk, device=args.device),
            )
            result = validate_absolute_price_run(dataset, feature, scaler, run_root)
            result["action"] = "trained"
        results.append(result)
    print(json.dumps({"complete": True, "runs": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
