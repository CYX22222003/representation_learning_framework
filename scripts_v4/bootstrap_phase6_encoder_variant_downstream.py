#!/usr/bin/env python3
"""Freeze the 66-entry Phase 6 downstream matrix; train only with --execute."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file  # noqa: E402
from features.phase6_encoder_variant_features import (  # noqa: E402
    CONFIG_BRANCHES,
    TASKS,
    validate_variant_feature_store,
)
from training.phase5_absolute_price import validate_absolute_price_run  # noqa: E402
from training.phase5_downstream import validate_downstream_run  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_encoder_variant_downstream import (  # noqa: E402
    Phase6VariantDownstreamConfig,
    run_variant_downstream,
    smoke_test_variant_downstream,
    validate_variant_downstream,
)
from training.phase6_volatility import validate_volatility_run  # noqa: E402


CONFIGURATIONS = tuple(CONFIG_BRANCHES)
PHASE6_ROOT = ROOT / "experiments" / "phase6" / "encoder_variants"
MANIFEST = PHASE6_ROOT / "manifests" / "downstream_seed0.json"


def task_paths(task: str, walk: int) -> tuple[Path, Path]:
    if task == "classification_h2":
        dataset = ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
    elif task == "absolute_price_h8":
        dataset = ROOT / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8" / "data" / f"walk{walk}" / "market_1h_seq64_h8.npz"
    else:
        dataset = ROOT / "experiments" / "phase6" / "volatility_prediction" / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz"
    feature = PHASE6_ROOT / "features" / f"walk{walk}" / f"{task}.npz"
    return dataset, feature


def h0_reference(task: str, walk: int) -> tuple[Path, dict[str, object]]:
    if task == "classification_h2":
        dataset = ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
        feature = ROOT / "experiments" / "phase5" / "features" / f"walk{walk}" / "five_branch_epoch50.npz"
        scaler = ROOT / "experiments" / "phase5" / "downstream" / f"walk{walk}" / "feature_standardizer.npz"
        run = ROOT / "experiments" / "phase5" / "downstream" / f"walk{walk}" / "classification" / "seed0"
        return run, validate_downstream_run(dataset, feature, scaler, run)
    if task == "absolute_price_h8":
        base = ROOT / "experiments" / "phase5" / "downstream_addons"
        dataset = base / "shared" / "h8" / "data" / f"walk{walk}" / "market_1h_seq64_h8.npz"
        feature = base / "shared" / "h8" / "features" / f"walk{walk}" / "five_branch_epoch50.npz"
        scaler = base / "shared" / "h8" / "feature_scalers" / f"walk{walk}" / "feature_standardizer.npz"
        run = base / "tasks" / "absolute_price_h8" / f"walk{walk}" / "seed0"
        return run, validate_absolute_price_run(dataset, feature, scaler, run)
    base = ROOT / "experiments" / "phase6" / "volatility_prediction"
    dataset = base / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz"
    feature = base / "features" / f"walk{walk}" / "five_branch_epoch50_h8.npz"
    run = base / "runs" / "framework_h0" / f"walk{walk}" / "seed0"
    return run, validate_volatility_run(dataset, feature, run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        entries = []
        for task in TASKS:
            for walk in (1, 2):
                dataset, feature = task_paths(task, walk)
                feature_validation = validate_variant_feature_store(feature)
                for configuration in CONFIGURATIONS:
                    if configuration == "H0":
                        run_root, validation = h0_reference(task, walk)
                        mode = "immutable_replayed_reference"
                        config = None
                    else:
                        run_root = PHASE6_ROOT / "downstream" / task / configuration.lower() / f"walk{walk}" / "seed0"
                        mode = "phase6_training_required"
                        config = Phase6VariantDownstreamConfig(task, configuration, walk, device=args.device).to_dict()
                    entries.append({
                        "task": task, "walk": walk, "configuration": configuration,
                        "execution_mode": mode, "run_root": str(run_root.resolve()),
                        "dataset_path": str(dataset.resolve()), "dataset_sha256": sha256_file(dataset),
                        "feature_path": str(feature.resolve()), "feature_sha256": sha256_file(feature),
                        "feature_validation": feature_validation, "config": config,
                    })
        frozen = {
            "schema_version": "phase6-encoder-variant-downstream-matrix-v1",
            "phase": 6, "seed": 0, "entry_count": len(entries),
            "reference_entry_count": 6, "new_trajectory_count": 60,
            "entries": entries, "cpu_smoke_test": smoke_test_variant_downstream(),
            "default_action_trains_models": False,
            "evaluation_used_to_screen_matrix": False,
        }
        if len(entries) != 66:
            raise ValueError("Phase 6 downstream matrix must contain 66 entries")
        if MANIFEST.exists():
            existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
            for payload in (existing, frozen):
                for entry in payload["entries"]:
                    if entry["config"] is not None:
                        entry["config"]["device"] = "<runtime>"
            if existing != frozen:
                raise ValueError("existing Phase 6 downstream manifest differs from freeze")
        else:
            write_json(MANIFEST, frozen)
        results = []
        if args.execute:
            for entry in entries:
                if entry["configuration"] == "H0":
                    continue
                config_payload = dict(entry["config"])
                config_payload["snapshot_epochs"] = tuple(config_payload["snapshot_epochs"])
                config_payload["device"] = args.device
                config = Phase6VariantDownstreamConfig(**config_payload)
                run_root = Path(entry["run_root"])
                dataset = Path(entry["dataset_path"])
                feature = Path(entry["feature_path"])
                if run_root.is_dir():
                    result = validate_variant_downstream(dataset, feature, run_root)
                    result["action"] = "validated_existing"
                else:
                    run_variant_downstream(dataset, feature, run_root, config)
                    result = validate_variant_downstream(dataset, feature, run_root)
                    result["action"] = "trained"
                results.append(result)
        print(json.dumps({"valid": True, "manifest": str(MANIFEST.resolve()), "entry_count": 66, "executed": args.execute, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 downstream bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
