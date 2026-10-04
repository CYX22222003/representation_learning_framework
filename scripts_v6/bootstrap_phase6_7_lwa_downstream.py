#!/usr/bin/env python3
"""Freeze and optionally train the six LWA downstream probes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file  # noqa: E402
from features.phase6_7_lwa_features import METHOD, OUTPUT_DIM, TASKS, validate_lwa_feature_store  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_7_downstream import (  # noqa: E402
    Phase67DownstreamConfig,
    run_external_downstream,
    smoke_test_external_downstream,
    validate_external_downstream,
)


PHASE = ROOT / "experiments" / "phase6_7"
MANIFEST = PHASE / "manifests" / "lwa_downstream_seed0.json"


def task_dataset(task: str, walk: int) -> Path:
    if task == "classification_h2":
        return ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
    if task == "absolute_price_h8":
        return ROOT / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8" / "data" / f"walk{walk}" / "market_1h_seq64_h8.npz"
    return ROOT / "experiments" / "phase6" / "volatility_prediction" / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz"


def _comparison(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    for entry in normalized["entries"]:  # type: ignore[index]
        if entry["config"] is not None:
            entry["config"]["device"] = "<runtime>"
        for field in ("dataset_path", "feature_path", "run_root"):
            entry[field] = entry[field].casefold()
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        entries = []
        for task in TASKS:
            for walk in (1, 2):
                dataset = task_dataset(task, walk)
                feature = PHASE / "features" / METHOD / f"walk{walk}" / "representations.npz"
                run = PHASE / "downstream" / task / METHOD / f"walk{walk}" / "seed0"
                entries.append(
                    {
                        "task": task,
                        "walk": walk,
                        "method": METHOD,
                        "execution_mode": "phase6_7_training_required",
                        "dataset_path": str(dataset.resolve()),
                        "dataset_sha256": sha256_file(dataset),
                        "feature_path": str(feature.resolve()),
                        "feature_sha256": sha256_file(feature),
                        "feature_validation": validate_lwa_feature_store(feature, replay=False),
                        "run_root": str(run.resolve()),
                        "config": Phase67DownstreamConfig(task, METHOD, walk, input_dim=OUTPUT_DIM, device=args.device).to_dict(),
                    }
                )
        comparison_references = [
            {
                "task": task,
                "walk": walk,
                "method": "H0",
                "execution_mode": "immutable_reference_joined_during_core_reporting",
                "runtime_dependency": False,
            }
            for task in TASKS
            for walk in (1, 2)
        ]
        frozen = {
            "schema_version": "phase6-7-lwa-downstream-matrix-v2",
            "phase": "6.7",
            "seed": 0,
            "entry_count": len(entries),
            "new_trajectory_count": 6,
            "comparison_reference_count": len(comparison_references),
            "comparison_references": comparison_references,
            "implemented_scope": "LWA-Frozen downstream execution only",
            "h0_artifacts_required_for_execution": False,
            "comparison_policy": "Join already completed immutable H0 and SaURL results during core reporting; do not rerun or copy H0 artifacts to train LWA probes.",
            "full_core_manifest_pending": True,
            "entries": entries,
            "cpu_smoke_test": smoke_test_external_downstream(METHOD),
            "default_action_trains_models": False,
            "evaluation_used_to_screen_matrix": False,
        }
        if len(entries) != 6:
            raise ValueError("LWA downstream manifest must contain six entries")
        if MANIFEST.exists():
            if _comparison(json.loads(MANIFEST.read_text(encoding="utf-8"))) != _comparison(frozen):
                raise ValueError("existing LWA downstream manifest differs from freeze")
        else:
            write_json(MANIFEST, frozen)
        results = []
        if args.execute:
            for entry in entries:
                payload = dict(entry["config"])
                payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
                payload["device"] = args.device
                config = Phase67DownstreamConfig(**payload)
                dataset, feature, run = Path(entry["dataset_path"]), Path(entry["feature_path"]), Path(entry["run_root"])
                if run.is_dir():
                    result = validate_external_downstream(dataset, feature, run)
                    result["action"] = "validated_existing"
                else:
                    run_external_downstream(dataset, feature, run, config)
                    result = validate_external_downstream(dataset, feature, run)
                    result["action"] = "trained"
                results.append(result)
        print(json.dumps({"valid": True, "manifest": str(MANIFEST.resolve()), "executed": args.execute, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.7 LWA downstream bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
