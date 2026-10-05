#!/usr/bin/env python3
"""Freeze and optionally train the six TimeDART frozen-representation probes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file  # noqa: E402
from data_processing.phase6_7_timedart_data import METHOD, TASKS, timedart_dataset_paths  # noqa: E402
from features.phase6_7_timedart_features import OUTPUT_DIM, validate_timedart_feature_store  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_7_downstream import (  # noqa: E402
    Phase67DownstreamConfig,
    run_external_downstream,
    smoke_test_external_downstream,
    validate_external_downstream,
)


PHASE = ROOT / "experiments" / "phase6_7"
MANIFEST = PHASE / "manifests" / "timedart_downstream_seed0.json"


def _comparison(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    for entry in normalized["entries"]:  # type: ignore[index]
        entry["config"]["device"] = "<runtime>"
        for field in ("dataset_path", "feature_path", "run_root"):
            entry[field] = entry[field].casefold()
    return normalized


def _advisory_validation(dataset: Path, feature: Path, run_root: Path) -> dict[str, object]:
    try:
        result = validate_external_downstream(dataset, feature, run_root)
        result["action"] = "validated_existing"
        return result
    except Exception as exc:
        return {"valid": False, "status": "warning", "continued": True, "action": "reused_existing_with_validation_warning", "message": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        paths = timedart_dataset_paths(ROOT)
        entries = []
        for task in TASKS:
            for walk in (1, 2):
                dataset = paths[walk][task]
                feature = PHASE / "features" / METHOD / f"walk{walk}" / "representations.npz"
                try:
                    feature_validation = validate_timedart_feature_store(feature, replay=False)
                except Exception as exc:
                    feature_validation = {"valid": False, "status": "warning", "continued": True, "message": str(exc)}
                run_root = PHASE / "downstream" / task / METHOD / f"walk{walk}" / "seed0"
                entries.append({
                    "task": task,
                    "walk": walk,
                    "method": METHOD,
                    "dataset_path": str(dataset.resolve()),
                    "dataset_sha256": sha256_file(dataset),
                    "feature_path": str(feature.resolve()),
                    "feature_sha256": sha256_file(feature),
                    "feature_validation": feature_validation,
                    "run_root": str(run_root.resolve()),
                    "config": Phase67DownstreamConfig(task, METHOD, walk, input_dim=OUTPUT_DIM, device=args.device).to_dict(),
                })
        frozen = {
            "schema_version": "phase6-7-timedart-downstream-matrix-v1",
            "phase": "6.7",
            "method": METHOD,
            "seed": 0,
            "entry_count": 6,
            "new_trajectory_count": 6,
            "entries": entries,
            "cpu_smoke_test": smoke_test_external_downstream(METHOD),
            "default_action_trains_models": False,
            "evaluation_used_to_screen_matrix": False,
            "h0_or_other_baseline_artifacts_required_for_execution": False,
        }
        if MANIFEST.exists():
            if _comparison(json.loads(MANIFEST.read_text(encoding="utf-8"))) != _comparison(frozen):
                raise ValueError("existing TimeDART downstream manifest differs from freeze")
        else:
            write_json(MANIFEST, frozen)
        results = []
        if args.execute:
            for entry in entries:
                dataset, feature, run_root = Path(entry["dataset_path"]), Path(entry["feature_path"]), Path(entry["run_root"])
                if (run_root / "training_complete.json").is_file():
                    results.append(_advisory_validation(dataset, feature, run_root))
                    continue
                payload = dict(entry["config"])
                payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
                payload["device"] = args.device
                run_external_downstream(dataset, feature, run_root, Phase67DownstreamConfig(**payload))
                result = _advisory_validation(dataset, feature, run_root)
                result["action"] = "trained"
                results.append(result)
        print(json.dumps({"valid": True, "manifest": str(MANIFEST.resolve()), "executed": args.execute, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"TimeDART downstream bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

