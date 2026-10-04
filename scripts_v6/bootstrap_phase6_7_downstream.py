#!/usr/bin/env python3
"""Freeze SaURL/H0 Phase 6.7 probes; train SaURL heads only with --execute."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file  # noqa: E402
from evaluation.phase6_encoder_variant_reporting import validate_h0_reference  # noqa: E402
from features.phase6_7_external_features import (  # noqa: E402
    METHOD,
    OUTPUT_DIM,
    TASKS,
    validate_external_feature_store,
)
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_7_downstream import (  # noqa: E402
    Phase67DownstreamConfig,
    run_external_downstream,
    smoke_test_external_downstream,
    validate_external_downstream,
)


PHASE_ROOT = ROOT / "experiments" / "phase6_7"
MANIFEST = PHASE_ROOT / "manifests" / "saurl_downstream_seed0.json"


def task_dataset(task: str, walk: int) -> Path:
    if task == "classification_h2":
        return (
            ROOT
            / "experiments"
            / "phase5"
            / "data_preparation"
            / f"walk{walk}"
            / "market_1h_seq64_h2.npz"
        )
    if task == "absolute_price_h8":
        return (
            ROOT
            / "experiments"
            / "phase5"
            / "downstream_addons"
            / "shared"
            / "h8"
            / "data"
            / f"walk{walk}"
            / "market_1h_seq64_h8.npz"
        )
    return (
        ROOT
        / "experiments"
        / "phase6"
        / "volatility_prediction"
        / "data_preparation"
        / f"walk{walk}"
        / "volatility_1h_seq64_h8.npz"
    )


def h0_paths(task: str, walk: int) -> tuple[Path, Path]:
    if task == "classification_h2":
        return (
            ROOT / "experiments" / "phase5" / "features" / f"walk{walk}" / "five_branch_epoch50.npz",
            ROOT / "experiments" / "phase5" / "downstream" / f"walk{walk}" / "classification" / "seed0",
        )
    if task == "absolute_price_h8":
        base = ROOT / "experiments" / "phase5" / "downstream_addons"
        return (
            base / "shared" / "h8" / "features" / f"walk{walk}" / "five_branch_epoch50.npz",
            base / "tasks" / "absolute_price_h8" / f"walk{walk}" / "seed0",
        )
    base = ROOT / "experiments" / "phase6" / "volatility_prediction"
    return (
        base / "features" / f"walk{walk}" / "five_branch_epoch50_h8.npz",
        base / "runs" / "framework_h0" / f"walk{walk}" / "seed0",
    )


def _comparison_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    for entry in normalized["entries"]:
        if entry["config"] is not None:
            entry["config"]["device"] = "<runtime>"
        for field in ("dataset_path", "feature_path", "run_root"):
            entry[field] = entry[field].casefold()
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        entries = []
        for task in TASKS:
            for walk in (1, 2):
                dataset = task_dataset(task, walk)
                external_feature = (
                    PHASE_ROOT
                    / "features"
                    / METHOD
                    / f"walk{walk}"
                    / "representations.npz"
                )
                feature_validation = validate_external_feature_store(
                    external_feature, replay=False
                )
                external_run = (
                    PHASE_ROOT
                    / "downstream"
                    / task
                    / METHOD
                    / f"walk{walk}"
                    / "seed0"
                )
                entries.append(
                    {
                        "task": task,
                        "walk": walk,
                        "method": METHOD,
                        "execution_mode": "phase6_7_training_required",
                        "dataset_path": str(dataset.resolve()),
                        "dataset_sha256": sha256_file(dataset),
                        "feature_path": str(external_feature.resolve()),
                        "feature_sha256": sha256_file(external_feature),
                        "feature_validation": feature_validation,
                        "run_root": str(external_run.resolve()),
                        "config": Phase67DownstreamConfig(
                            task, METHOD, walk, input_dim=OUTPUT_DIM, device=args.device
                        ).to_dict(),
                    }
                )
                h0_feature, h0_run = h0_paths(task, walk)
                h0_validation = validate_h0_reference(ROOT, task, walk)
                entries.append(
                    {
                        "task": task,
                        "walk": walk,
                        "method": "H0",
                        "execution_mode": "immutable_replayed_reference",
                        "dataset_path": str(dataset.resolve()),
                        "dataset_sha256": sha256_file(dataset),
                        "feature_path": str(h0_feature.resolve()),
                        "feature_sha256": sha256_file(h0_feature),
                        "feature_validation": h0_validation,
                        "run_root": str(h0_run.resolve()),
                        "config": None,
                    }
                )
        frozen = {
            "schema_version": "phase6-7-saurl-downstream-matrix-v1",
            "phase": "6.7",
            "seed": 0,
            "entry_count": len(entries),
            "new_trajectory_count": 6,
            "immutable_reference_count": 6,
            "implemented_scope": "SaURL-Frozen plus immutable H0 references",
            "full_core_manifest_pending_lwa": True,
            "saurl_execution_permitted_before_lwa": True,
            "staged_execution_decision_date": "2026-10-04",
            "saurl_results_may_influence_lwa_implementation": False,
            "entries": entries,
            "cpu_smoke_test": smoke_test_external_downstream(),
            "default_action_trains_models": False,
            "evaluation_used_to_screen_matrix": False,
        }
        if len(entries) != 12:
            raise ValueError("SaURL/H0 downstream manifest must contain 12 entries")
        if MANIFEST.exists():
            existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
            if _comparison_payload(existing) != _comparison_payload(frozen):
                raise ValueError("existing Phase 6.7 downstream manifest differs from freeze")
        else:
            write_json(MANIFEST, frozen)
        results = []
        if args.execute:
            for entry in entries:
                if entry["method"] == "H0":
                    continue
                payload = dict(entry["config"])
                payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
                payload["device"] = args.device
                config = Phase67DownstreamConfig(**payload)
                dataset = Path(entry["dataset_path"])
                feature = Path(entry["feature_path"])
                run_root = Path(entry["run_root"])
                if run_root.is_dir():
                    result = validate_external_downstream(dataset, feature, run_root)
                    result["action"] = "validated_existing"
                else:
                    run_external_downstream(dataset, feature, run_root, config)
                    result = validate_external_downstream(dataset, feature, run_root)
                    result["action"] = "trained"
                results.append(result)
        print(
            json.dumps(
                {
                    "valid": True,
                    "manifest": str(MANIFEST.resolve()),
                    "entry_count": len(entries),
                    "executed": args.execute,
                    "runs": results,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"Phase 6.7 downstream bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
