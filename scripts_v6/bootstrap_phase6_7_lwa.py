#!/usr/bin/env python3
"""Prepare/freeze LWA views and runs; train only with explicit --execute."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.lwa import LWAConfig  # noqa: E402
from data_processing.phase5_walks import sha256_file, validate_phase5_bundle_files  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_7_lwa import (  # noqa: E402
    METHOD,
    Phase67LWATrainingConfig,
    build_lwa_view_cache,
    run_lwa_pretraining,
    smoke_test_lwa_pretraining,
    validate_lwa_pretraining,
    validate_lwa_view_cache,
)


PHASE_ROOT = ROOT / "experiments" / "phase6_7"
MANIFEST = PHASE_ROOT / "manifests" / "lwa_pretraining_seed0.json"
FEASIBILITY = PHASE_ROOT / "feasibility" / "lwa" / "feasibility_manifest.json"


def dataset_path(walk: int) -> Path:
    return ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"


def cache_path(walk: int) -> Path:
    return PHASE_ROOT / "view_cache" / METHOD / f"walk{walk}"


def _comparison(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    for entry in normalized["entries"]:  # type: ignore[index]
        entry["config"]["device"] = "<runtime>"
        for field in ("dataset_path", "cache_root", "run_root"):
            entry[field] = entry[field].casefold()
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--cache-chunk-size", type=int, default=256)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        entries = []
        for walk in (1, 2):
            dataset = dataset_path(walk)
            validation = validate_phase5_bundle_files(dataset, replay_source=False)
            cache = cache_path(walk)
            cache_validation = None
            if args.prepare_cache:
                cache_validation = build_lwa_view_cache(
                    dataset, cache, walk=walk, chunk_size=args.cache_chunk_size
                )
            elif cache.is_dir():
                cache_validation = validate_lwa_view_cache(dataset, cache, walk=walk)
            run_root = PHASE_ROOT / "encoder_pretraining" / METHOD / f"walk{walk}" / "seed0"
            entries.append(
                {
                    "method": METHOD,
                    "walk": walk,
                    "dataset_path": str(dataset.resolve()),
                    "dataset_sha256": sha256_file(dataset),
                    "dataset_manifest_sha256": sha256_file(Path(f"{dataset}.manifest.json")),
                    "encoder_train_identity_hash": validation["identity_hashes"]["encoder_train"],
                    "cache_root": str(cache.resolve()),
                    "cache_validation": cache_validation,
                    "run_root": str(run_root.resolve()),
                    "config": Phase67LWATrainingConfig(walk=walk, device=args.device).to_dict(),
                    "model_config": LWAConfig().to_dict(),
                }
            )
        frozen = {
            "schema_version": "phase6-7-lwa-pretraining-matrix-v1",
            "phase": "6.7",
            "entries": entries,
            "entry_count": 2,
            "implemented_methods": [METHOD],
            "two_stage_contract": {"joint_epochs": 50, "mapper_epochs": 50},
            "default_action_trains_models": False,
            "evaluation_values_loaded": False,
            "cpu_smoke_test": smoke_test_lwa_pretraining(),
        }
        caches_ready = all(entry["cache_validation"] is not None for entry in entries)
        if caches_ready:
            if MANIFEST.exists():
                existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
                if _comparison(existing) != _comparison(frozen):
                    raise ValueError("existing LWA pretraining manifest differs from freeze")
            else:
                write_json(MANIFEST, frozen)
        results = []
        if args.execute:
            if not caches_ready:
                raise FileNotFoundError("prepare both LWA view caches before training")
            if not FEASIBILITY.is_file():
                raise FileNotFoundError("run and admit scripts_v6/audit_phase6_7_lwa.py first")
            feasibility = json.loads(FEASIBILITY.read_text(encoding="utf-8"))
            if feasibility.get("admitted_for_training") is not True:
                raise ValueError("LWA feasibility manifest has not admitted training")
            for entry in entries:
                payload = dict(entry["config"])
                payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
                payload["device"] = args.device
                config = Phase67LWATrainingConfig(**payload)
                dataset, cache, run_root = Path(entry["dataset_path"]), Path(entry["cache_root"]), Path(entry["run_root"])
                if (run_root / "training_complete.json").is_file():
                    result = validate_lwa_pretraining(dataset, cache, run_root)
                    result["action"] = "validated_existing"
                else:
                    result = run_lwa_pretraining(
                        dataset, cache, run_root, config, FEASIBILITY
                    )
                    result["action"] = "trained_or_resumed"
                results.append(result)
        print(json.dumps({"valid": True, "manifest": str(MANIFEST.resolve()) if caches_ready else None, "caches_ready": caches_ready, "executed": args.execute, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.7 LWA bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
