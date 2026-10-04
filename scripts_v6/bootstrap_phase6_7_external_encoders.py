#!/usr/bin/env python3
"""Freeze SaURL Phase 6.7 encoder runs; train only with --execute."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file, validate_phase5_bundle_files  # noqa: E402
from baselines.saurl_ts import SaURLConfig  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_7_external_encoders import (  # noqa: E402
    METHODS,
    Phase67ExternalEncoderConfig,
    run_external_encoder,
    smoke_test_external_encoders,
    validate_external_encoder,
)


PHASE_ROOT = ROOT / "experiments" / "phase6_7"
MANIFEST = PHASE_ROOT / "manifests" / "saurl_pretraining_seed0.json"
FEASIBILITY = PHASE_ROOT / "feasibility" / "saurl_ts" / "feasibility_manifest.json"


def _comparison_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    for entry in normalized["entries"]:
        entry["config"]["device"] = "<runtime>"
        entry["dataset_path"] = entry["dataset_path"].casefold()
        entry["run_root"] = entry["run_root"].casefold()
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        if not FEASIBILITY.is_file():
            raise FileNotFoundError(
                "run scripts_v6/audit_phase6_7_sources.py before freezing pretraining"
            )
        feasibility = json.loads(FEASIBILITY.read_text(encoding="utf-8"))
        if args.execute and feasibility.get("admitted_for_training") is not True:
            raise ValueError("SaURL feasibility manifest has not admitted training")
        entries = []
        for walk in (1, 2):
            dataset = (
                ROOT
                / "experiments"
                / "phase5"
                / "data_preparation"
                / f"walk{walk}"
                / "market_1h_seq64_h2.npz"
            )
            validation = validate_phase5_bundle_files(dataset)
            for method in METHODS:
                run_root = (
                    PHASE_ROOT
                    / "encoder_pretraining"
                    / method
                    / f"walk{walk}"
                    / "seed0"
                )
                entries.append(
                    {
                        "method": method,
                        "walk": walk,
                        "dataset_path": str(dataset.resolve()),
                        "dataset_sha256": sha256_file(dataset),
                        "dataset_manifest_sha256": sha256_file(
                            Path(f"{dataset}.manifest.json")
                        ),
                        "encoder_train_identity_hash": validation[
                            "identity_hashes"
                        ]["encoder_train"],
                        "run_root": str(run_root.resolve()),
                        "config": Phase67ExternalEncoderConfig(
                            method, walk, device=args.device
                        ).to_dict(),
                        "model_config": SaURLConfig().to_dict(),
                    }
                )
        frozen = {
            "schema_version": "phase6-7-external-pretraining-matrix-v1",
            "phase": "6.7",
            "entries": entries,
            "entry_count": len(entries),
            "implemented_methods": list(METHODS),
            "scope": "SaURL-only implementation stage; registry is extensible for LWA",
            "default_action_trains_models": False,
            "evaluation_values_loaded": False,
            "cpu_smoke_test": smoke_test_external_encoders(),
            "feasibility_manifest": str(FEASIBILITY.resolve()),
            "feasibility_manifest_sha256": sha256_file(FEASIBILITY),
            "admitted_for_training": feasibility.get("admitted_for_training") is True,
        }
        if MANIFEST.exists():
            existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
            if _comparison_payload(existing) != _comparison_payload(frozen):
                raise ValueError("existing Phase 6.7 pretraining manifest differs from freeze")
        else:
            write_json(MANIFEST, frozen)
        results = []
        if args.execute:
            for entry in entries:
                payload = dict(entry["config"])
                payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
                payload["device"] = args.device
                config = Phase67ExternalEncoderConfig(**payload)
                dataset = Path(entry["dataset_path"])
                run_root = Path(entry["run_root"])
                if (run_root / "training_complete.json").is_file():
                    result = validate_external_encoder(dataset, run_root)
                    result["action"] = "validated_existing"
                else:
                    result = run_external_encoder(dataset, run_root, config)
                    result["action"] = "trained_or_resumed"
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
        print(f"Phase 6.7 external encoder bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
