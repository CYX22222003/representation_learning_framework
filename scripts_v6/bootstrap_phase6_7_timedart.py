#!/usr/bin/env python3
"""Freeze and optionally execute both walk-specific TimeDART trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file  # noqa: E402
from data_processing.phase6_7_timedart_data import METHOD, encoder_dataset_path  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_7_timedart import (  # noqa: E402
    Phase67TimeDARTConfig,
    run_timedart_pretraining,
    smoke_test_timedart,
    validate_admission_manifest,
    validate_timedart_pretraining,
)


PHASE = ROOT / "experiments" / "phase6_7"
MANIFEST = PHASE / "manifests" / "timedart_encoder_seed0.json"
ADMISSION = PHASE / "feasibility" / "timedart" / "cuda_admission.json"


def _comparison(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    for entry in normalized["entries"]:  # type: ignore[index]
        entry["config"]["device"] = "<runtime>"
        for field in ("dataset_path", "run_root"):
            entry[field] = entry[field].casefold()
    return normalized


def _advisory_validation(dataset: Path, run_root: Path) -> dict[str, object]:
    try:
        result = validate_timedart_pretraining(dataset, run_root)
        result["action"] = "validated_existing"
        return result
    except Exception as exc:
        return {
            "valid": False,
            "status": "warning",
            "continued": True,
            "action": "reused_existing_with_validation_warning",
            "message": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        entries = []
        for walk in (1, 2):
            dataset = encoder_dataset_path(ROOT, walk)
            run_root = PHASE / "encoder_pretraining" / METHOD / f"walk{walk}" / "seed0"
            entries.append(
                {
                    "walk": walk,
                    "method": METHOD,
                    "dataset_path": str(dataset.resolve()),
                    "dataset_sha256": sha256_file(dataset),
                    "run_root": str(run_root.resolve()),
                    "config": Phase67TimeDARTConfig(walk=walk, device=args.device).to_dict(),
                }
            )
        frozen = {
            "schema_version": "phase6-7-timedart-encoder-matrix-v1",
            "phase": "6.7",
            "method": METHOD,
            "seed": 0,
            "entry_count": 2,
            "entries": entries,
            "cpu_smoke_test": smoke_test_timedart(),
            "default_action_trains_models": False,
            "targets_or_evaluation_loaded": False,
            "principal_epoch": 50,
        }
        if MANIFEST.exists():
            if _comparison(json.loads(MANIFEST.read_text(encoding="utf-8"))) != _comparison(frozen):
                raise ValueError("existing TimeDART encoder manifest differs from freeze")
        else:
            write_json(MANIFEST, frozen)
        results = []
        if args.execute:
            for entry in entries:
                dataset = Path(entry["dataset_path"])
                run_root = Path(entry["run_root"])
                walk = int(entry["walk"])
                validate_admission_manifest(ADMISSION, dataset, walk)
                if (run_root / "training_complete.json").is_file():
                    results.append(_advisory_validation(dataset, run_root))
                    continue
                payload = dict(entry["config"])
                payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
                payload["device"] = args.device
                result = run_timedart_pretraining(
                    dataset, run_root, Phase67TimeDARTConfig(**payload)
                )
                result["action"] = "trained_or_resumed"
                results.append(result)
        print(json.dumps({"valid": True, "manifest": str(MANIFEST.resolve()), "executed": args.execute, "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"TimeDART encoder bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

