#!/usr/bin/env python3
"""Freeze the ten-run Phase 6.5C matrix; train only with --execute."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.ta_mlp_baseline.phase6_5 import (  # noqa: E402
    Phase65TAConfig,
    run_phase65_ta_training,
    smoke_test_phase65_ta_models,
    strict_matrix_entries,
    validate_phase65_ta_run,
    validate_phase65_ta_store,
)
from data_processing.phase5_walks import sha256_file  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402


PHASE_ROOT = ROOT / "experiments" / "phase6_5" / "ta_mlp"
MANIFEST = ROOT / "experiments" / "phase6_5" / "manifests" / "ta_mlp_classification_seed0.json"


def store_path(walk: int) -> Path:
    return PHASE_ROOT / "data_preparation" / f"walk{walk}" / "classification_h2_ta36.npz"


def run_path(walk: int, model: str, protocol: str) -> Path:
    return PHASE_ROOT / "downstream" / model / f"walk{walk}" / protocol.lower() / "seed0"


def comparable(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    for entry in normalized["entries"]:
        entry["config"]["device"] = "<runtime>"
        entry["store_path"] = entry["store_path"].casefold()
        entry["run_root"] = entry["run_root"].casefold()
    return normalized


def freeze(device: str) -> dict[str, object]:
    entries = []
    stores = {}
    for walk in (1, 2):
        store = store_path(walk)
        stores[f"walk{walk}"] = {
            "path": str(store.resolve()),
            "sha256": sha256_file(store),
            "validation": validate_phase65_ta_store(store),
        }
        for model, protocol in strict_matrix_entries():
            entries.append(
                {
                    "walk": walk,
                    "model": model,
                    "protocol": protocol,
                    "store_path": str(store.resolve()),
                    "store_sha256": sha256_file(store),
                    "run_root": str(run_path(walk, model, protocol).resolve()),
                    "config": Phase65TAConfig(model, protocol, walk, device=device).to_dict(),
                }
            )
    payload = {
        "schema_version": "phase6-5c-ta-matrix-v1",
        "phase": "6.5C",
        "entry_count": len(entries),
        "p2_entry_count": 8,
        "p1u_entry_count": 2,
        "entries": entries,
        "stores": stores,
        "cpu_smoke_test": smoke_test_phase65_ta_models(),
        "default_action_trains_models": False,
        "evaluation_used_to_screen_matrix": False,
        "principal_epoch": 50,
    }
    if len(entries) != 10:
        raise ValueError("Phase 6.5C matrix must contain ten trajectories")
    if MANIFEST.exists():
        if comparable(json.loads(MANIFEST.read_text(encoding="utf-8"))) != comparable(payload):
            raise ValueError("existing Phase 6.5C manifest differs from freeze")
    else:
        write_json(MANIFEST, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        manifest = freeze(args.device)
        runs = []
        if args.execute:
            for entry in manifest["entries"]:
                store = Path(entry["store_path"])
                run = Path(entry["run_root"])
                payload = dict(entry["config"])
                payload.pop("batch_size")
                payload.pop("learning_rate")
                payload.pop("weight_decay")
                payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
                payload["device"] = args.device
                config = Phase65TAConfig(**payload)
                if run.is_dir():
                    result = validate_phase65_ta_run(store, run)
                    result["action"] = "validated_existing"
                else:
                    run_phase65_ta_training(store, run, config)
                    result = validate_phase65_ta_run(store, run)
                    result["action"] = "trained"
                runs.append(result)
        print(json.dumps({"valid": True, "manifest": str(MANIFEST.resolve()), "entry_count": len(manifest["entries"]), "executed": args.execute, "runs": runs}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5C TA-MLP bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
