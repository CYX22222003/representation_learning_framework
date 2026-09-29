#!/usr/bin/env python3
"""Freeze the 24-run Phase 6.5A downstream matrix; train only with --execute."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file  # noqa: E402
from features.phase6_5_lstm_capacity_features import (  # noqa: E402
    CONFIG_BRANCHES,
    TASKS,
    validate_lstm_capacity_feature_store,
)
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_5_lstm_capacity_downstream import (  # noqa: E402
    Phase65LSTMCapacityDownstreamConfig,
    run_lstm_capacity_downstream,
    smoke_test_lstm_capacity_downstream,
    validate_lstm_capacity_downstream,
)


CONFIGURATIONS = tuple(name for name in CONFIG_BRANCHES if name != "H0")
ROOT_OUT = ROOT / "experiments" / "phase6_5" / "lstm_capacity"
MANIFEST = ROOT / "experiments" / "phase6_5" / "manifests" / "lstm_capacity_downstream_seed0.json"


def _task_paths(task: str, walk: int) -> tuple[Path, Path]:
    if task == "classification_h2":
        dataset = ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
    elif task == "absolute_price_h8":
        dataset = ROOT / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8" / "data" / f"walk{walk}" / "market_1h_seq64_h8.npz"
    else:
        dataset = ROOT / "experiments" / "phase6" / "volatility_prediction" / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz"
    feature = ROOT_OUT / "features" / f"walk{walk}" / f"{task}.npz"
    return dataset, feature


def _manifest_for_comparison(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    for entry in normalized["entries"]:
        entry["config"]["device"] = "<runtime>"
        for field in ("run_root", "dataset_path", "feature_path"):
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
                dataset, feature = _task_paths(task, walk)
                feature_validation = validate_lstm_capacity_feature_store(feature)
                for configuration in CONFIGURATIONS:
                    run_root = ROOT_OUT / "downstream" / task / configuration.lower() / f"walk{walk}" / "seed0"
                    config = Phase65LSTMCapacityDownstreamConfig(
                        task, configuration, walk, device=args.device
                    ).to_dict()
                    entries.append(
                        {
                            "task": task,
                            "walk": walk,
                            "configuration": configuration,
                            "run_root": str(run_root.resolve()),
                            "dataset_path": str(dataset.resolve()),
                            "dataset_sha256": sha256_file(dataset),
                            "feature_path": str(feature.resolve()),
                            "feature_sha256": sha256_file(feature),
                            "feature_validation": feature_validation,
                            "config": config,
                        }
                    )
        frozen = {
            "schema_version": "phase6-5a-lstm-capacity-downstream-v1",
            "phase": "6.5A",
            "seed": 0,
            "entry_count": len(entries),
            "new_trajectory_count": 24,
            "entries": entries,
            "cpu_smoke_test": smoke_test_lstm_capacity_downstream(),
            "default_action_trains_models": False,
            "evaluation_used_to_screen_matrix": False,
            "principal_epoch": 50,
        }
        if len(entries) != 24:
            raise ValueError("Phase 6.5A downstream matrix must contain 24 entries")
        if MANIFEST.exists():
            existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
            if _manifest_for_comparison(existing) != _manifest_for_comparison(frozen):
                raise ValueError("existing Phase 6.5A downstream manifest differs from freeze")
        else:
            write_json(MANIFEST, frozen)
        results = []
        if args.execute:
            for entry in entries:
                config_payload = dict(entry["config"])
                config_payload["snapshot_epochs"] = tuple(config_payload["snapshot_epochs"])
                config_payload["device"] = args.device
                config = Phase65LSTMCapacityDownstreamConfig(**config_payload)
                run_root = Path(entry["run_root"])
                dataset = Path(entry["dataset_path"])
                feature = Path(entry["feature_path"])
                if run_root.is_dir():
                    result = validate_lstm_capacity_downstream(dataset, feature, run_root)
                    result["action"] = "validated_existing"
                else:
                    run_lstm_capacity_downstream(dataset, feature, run_root, config)
                    result = validate_lstm_capacity_downstream(dataset, feature, run_root)
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
        print(f"Phase 6.5A downstream bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
