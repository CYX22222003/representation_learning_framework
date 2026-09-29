#!/usr/bin/env python3
"""Freeze Phase 6.5B strict GARCH--LSTM runs; train only with --execute."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.garch_lstm_stacking.phase6_5 import (  # noqa: E402
    Phase65GarchLSTMConfig,
    make_calendar_expanding_oof_plan,
    run_phase65_garch_lstm,
    smoke_test_phase65_garch_lstm,
    validate_phase65_garch_lstm_run,
)
from data_processing.phase5_walks import sha256_file  # noqa: E402
from data_processing.phase6_volatility_labels import validate_volatility_label_bundle_files  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402


PHASE_ROOT = ROOT / "experiments" / "phase6_5" / "garch_lstm"
MANIFEST = ROOT / "experiments" / "phase6_5" / "manifests" / "garch_lstm_seed0.json"


def paths(walk: int) -> tuple[Path, Path, Path]:
    label = ROOT / "experiments" / "phase6" / "volatility_prediction" / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz"
    raw = ROOT / "experiments" / "phase6" / "volatility_prediction" / "runs" / "raw_lstm" / f"walk{walk}" / "seed0"
    run = PHASE_ROOT / f"walk{walk}" / "seed0"
    return label, raw, run


def comparable(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    for entry in normalized["entries"]:
        entry["config"]["device"] = "<runtime>"
        for key in ("label_path", "raw_lstm_root", "run_root"):
            entry[key] = entry[key].casefold()
    return normalized


def freeze(device: str) -> dict[str, object]:
    entries = []
    for walk in (1, 2):
        label, raw, run = paths(walk)
        validation = validate_volatility_label_bundle_files(label)
        with np.load(label, allow_pickle=False) as stored:
            plan = make_calendar_expanding_oof_plan(
                stored["train_decision_availability_ns"],
                stored["train_target_availability_ns"],
                stored["train_condition_ids"],
                decision_date_ns=stored["train_decision_date_ns"],
            )
        entries.append(
            {
                "walk": walk,
                "label_path": str(label.resolve()),
                "label_sha256": sha256_file(label),
                "label_validation": validation,
                "raw_lstm_root": str(raw.resolve()),
                "raw_lstm_training_complete_sha256": sha256_file(raw / "training_complete.json"),
                "run_root": str(run.resolve()),
                "oof_plan": plan.manifest(),
                "config": Phase65GarchLSTMConfig(walk=walk, device=device).to_dict(),
            }
        )
    payload = {
        "schema_version": "phase6-5b-garch-lstm-matrix-v1",
        "phase": "6.5B",
        "entry_count": 2,
        "entries": entries,
        "cpu_smoke_test": smoke_test_phase65_garch_lstm(),
        "default_action_trains_models": False,
        "evaluation_rows_validated_for_provenance": True,
        "evaluation_metrics_loaded_by_bootstrap": False,
        "principal_epoch": 50,
    }
    if MANIFEST.exists():
        if comparable(json.loads(MANIFEST.read_text(encoding="utf-8"))) != comparable(payload):
            raise ValueError("existing Phase 6.5B manifest differs from freeze")
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
                label = Path(entry["label_path"])
                raw = Path(entry["raw_lstm_root"])
                run = Path(entry["run_root"])
                payload = dict(entry["config"])
                payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
                payload["device"] = args.device
                config = Phase65GarchLSTMConfig(**payload)
                if run.is_dir():
                    result = validate_phase65_garch_lstm_run(label, raw, run)
                    result["action"] = "validated_existing"
                else:
                    run_phase65_garch_lstm(label, raw, run, config)
                    result = validate_phase65_garch_lstm_run(label, raw, run)
                    result["action"] = "trained"
                runs.append(result)
        print(json.dumps({"valid": True, "manifest": str(MANIFEST.resolve()), "executed": args.execute, "runs": runs}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5B GARCH--LSTM bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
