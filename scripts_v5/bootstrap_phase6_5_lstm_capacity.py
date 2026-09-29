#!/usr/bin/env python3
"""Freeze Phase 6.5A two-layer LSTM runs; execute only with --execute."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file  # noqa: E402
from evaluation.phase6_encoder_variants import cka_sample_contract  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_5_lstm_capacity import (  # noqa: E402
    VARIANTS,
    Phase65LSTMCapacityConfig,
    run_lstm_capacity_encoder,
    smoke_test_lstm_capacity,
    validate_lstm_capacity_encoder,
)


ROOT_OUT = ROOT / "experiments" / "phase6_5"
CAPACITY_ROOT = ROOT_OUT / "lstm_capacity"
MANIFEST = ROOT_OUT / "manifests" / "lstm_capacity_pretraining_seed0.json"


def _manifest_for_comparison(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    for entry in normalized["entries"]:
        entry["config"]["device"] = "<runtime>"
        entry["dataset"] = entry["dataset"].casefold()
        entry["run_root"] = entry["run_root"].casefold()
    for source in normalized["source_datasets"].values():
        source["path"] = source["path"].casefold()
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
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
            for variant in VARIANTS:
                family = "contrastive" if variant.startswith("contrastive") else "byol"
                run_root = (
                    CAPACITY_ROOT
                    / "pretraining"
                    / f"walk{walk}"
                    / family
                    / "lstm2"
                    / "seed0"
                )
                entries.append(
                    {
                        "walk": walk,
                        "variant": variant,
                        "dataset": str(dataset.resolve()),
                        "run_root": str(run_root.resolve()),
                        "config": Phase65LSTMCapacityConfig(
                            variant=variant, walk=walk, device=args.device
                        ).to_dict(),
                    }
                )
        frozen = {
            "schema_version": "phase6-5a-lstm-capacity-pretraining-v1",
            "phase": "6.5A",
            "entries": entries,
            "entry_count": 4,
            "default_action_trains_models": False,
            "evaluation_values_loaded": False,
            "source_datasets": {
                f"walk{walk}": {
                    "path": str(
                        (
                            ROOT
                            / "experiments"
                            / "phase5"
                            / "data_preparation"
                            / f"walk{walk}"
                            / "market_1h_seq64_h2.npz"
                        ).resolve()
                    ),
                    "sha256": sha256_file(
                        ROOT
                        / "experiments"
                        / "phase5"
                        / "data_preparation"
                        / f"walk{walk}"
                        / "market_1h_seq64_h2.npz"
                    ),
                    "manifest_sha256": sha256_file(
                        ROOT
                        / "experiments"
                        / "phase5"
                        / "data_preparation"
                        / f"walk{walk}"
                        / "market_1h_seq64_h2.npz.manifest.json"
                    ),
                    "cka_sample": cka_sample_contract(
                        ROOT
                        / "experiments"
                        / "phase5"
                        / "data_preparation"
                        / f"walk{walk}"
                        / "market_1h_seq64_h2.npz"
                    ),
                }
                for walk in (1, 2)
            },
            "cpu_smoke_test": smoke_test_lstm_capacity(),
        }
        if len(entries) != 4:
            raise ValueError("Phase 6.5A pretraining matrix must contain four runs")
        if MANIFEST.exists():
            existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
            if _manifest_for_comparison(existing) != _manifest_for_comparison(frozen):
                raise ValueError("existing Phase 6.5A pretraining manifest differs from freeze")
        else:
            write_json(MANIFEST, frozen)
        results = []
        if args.execute:
            for entry in entries:
                dataset = Path(entry["dataset"])
                run_root = Path(entry["run_root"])
                config_payload = dict(entry["config"])
                config_payload["snapshot_epochs"] = tuple(config_payload["snapshot_epochs"])
                config_payload["device"] = args.device
                config = Phase65LSTMCapacityConfig(**config_payload)
                if run_root.is_dir():
                    result = validate_lstm_capacity_encoder(dataset, run_root)
                    result["action"] = "validated_existing"
                else:
                    result = run_lstm_capacity_encoder(dataset, run_root, config)
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
        print(f"Phase 6.5A LSTM-capacity bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
