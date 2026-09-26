#!/usr/bin/env python3
"""Freeze Phase 6 temporal encoder runs; execute only with --execute."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.encoder_variants import BYOL_VARIANTS, CONTRASTIVE_VARIANTS  # noqa: E402
from data_processing.phase5_walks import sha256_file  # noqa: E402
from evaluation.phase6_encoder_variants import cka_sample_contract  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_encoder_variants import (  # noqa: E402
    Phase6EncoderVariantConfig,
    run_encoder_variant,
    smoke_test_encoder_variants,
    validate_encoder_variant,
)


VARIANTS = (*CONTRASTIVE_VARIANTS, *BYOL_VARIANTS)
ROOT_OUT = ROOT / "experiments" / "phase6" / "encoder_variants"
MANIFEST = ROOT_OUT / "manifests" / "pretraining_seed0.json"


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
                family, backbone = variant.split("_", 1)
                run_root = (
                    ROOT_OUT / "pretraining" / f"walk{walk}" / family / backbone / "seed0"
                )
                entries.append(
                    {
                        "walk": walk,
                        "variant": variant,
                        "dataset": str(dataset.resolve()),
                        "run_root": str(run_root.resolve()),
                        "config": Phase6EncoderVariantConfig(
                            variant=variant, walk=walk, device=args.device
                        ).to_dict(),
                    }
                )
        frozen = {
            "schema_version": "phase6-temporal-pretraining-matrix-v1",
            "entries": entries,
            "entry_count": 8,
            "default_action_trains_models": False,
            "evaluation_values_loaded": False,
            "source_datasets": {
                f"walk{walk}": {
                    "path": str((ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz").resolve()),
                    "sha256": sha256_file(ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"),
                    "manifest_sha256": sha256_file(ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz.manifest.json"),
                    "cka_sample": cka_sample_contract(ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"),
                }
                for walk in (1, 2)
            },
            "cpu_smoke_test": smoke_test_encoder_variants(),
        }
        if MANIFEST.exists():
            existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
            # Device is execution metadata, not a scientific configuration.
            for payload in (existing, frozen):
                for entry in payload["entries"]:
                    entry["config"]["device"] = "<runtime>"
            if existing != frozen:
                raise ValueError("existing temporal pretraining manifest differs from freeze")
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
                config = Phase6EncoderVariantConfig(**config_payload)
                if run_root.is_dir():
                    result = validate_encoder_variant(dataset, run_root)
                    result["action"] = "validated_existing"
                else:
                    result = run_encoder_variant(dataset, run_root, config)
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
        print(f"Phase 6 encoder-variant bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
