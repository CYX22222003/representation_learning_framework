#!/usr/bin/env python3
"""Freeze/smoke-test Phase 6 volatility; train only with explicit --execute."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file  # noqa: E402
from data_processing.phase6_volatility_labels import (  # noqa: E402
    validate_volatility_label_bundle_files,
)
from features.phase6_volatility_features import (  # noqa: E402
    validate_phase6_volatility_feature_bundle,
)
from models.encoder_variants import (  # noqa: E402
    build_byol_variant,
    build_contrastive_variant,
)
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_volatility import (  # noqa: E402
    MODELS,
    Phase6VolatilityConfig,
    run_volatility_training,
    smoke_test_volatility_models,
    validate_volatility_run,
)


VOLATILITY_ROOT = ROOT / "experiments" / "phase6" / "volatility_prediction"
MANIFEST_PATH = VOLATILITY_ROOT / "manifests" / "model_matrix_seed0.json"
FREEZE_DOC = (
    ROOT
    / "docs"
    / "phase_plan"
    / "2026-09-25-phase-6-volatility-model-matrix-freeze.md"
)
TEMPORAL_CONFIGS = (
    "H0",
    "HC-SL",
    "HC-ST",
    "HB-SL",
    "HB-ST",
    "HC-AL",
    "HC-AT",
    "HB-AL",
    "HB-AT",
    "HC-DC",
    "HB-DC",
)


def _paths(walk: int) -> tuple[Path, Path]:
    label = (
        VOLATILITY_ROOT
        / "data_preparation"
        / f"walk{walk}"
        / "volatility_1h_seq64_h8.npz"
    )
    features = VOLATILITY_ROOT / "features" / f"walk{walk}" / "five_branch_epoch50_h8.npz"
    return label, features


def _temporal_smoke() -> dict[str, object]:
    import torch

    results = {}
    values = torch.randn(4, 64, 5)
    for variant in ("contrastive_lstm", "contrastive_transformer"):
        model = build_contrastive_variant(variant, input_dim=5)
        hidden, projected = model(values)
        if hidden.shape != (4, 128) or projected.shape != (4, 128):
            raise RuntimeError(f"temporal contrastive smoke failed: {variant}")
        results[variant] = {"embedding_shape": list(hidden.shape)}
    for variant in ("byol_lstm", "byol_transformer"):
        model = build_byol_variant(variant, input_dim=5)
        hidden = model.encode(values)
        output = model(values, values)
        if hidden.shape != (4, 128) or any(item.shape != (4, 128) for item in output):
            raise RuntimeError(f"temporal BYOL smoke failed: {variant}")
        results[variant] = {"embedding_shape": list(hidden.shape)}
    return {"valid": True, "models": results}


def freeze_manifest() -> dict[str, object]:
    if not FREEZE_DOC.is_file():
        raise FileNotFoundError(FREEZE_DOC)
    inputs = {}
    for walk in (1, 2):
        label, features = _paths(walk)
        label_result = validate_volatility_label_bundle_files(label)
        feature_result = validate_phase6_volatility_feature_bundle(features)
        if label_result["row_counts"] != {
            "train": feature_result["train_rows"],
            "test": feature_result["test_rows"],
        }:
            raise ValueError(f"walk {walk} label/feature row mismatch")
        inputs[f"walk{walk}"] = {
            "label_path": str(label.resolve()),
            "label_sha256": sha256_file(label),
            "feature_path": str(features.resolve()),
            "feature_sha256": sha256_file(features),
            "row_counts": label_result["row_counts"],
            "identity_hashes": label_result["identity_hashes"],
            "target_hashes": label_result["target_hashes"],
        }
    smoke = {
        "volatility_models": smoke_test_volatility_models(),
        "temporal_encoder_definitions": _temporal_smoke(),
    }
    entries = []
    for walk in (1, 2):
        for configuration in TEMPORAL_CONFIGS:
            entries.append(
                {
                    "walk": walk,
                    "family": "framework",
                    "configuration": configuration,
                    "feature_width": 445 if configuration in TEMPORAL_CONFIGS[:5] else 573,
                    "run_root": str(
                        (
                            ROOT
                            / "experiments"
                            / "phase6"
                            / "encoder_variants"
                            / "downstream"
                            / "realised_variance"
                            / configuration.lower()
                            / f"walk{walk}"
                            / "seed0"
                        ).resolve()
                    ),
                    "input_ready": configuration == "H0",
                }
            )
        for model in ("raw_ohlcv_mlp", "raw_lstm"):
            entries.append(
                {
                    "walk": walk,
                    "family": "baseline",
                    "configuration": model,
                    "run_root": str(
                        (VOLATILITY_ROOT / "runs" / model / f"walk{walk}" / "seed0").resolve()
                    ),
                    "input_ready": True,
                }
            )
    payload = {
        "schema_version": "phase6-volatility-model-matrix-v2",
        "phase": 6,
        "task": "future_realised_variance_h8",
        "seed": 0,
        "freeze_document": str(FREEZE_DOC.resolve()),
        "freeze_document_sha256": sha256_file(FREEZE_DOC),
        "inputs": inputs,
        "training_contract": Phase6VolatilityConfig(
            model="framework_h0", walk=1, device="cuda"
        ).to_dict(),
        "entries": entries,
        "entry_count": len(entries),
        "framework_entry_count": 22,
        "baseline_entry_count": 4,
        "smoke_tests": smoke,
        "evaluation_values_inspected_by_bootstrap": False,
        "default_action_trains_models": False,
        "phased_execution": {
            "ready_now": ["H0", "raw_ohlcv_mlp", "raw_lstm"],
            "pending_temporal_features": list(TEMPORAL_CONFIGS[1:]),
            "deferred_next_round": ["garch_lstm"],
            "matrix_membership_cannot_change_from_results": True,
        },
    }
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing Phase 6 volatility matrix manifest differs from freeze")
    else:
        write_json(MANIFEST_PATH, payload)
    return payload


def execute_ready(device: str) -> list[dict[str, object]]:
    payload = freeze_manifest()
    if payload["entry_count"] != 26:
        raise ValueError("complete Phase 6 volatility matrix was not frozen")
    results = []
    for walk in (1, 2):
        label, features = _paths(walk)
        for model_name in MODELS:
            run_root = VOLATILITY_ROOT / "runs" / model_name / f"walk{walk}" / "seed0"
            feature_path = features if model_name == "framework_h0" else None
            if run_root.is_dir():
                result = validate_volatility_run(label, feature_path, run_root)
                result["action"] = "validated_existing"
            else:
                config = Phase6VolatilityConfig(
                    model=model_name, walk=walk, device=device
                )
                run_volatility_training(label, feature_path, run_root, config)
                result = validate_volatility_run(label, feature_path, run_root)
                result["action"] = "trained"
            results.append(result)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = freeze_manifest()
        result: dict[str, object] = {
            "valid": True,
            "manifest_path": str(MANIFEST_PATH.resolve()),
            "entry_count": manifest["entry_count"],
            "smoke_tests": manifest["smoke_tests"],
            "executed": False,
        }
        if args.execute:
            result["runs"] = execute_ready(args.device)
            result["executed"] = True
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 volatility bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
