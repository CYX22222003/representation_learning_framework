#!/usr/bin/env python3
"""Freeze and optionally execute the two walk-specific XM-MV8 trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_9_xlstm_mixer import (  # noqa: E402
    MATRIX_SCHEMA_VERSION,
    METHOD,
    Phase69XLSTMMixerConfig,
    load_xlstm_mixer_data,
    phase_admission_path,
    phase_data_path,
    phase_run_root,
    run_xlstm_mixer_training,
    smoke_test_xlstm_mixer_training,
    validate_runtime_admission,
    validate_xlstm_mixer_training,
)


PHASE_ROOT = ROOT / "experiments" / "phase6_9" / "xlstm_mixer"
MATRIX_MANIFEST = PHASE_ROOT / "manifests" / "training_seed0.json"


def _comparison(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    smoke = normalized.get("synthetic_vanilla_cpu_smoke", {})
    if isinstance(smoke, dict):
        smoke.pop("loss", None)
        smoke.pop("gradient_norm_before_clip", None)
    for entry in normalized["entries"]:  # type: ignore[index]
        entry["config"]["device"] = "<runtime>"
        for field in ("dataset_path", "run_root"):
            entry[field] = entry[field].casefold()
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--backend", choices=("vanilla", "cuda"), default="vanilla")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        if args.execute and not args.device.startswith("cuda"):
            raise ValueError("XM-MV8 execution requires --device cuda")
        admission = phase_admission_path(ROOT)
        entries = []
        for walk in (1, 2):
            dataset_path = phase_data_path(ROOT, walk)
            data = load_xlstm_mixer_data(dataset_path, walk)
            config = Phase69XLSTMMixerConfig(
                walk=walk,
                batch_size=args.batch_size,
                device="cuda" if not args.execute else args.device,
                backend=args.backend,
            )
            entries.append(
                {
                    "walk": walk,
                    "method": METHOD,
                    "dataset_path": str(dataset_path.resolve()),
                    "dataset_sha256": data.dataset_sha256,
                    "data_manifest_sha256": data.manifest_sha256,
                    "train_identity_sha256": data.train_identity_sha256,
                    "evaluation_identity_sha256": data.evaluation_identity_sha256,
                    "run_root": str(phase_run_root(ROOT, walk).resolve()),
                    "config": config.to_dict(),
                }
            )
        frozen = {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "phase": "6.9",
            "method": METHOD,
            "seed": 0,
            "entry_count": 2,
            "entries": entries,
            "synthetic_vanilla_cpu_smoke": smoke_test_xlstm_mixer_training(),
            "default_action_trains_models": False,
            "validation_split": False,
            "early_stopping": False,
            "principal_epoch": 50,
            "extra_supervision_disclosed": True,
        }
        if MATRIX_MANIFEST.is_file():
            existing = json.loads(MATRIX_MANIFEST.read_text(encoding="utf-8"))
            if _comparison(existing) != _comparison(frozen):
                raise ValueError("existing XM-MV8 training matrix differs from the freeze")
        else:
            write_json(MATRIX_MANIFEST, frozen)

        results = []
        if args.execute:
            for entry in entries:
                dataset_path = Path(entry["dataset_path"])
                run_root = Path(entry["run_root"])
                payload = dict(entry["config"])
                payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
                payload["device"] = args.device
                config = Phase69XLSTMMixerConfig(**payload)
                validate_runtime_admission(admission, dataset_path, config)
                if (run_root / "training_complete.json").is_file():
                    result = validate_xlstm_mixer_training(
                        dataset_path, run_root, admission, device=args.device
                    )
                    result["action"] = "validated_existing"
                else:
                    result = run_xlstm_mixer_training(
                        dataset_path, run_root, admission, config
                    )
                    result["action"] = "trained_or_resumed"
                results.append(result)
        print(
            json.dumps(
                {
                    "valid": True,
                    "manifest": str(MATRIX_MANIFEST.resolve()),
                    "manifest_sha256": sha256_file(MATRIX_MANIFEST),
                    "executed": args.execute,
                    "runs": results,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"Phase 6.9 xLSTM-Mixer bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
