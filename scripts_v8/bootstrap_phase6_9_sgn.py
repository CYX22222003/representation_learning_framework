#!/usr/bin/env python3
"""Freeze/admit SGN-C; train both walks only with explicit --execute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.phase5_encoder import environment_manifest, resolve_device, write_json  # noqa: E402
from training.phase6_9_sgn import (  # noqa: E402
    METHOD, SCHEMA, SGNTrainingConfig, freeze_training_matrix,
    implementation_fingerprint, load_initial_logits, load_sgn_data,
    phase_root, run_sgn_training, smoke_test_sgn, validate_sgn_training,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--admit-training", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-peak-memory-gib", type=float, default=6.0)
    parser.add_argument("--max-smoke-seconds", type=float, default=600.0)
    args = parser.parse_args()
    try:
        if args.execute and args.validate_only:
            raise ValueError("--execute and --validate-only are mutually exclusive")
        if (args.execute or args.admit_training) and not args.device.startswith("cuda"):
            raise ValueError("SGN-C execution/admission requires CUDA")
        matrix = freeze_training_matrix(ROOT)
        cpu_smoke = smoke_test_sgn()
        root = phase_root(ROOT)
        if args.admit_training:
            device = resolve_device(args.device)
            environment = environment_manifest(device)
            data, initialization = {}, {}
            for walk in (1, 2):
                _, data[str(walk)] = load_sgn_data(ROOT, walk)
                _, initialization[str(walk)] = load_initial_logits(ROOT, walk)
            semantic = {
                "schema_version": SCHEMA, "method": METHOD,
                "implementation_sha256": implementation_fingerprint(ROOT),
                "batch_size": 256, "environment": environment,
                "data": data, "initialization": initialization,
                "max_peak_memory_gib": args.max_peak_memory_gib,
                "max_smoke_seconds": args.max_smoke_seconds,
                "admitted": True,
            }
            path = root / "feasibility/cuda_admission.json"
            if path.exists():
                existing = json.loads(path.read_text())
                if any(existing.get(key) != value for key, value in semantic.items()):
                    if any((root / f"downstream/classification_h2/walk{walk}/{METHOD}/seed0").exists() for walk in (1, 2)):
                        raise ValueError("existing SGN-C CUDA admission differs after a run root was created")
                    path.unlink()
            if not path.exists():
                walk1_data, _ = load_sgn_data(ROOT, 1)
                walk1_logits, _ = load_initial_logits(ROOT, 1)
                from tasks.phase2_classification.protocols import class_priors
                probe = smoke_test_sgn(
                    args.device, batch_size=256, initial_logits=walk1_logits,
                    contexts=walk1_data["X_train"], labels=walk1_data["y_train"],
                    priors=class_priors(walk1_data["y_train"], 3),
                )
                if probe["peak_cuda_memory_bytes"] > args.max_peak_memory_gib * 1024**3:
                    raise RuntimeError("SGN-C peak memory exceeds admission ceiling")
                if probe["seconds"] > args.max_smoke_seconds:
                    raise RuntimeError("SGN-C smoke time exceeds admission ceiling")
                semantic["probe"] = probe
                write_json(path, semantic)
        results = []
        if args.execute or args.validate_only:
            for walk in (1, 2):
                config = SGNTrainingConfig(walk=walk, device=args.device)
                results.append(run_sgn_training(ROOT, config) if args.execute else validate_sgn_training(ROOT, config))
        readiness = {
            "schema_version": SCHEMA, "method": METHOD, "matrix_valid": True,
            "matrix": matrix, "cpu_smoke": cpu_smoke,
            "full_trajectories_launched_by_this_command": args.execute,
            "results": results,
        }
        write_json(root / "readiness.json", readiness)
        print(json.dumps({"valid": True, "training_requested": args.execute, "results": results}, indent=2))
        return 0
    except Exception as exc:
        print(f"SGN-C gate failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
