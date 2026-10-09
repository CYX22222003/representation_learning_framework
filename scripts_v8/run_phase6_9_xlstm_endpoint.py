#!/usr/bin/env python3
"""Audit/freeze XM-C8; train only with explicit --execute and CUDA admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.xlstm_mixer.endpoint import METHOD_ID, architecture_manifest
from training.phase5_encoder import resolve_device, write_json
from training.phase6_9_xlstm_endpoint import (
    SCHEMA, EndpointTrainingConfig, audit_existing_controls, dependency_identity, implementation_fingerprint,
    load_endpoint_data, phase_root, run_endpoint_training, run_root, smoke_test,
    validate_endpoint_training,
)
from training.phase6_9_xlstm_mixer import runtime_environment_payload


def freeze_or_validate(path: Path, payload: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"existing XM-C8 freeze differs: {path}")
    else:
        write_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("vanilla", "cuda"), default="vanilla")
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
            raise ValueError("training/admission requires --device cuda")
        if args.max_peak_memory_gib <= 0 or args.max_smoke_seconds <= 0:
            raise ValueError("admission limits must be positive")
        root = phase_root(ROOT)
        manifests, controls = {}, {}
        for walk in (1, 2):
            _, manifest = load_endpoint_data(ROOT, walk)
            manifests[str(walk)] = manifest
            controls[str(walk)] = audit_existing_controls(ROOT, walk, manifest)
        cpu_smoke = smoke_test()
        matrix = {"schema_version": SCHEMA, "method_id": METHOD_ID,
                  "implementation_sha256": implementation_fingerprint(),
                  "architecture": architecture_manifest(), "data": manifests,
                  "controls": controls, "principal_epoch": 50,
                  "entries": [EndpointTrainingConfig(walk, backend=args.backend).to_dict() for walk in (1, 2)],
                  "validation_split": False, "early_stopping": False,
                  "default_action_trains_models": False,
                  "synthetic_cpu_smoke_passed": cpu_smoke["valid"]}
        freeze_or_validate(root / "manifests/training_seed0.json", matrix)
        if args.admit_training:
            device = resolve_device(args.device)
            admission = {"schema_version": SCHEMA, "method_id": METHOD_ID,
                         "implementation_sha256": implementation_fingerprint(),
                         "backend": args.backend, "batch_size": 512,
                         "environment": runtime_environment_payload(device),
                         "dependency": dependency_identity(), "data": manifests,
                         "max_peak_memory_gib": args.max_peak_memory_gib,
                         "max_smoke_seconds": args.max_smoke_seconds, "admitted": True}
            path = root / "feasibility/cuda_admission.json"
            if path.exists():
                existing = json.loads(path.read_text())
                if any(existing.get(key) != value for key, value in admission.items()):
                    raise ValueError("existing endpoint admission has changed semantics")
            else:
                probe = smoke_test(args.device, args.backend, batch_size=512)
                if probe["peak_cuda_memory_bytes"] > args.max_peak_memory_gib * 1024**3 or probe["seconds"] > args.max_smoke_seconds:
                    raise RuntimeError("XM-C8 selected-backend resource admission failed")
                admission["probe"] = probe
                write_json(path, admission)
        results = []
        if args.execute or args.validate_only:
            for walk in (1, 2):
                config = EndpointTrainingConfig(walk, device=args.device, backend=args.backend)
                results.append(run_endpoint_training(ROOT, config) if args.execute else validate_endpoint_training(ROOT, config))
            rows = []
            for walk in (1, 2):
                locations = {METHOD_ID: run_root(ROOT, walk),
                             **{name: Path(row["run_root"]) for name, row in controls[str(walk)].items()}}
                for method, location in locations.items():
                    metrics = json.loads((location / "e50/metrics.json").read_text())
                    price = metrics["price"]["overall"]
                    rows.append({"walk": walk, "method": method, "mae": price["mae"], "rmse": price["rmse"],
                                 "implied_movement_spearman": metrics["implied_movement"]["spearman"]})
            write_json(root / "reports/seed0/summary.json", {"principal_epoch": 50,
                       "endpoint_matched_comparison": True, "rows": rows,
                       "controls": controls, "replay": results})
            report = "# Endpoint-only xLSTM-Mixer comparison\n\nSeed 0, fixed epoch 50; original h8 price rows and endpoint-only MSE.\n\n"
            report += "| Walk | Method | MAE | RMSE | Movement Spearman |\n|---:|---|---:|---:|---:|\n"
            report += "".join(f"| {r['walk']} | {r['method']} | {r['mae']:.9f} | {r['rmse']:.9f} | {r['implied_movement_spearman']:.6f} |\n" for r in rows)
            report += "\nXM-C8 is an endpoint-only adaptation, not a paper reproduction or frozen-representation control. Its inverse-RevIN output and clip norm differ from the sigmoid controls. Original XM-MV8 results are separate contextual evidence. Preserve persistence diagnostics in each run's metrics; no universal, multi-seed, significance, or trading claim follows.\n"
            (root / "reports/seed0/summary.md").write_text(report)
        write_json(root / "readiness.json", {"method_id": METHOD_ID, "matrix_valid": True,
                   "original_rows_preserved": True, "existing_controls_audited": True,
                   "cpu_smoke": cpu_smoke, "full_trajectories_launched_by_this_command": args.execute,
                   "results": results})
        print(json.dumps({"method_id": METHOD_ID, "matrix_valid": True,
                          "training_requested": args.execute, "results": results}, indent=2))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        print(f"XM-C8 gate failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
