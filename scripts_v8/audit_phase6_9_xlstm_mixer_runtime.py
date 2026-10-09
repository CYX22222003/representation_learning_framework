#!/usr/bin/env python3
"""Audit xLSTM-Mixer runtime readiness; admit training only after CUDA smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.xlstm_mixer import SOURCE_CONTRACT  # noqa: E402
from data_processing.phase5_walks import sha256_file  # noqa: E402
from training.phase5_downstream import resolve_device  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_9_xlstm_mixer import (  # noqa: E402
    ADMISSION_SCHEMA_VERSION,
    METHOD,
    implementation_fingerprint,
    phase_data_path,
    resource_smoke_xlstm_mixer,
    runtime_environment_payload,
    smoke_test_xlstm_mixer_training,
    xlstm_dependency_manifest,
    Phase69XLSTMMixerConfig,
    validate_runtime_admission,
)
from data_processing.phase6_9_xlstm_mixer import validate_future_path_data


SOURCE_MANIFEST = ROOT / "docs" / "baselines" / "xLSTM-Mixer" / "source_manifest.json"
FEASIBILITY_ROOT = ROOT / "experiments" / "phase6_9" / "xlstm_mixer" / "feasibility"
RUNTIME_AUDIT = FEASIBILITY_ROOT / "runtime_audit.json"
CUDA_ADMISSION = FEASIBILITY_ROOT / "cuda_admission.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("vanilla", "cuda"), default="vanilla")
    parser.add_argument("--admit-training", action="store_true")
    parser.add_argument("--physical-batch-size", type=int, default=512)
    parser.add_argument("--approved-smaller-batch", action="store_true")
    parser.add_argument("--max-peak-memory-gib", type=float)
    parser.add_argument("--max-smoke-seconds", type=float)
    args = parser.parse_args()
    try:
        source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
        gate = source["gate"]
        if (
            gate.get("owner_decisions_complete") is not True
            or gate.get("license_dependency_decision_complete") is not True
        ):
            raise ValueError("xLSTM-Mixer source/licence decisions are not complete")
        if args.physical_batch_size <= 0 or args.physical_batch_size > 512:
            raise ValueError("physical batch size must be in [1,512]")
        if args.physical_batch_size != 512 and not args.approved_smaller_batch:
            raise ValueError(
                "a smaller batch requires the explicit --approved-smaller-batch decision"
            )
        if args.admit_training:
            for walk in (1, 2):
                validate_future_path_data(ROOT, walk)
            if CUDA_ADMISSION.is_file():
                for walk in (1, 2):
                    validate_runtime_admission(CUDA_ADMISSION, phase_data_path(ROOT, walk),
                        Phase69XLSTMMixerConfig(walk=walk, device=args.device,
                            backend=args.backend, batch_size=args.physical_batch_size))
                existing = json.loads(CUDA_ADMISSION.read_text())
                requested_limits = {"max_peak_memory_gib": args.max_peak_memory_gib,
                                    "max_smoke_seconds": args.max_smoke_seconds}
                if existing.get("resource_limits") != requested_limits:
                    raise ValueError("existing admission resource limits differ; preserve the admitted recipe")
                print(json.dumps({"valid": True, "action": "validated_existing", "admitted": True}, indent=2))
                return 0

        device = resolve_device(args.device)
        dependency = xlstm_dependency_manifest()
        cpu_smoke = smoke_test_xlstm_mixer_training()
        environment = runtime_environment_payload(device)
        base = {
            "phase": "6.9",
            "method": METHOD,
            "source_manifest_path": str(SOURCE_MANIFEST.resolve()),
            "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "source_contract": SOURCE_CONTRACT.to_dict(),
            "source_contract_sha256": SOURCE_CONTRACT.sha256,
            "implementation_sha256": implementation_fingerprint(ROOT),
            "dependency": dependency,
            "environment": environment,
            "synthetic_vanilla_cpu_smoke": cpu_smoke,
            "physical_batch_size": args.physical_batch_size,
            "smaller_batch_owner_approved": bool(args.approved_smaller_batch),
        }
        if not args.admit_training:
            payload = {
                "schema_version": "phase6-9-xlstm-mixer-runtime-audit-v1",
                **base,
                "admitted": False,
                "training_launched": False,
            }
            write_json(RUNTIME_AUDIT, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        if device.type != "cuda":
            raise ValueError("training admission requires --device cuda")
        if not args.max_peak_memory_gib or args.max_peak_memory_gib <= 0:
            raise ValueError("training admission requires a positive memory limit")
        if not args.max_smoke_seconds or args.max_smoke_seconds <= 0:
            raise ValueError("training admission requires a positive time limit")
        torch_capability = environment.get("cuda_compute_capability")
        if args.backend == "cuda" and (not torch_capability or tuple(torch_capability) < (8, 0)):
            raise RuntimeError("the pinned CUDA sLSTM backend requires compute capability >= 8.0")

        walks = [
            resource_smoke_xlstm_mixer(
                phase_data_path(ROOT, walk),
                walk=walk,
                batch_size=args.physical_batch_size,
                device=args.device,
                backend=args.backend,
            )
            for walk in (1, 2)
        ]
        memory_limit = int(args.max_peak_memory_gib * 1024**3)
        within_memory = all(row["peak_cuda_memory_bytes"] <= memory_limit for row in walks)
        within_time = all(
            row["model_build_seconds"] + row["step_seconds"] <= args.max_smoke_seconds
            for row in walks
        )
        admitted = bool(
            within_memory
            and within_time
            and all(row["loss_finite"] for row in walks)
            and all(row["gradients_finite"] for row in walks)
            and all(row["same_backend_checkpoint_replay"] for row in walks)
        )
        payload = {
            "schema_version": ADMISSION_SCHEMA_VERSION,
            **base,
            "backend": args.backend,
            "resource_limits": {
                "max_peak_memory_gib": args.max_peak_memory_gib,
                "max_smoke_seconds": args.max_smoke_seconds,
            },
            "walks": walks,
            "within_memory_limit": within_memory,
            "within_time_limit": within_time,
            "admitted": admitted,
            "training_launched": False,
        }
        write_json(CUDA_ADMISSION, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if admitted else 1
    except Exception as exc:
        print(f"Phase 6.9 xLSTM-Mixer runtime audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
