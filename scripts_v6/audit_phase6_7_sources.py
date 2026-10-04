#!/usr/bin/env python3
"""Freeze the SaURL provenance/correctness/resource gate without training a run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_7_external_encoders import (  # noqa: E402
    REPORTING_LABEL,
    resource_smoke_external_encoder,
    smoke_test_external_encoders,
)


SOURCE_RECORD = ROOT / "docs" / "baselines" / "SaURL_TS" / "source_manifest.json"
OUTPUT = (
    ROOT
    / "experiments"
    / "phase6_7"
    / "feasibility"
    / "saurl_ts"
    / "feasibility_manifest.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--admit-training", action="store_true")
    parser.add_argument("--max-peak-memory-gib", type=float)
    parser.add_argument("--max-smoke-seconds", type=float)
    args = parser.parse_args()
    try:
        source = json.loads(SOURCE_RECORD.read_text(encoding="utf-8"))
        code = source["code"]
        gate = source["gate"]
        if (
            code.get("license_file_found") is not False
            or code.get("license_declaration_found") is not False
            or gate.get("public_source_code_reuse_admitted") is not False
            or gate.get("paper_guided_reimplementation_admitted") is not True
        ):
            raise ValueError("SaURL source/no-reuse record differs from the approved contract")
        paper_path = Path(source["paper"]["local_source_wsl"])
        if not paper_path.is_file() or sha256_file(paper_path) != source["paper"]["sha256"]:
            raise ValueError("SaURL paper identity/hash mismatch")
        if args.admit_training:
            if not args.device.startswith("cuda"):
                raise ValueError("training admission requires a CUDA resource smoke")
            if args.max_peak_memory_gib is None or args.max_peak_memory_gib <= 0.0:
                raise ValueError("--admit-training requires --max-peak-memory-gib")
            if args.max_smoke_seconds is None or args.max_smoke_seconds <= 0.0:
                raise ValueError("--admit-training requires --max-smoke-seconds")
        synthetic = smoke_test_external_encoders()
        real_smokes = []
        for walk in (1, 2):
            dataset = (
                ROOT
                / "experiments"
                / "phase5"
                / "data_preparation"
                / f"walk{walk}"
                / "market_1h_seq64_h2.npz"
            )
            real_smokes.append(
                resource_smoke_external_encoder(dataset, walk=walk, device=args.device)
            )
        memory_limit = (
            int(args.max_peak_memory_gib * 1024**3)
            if args.max_peak_memory_gib is not None
            else None
        )
        within_memory = (
            memory_limit is None
            or all(row["peak_cuda_memory_bytes"] <= memory_limit for row in real_smokes)
        )
        within_time = (
            args.max_smoke_seconds is None
            or all(row["elapsed_seconds"] <= args.max_smoke_seconds for row in real_smokes)
        )
        admitted = bool(args.admit_training and within_memory and within_time)
        payload = {
            "schema_version": "phase6-7-saurl-feasibility-v1",
            "phase": "6.7",
            "method": "saurl_frozen",
            "reporting_label": REPORTING_LABEL,
            "source_manifest_path": str(SOURCE_RECORD.resolve()),
            "source_manifest_sha256": sha256_file(SOURCE_RECORD),
            "paper_sha256": source["paper"]["sha256"],
            "audited_repository": code["repository"],
            "audited_commit": code["commit"],
            "software_license_found": False,
            "upstream_code_reuse_admitted": False,
            "independent_implementation_only": True,
            "synthetic_cpu_smoke": synthetic,
            "real_training_only_resource_smokes": real_smokes,
            "resource_limits": {
                "max_peak_memory_bytes": memory_limit,
                "max_smoke_seconds_per_walk": args.max_smoke_seconds,
                "within_memory_limit": within_memory,
                "within_time_limit": within_time,
            },
            "correctness_gate_passed": bool(
                synthetic["valid"] and all(row["valid"] for row in real_smokes)
            ),
            "cuda_gate_passed": bool(
                args.device.startswith("cuda") and all(row["valid"] for row in real_smokes)
            ),
            "admitted_for_training": admitted,
            "admission_requested_explicitly": bool(args.admit_training),
            "evaluation_metrics_read": False,
            "training_started": False,
            "fallback_if_not_admitted": "SISSEL-Frozen",
        }
        write_json(OUTPUT, payload)
        print(json.dumps({"valid": True, "manifest": str(OUTPUT.resolve()), **payload}, indent=2, sort_keys=True))
        return 0 if (not args.admit_training or admitted) else 1
    except Exception as exc:
        print(f"Phase 6.7 SaURL feasibility audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
