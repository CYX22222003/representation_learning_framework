#!/usr/bin/env python3
"""Audit LWA dependency/model readiness; admit training only after CUDA smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase5_walks import sha256_file  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_7_lwa import (  # noqa: E402
    REPORTING_LABEL,
    resource_smoke_lwa,
    smoke_test_lwa_pretraining,
    validate_lwa_view_cache,
)


SOURCE = ROOT / "docs" / "baselines" / "LWA" / "source_manifest.json"
OUTPUT = ROOT / "experiments" / "phase6_7" / "feasibility" / "lwa" / "feasibility_manifest.json"


def dataset_path(walk: int) -> Path:
    return ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"


def cache_path(walk: int) -> Path:
    return ROOT / "experiments" / "phase6_7" / "view_cache" / "lwa_frozen" / f"walk{walk}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--paper-path",
        type=Path,
        help="optional local copy of the audited paper for hash re-verification",
    )
    parser.add_argument("--admit-training", action="store_true")
    parser.add_argument("--max-peak-memory-gib", type=float)
    parser.add_argument("--max-smoke-seconds", type=float)
    args = parser.parse_args()
    try:
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        gate = source["gate"]
        if gate.get("public_source_code_reuse_admitted") is not False or gate.get("paper_guided_reimplementation_admitted") is not True:
            raise ValueError("LWA source/reimplementation gate differs from the approved contract")
        paper = args.paper_path or Path(source["paper"]["local_source_wsl"])
        paper_reverified = False
        if paper.is_file():
            if sha256_file(paper) != source["paper"]["sha256"]:
                raise ValueError("LWA paper identity/hash mismatch")
            paper_reverified = True
        elif args.paper_path is not None:
            raise FileNotFoundError(f"explicit LWA paper path does not exist: {paper}")
        import pywt

        if pywt.__version__ != source["approved_dependencies"]["pywavelets_version"]:
            raise ValueError("LWA PyWavelets runtime version differs from the approved pin")
        caches = []
        missing_caches = []
        for walk in (1, 2):
            if cache_path(walk).is_dir():
                caches.append(validate_lwa_view_cache(dataset_path(walk), cache_path(walk), walk=walk))
            else:
                missing_caches.append(walk)
        resource = []
        if args.admit_training:
            if not args.device.startswith("cuda"):
                raise ValueError("LWA training admission requires CUDA")
            if not args.max_peak_memory_gib or not args.max_smoke_seconds:
                raise ValueError("training admission requires positive memory and time limits")
            if missing_caches:
                raise FileNotFoundError(f"training admission requires LWA caches for walks {missing_caches}")
            for walk in (1, 2):
                resource.append({"walk": walk, **resource_smoke_lwa(cache_path(walk), device=args.device)})
        memory_limit = int(args.max_peak_memory_gib * 1024**3) if args.max_peak_memory_gib else None
        within_memory = memory_limit is None or all(row["peak_cuda_memory_bytes"] <= memory_limit for row in resource)
        within_time = args.max_smoke_seconds is None or all(row["elapsed_seconds"] <= args.max_smoke_seconds for row in resource)
        admitted = bool(args.admit_training and within_memory and within_time)
        payload = {
            "schema_version": "phase6-7-lwa-feasibility-v1",
            "phase": "6.7",
            "method": "lwa_frozen",
            "reporting_label": REPORTING_LABEL,
            "source_manifest_path": str(SOURCE.resolve()),
            "source_manifest_sha256": sha256_file(SOURCE),
            "paper_sha256": source["paper"]["sha256"],
            "paper_runtime_path": str(paper),
            "paper_file_reverified": paper_reverified,
            "paper_identity_authority": (
                "runtime file hash" if paper_reverified else "versioned source manifest"
            ),
            "independent_implementation_only": True,
            "pywavelets_version": pywt.__version__,
            "cache_validations": caches,
            "missing_cache_walks": missing_caches,
            "synthetic_cpu_smoke": smoke_test_lwa_pretraining(),
            "fixed_batch_resource_smokes": resource,
            "resource_limits": {"max_peak_memory_gib": args.max_peak_memory_gib, "max_smoke_seconds": args.max_smoke_seconds},
            "within_memory_limit": within_memory,
            "within_time_limit": within_time,
            "admitted_for_training": admitted,
            "evaluation_values_loaded": False,
        }
        write_json(OUTPUT, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.7 LWA audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
