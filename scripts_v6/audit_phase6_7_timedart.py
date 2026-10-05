#!/usr/bin/env python3
"""Run TimeDART CPU or CUDA feasibility checks without training a trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase6_7_timedart_data import METHOD, encoder_dataset_path  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_7_timedart import resource_smoke_timedart, smoke_test_timedart  # noqa: E402


ADMISSION = ROOT / "experiments" / "phase6_7" / "feasibility" / "timedart" / "cuda_admission.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--admit-training", action="store_true")
    parser.add_argument("--max-peak-memory-gib", type=float, default=23.0)
    parser.add_argument("--max-smoke-seconds", type=float, default=600.0)
    args = parser.parse_args()
    try:
        if not args.admit_training:
            payload = {
                "method": METHOD,
                "device": args.device,
                "synthetic_smoke": smoke_test_timedart(),
                "training_admitted": False,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.device != "cuda":
            raise ValueError("training admission must use --device cuda")
        walks = [
            resource_smoke_timedart(
                encoder_dataset_path(ROOT, walk), walk=walk, device=args.device
            )
            for walk in (1, 2)
        ]
        memory_limit = int(args.max_peak_memory_gib * 1024**3)
        admitted = all(
            row["peak_cuda_memory_bytes"] <= memory_limit
            and row["elapsed_seconds"] <= args.max_smoke_seconds
            and row["loss_finite"]
            and row["gradients_finite"]
            and row["features_finite"]
            for row in walks
        )
        payload = {
            "schema_version": "phase6-7-timedart-cuda-admission-v1",
            "method": METHOD,
            "device": args.device,
            "physical_batch_size": 16,
            "max_peak_memory_gib": args.max_peak_memory_gib,
            "max_smoke_seconds": args.max_smoke_seconds,
            "walks": walks,
            "admitted": admitted,
        }
        write_json(ADMISSION, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if admitted else 1
    except Exception as exc:
        print(f"TimeDART feasibility audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

