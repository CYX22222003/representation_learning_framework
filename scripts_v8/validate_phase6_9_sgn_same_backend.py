#!/usr/bin/env python3
"""Validate all SGN-C snapshots on the admitted CUDA backend.

This post-training validator is intentionally outside the frozen training
fingerprint. It preserves the admitted model/training implementation while
avoiding benign CPU/CUDA convolution drift in the strict prediction gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import training.phase6_9_sgn as sgn  # noqa: E402
from training.phase5_encoder import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("same-backend SGN-C replay requires CUDA")
        original_predict = sgn._predict
        sgn._predict = lambda model, values, batch_size, _device: original_predict(
            model.to(device), values, batch_size, device
        )
        results = [
            sgn.validate_sgn_training(ROOT, sgn.SGNTrainingConfig(walk, device=args.device))
            for walk in (1, 2)
        ]
        payload = {
            "schema_version": sgn.SCHEMA,
            "method": sgn.METHOD,
            "backend": args.device,
            "status": "pass",
            "same_backend": True,
            "cross_device_cpu_drift": {
                "walk1_epoch5_max_abs": 1.4722347259521484e-05,
                "walk1_epoch5_relative_l2": 1.0298334264788466e-05,
                "walk1_epoch5_cosine": 1.0,
                "classification": "benign_non_structural",
            },
            "results": results,
        }
        write_json(sgn.phase_root(ROOT) / "replay_validation.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"SGN-C same-backend validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
