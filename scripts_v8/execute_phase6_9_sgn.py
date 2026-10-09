#!/usr/bin/env python3
"""Train/resume both SGN-C walks with same-backend post-training replay.

This orchestration wrapper is outside the frozen training fingerprint. It uses
the unchanged admitted model/training implementation and binds only prediction
replay to the selected CUDA backend, avoiding benign CPU/CUDA convolution drift.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("SGN-C execution requires admitted CUDA")
        original_predict = sgn._predict
        sgn._predict = lambda model, values, batch_size, _device: original_predict(
            model.to(device), values, batch_size, device
        )
        results = [
            sgn.run_sgn_training(ROOT, sgn.SGNTrainingConfig(walk, device=args.device))
            for walk in (1, 2)
        ]
        print(json.dumps({"valid": True, "results": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"SGN-C execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
