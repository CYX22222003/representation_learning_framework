#!/usr/bin/env python3
"""Check xLSTM-Mixer CPU/CUDA compatibility using synthetic inputs only."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from platform_integration.xlstm_mixer_environment import (  # noqa: E402
    check_xlstm_mixer_environment,
)
from training.phase5_encoder import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--backend", choices=("vanilla", "cuda"), default="vanilla",
        help="start with vanilla PyTorch operations on GPU; cuda compiles the custom kernel",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()
    try:
        payload = check_xlstm_mixer_environment(args.device, args.batch_size, args.backend)
        status = 0
    except Exception as exc:
        payload = {
            "valid": False,
            "device": args.device,
            "backend": args.backend,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "training_admitted": False,
            "training_launched": False,
        }
        # Compiler diagnostics must survive a failed JIT build in the run log.
        traceback.print_exc(file=sys.stderr)
        status = 1
    if args.output is not None:
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
