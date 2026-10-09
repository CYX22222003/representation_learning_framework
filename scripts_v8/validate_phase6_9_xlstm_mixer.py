#!/usr/bin/env python3
"""Replay-validate both Phase 6.9 XM-MV8 trajectories on the same backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.phase6_9_xlstm_mixer import (  # noqa: E402
    phase_admission_path,
    phase_data_path,
    phase_run_root,
    validate_xlstm_mixer_training,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        if not args.device.startswith("cuda"):
            raise ValueError("XM-MV8 replay requires the same CUDA backend")
        admission = phase_admission_path(ROOT)
        results = [
            validate_xlstm_mixer_training(
                phase_data_path(ROOT, walk),
                phase_run_root(ROOT, walk),
                admission,
                device=args.device,
            )
            for walk in (1, 2)
        ]
        print(json.dumps({"valid": True, "status": "pass", "runs": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.9 xLSTM-Mixer validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
