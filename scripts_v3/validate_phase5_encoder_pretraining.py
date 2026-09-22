"""Replay and validate all six completed Phase 5 encoder trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_phase5_encoder import dataset_path, default_run_root  # noqa: E402
from training.phase5_encoder import ENCODERS, validate_encoder_run  # noqa: E402


def main() -> int:
    matrix_path = ROOT / "experiments" / "phase5" / "manifests" / "encoder_pretraining_seed0.json"
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        if matrix.get("status") != "complete":
            raise ValueError("encoder matrix is not marked complete")
        for walk in (1, 2):
            for encoder in ENCODERS:
                result = validate_encoder_run(
                    dataset_path(walk), default_run_root(walk, encoder)
                )
                print(result, flush=True)
        print("All Phase 5 encoder trajectories replay-valid", flush=True)
        return 0
    except Exception as exc:
        print(f"Phase 5 encoder validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
