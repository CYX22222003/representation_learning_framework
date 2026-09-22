#!/usr/bin/env python3
"""Replay-validate the frozen Phase 5 raw-sequence baseline matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts_v3") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts_v3"))

from bootstrap_phase5_baselines import MATRIX_PATH
from training.phase5_baselines import validate_baseline_run


def main() -> None:
    if not MATRIX_PATH.is_file():
        raise FileNotFoundError("freeze the baseline matrix before validation")
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    if matrix.get("run_count") != 12 or len(matrix.get("runs", [])) != 12:
        raise ValueError("baseline matrix is incomplete")
    results = [
        validate_baseline_run(Path(row["dataset_path"]), Path(row["run_root"]))
        for row in matrix["runs"]
    ]
    print(json.dumps({"valid": True, "runs": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
