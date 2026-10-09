#!/usr/bin/env python3
"""Report all fixed XM-MV8 snapshots, without a best-on-test selection."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training.phase5_encoder import write_json
from training.phase6_9_xlstm_mixer import (
    phase_admission_path, phase_data_path, phase_run_root, validate_xlstm_mixer_training,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        rows = []
        for walk in (1, 2):
            run = phase_run_root(ROOT, walk)
            validate_xlstm_mixer_training(phase_data_path(ROOT, walk), run,
                                         phase_admission_path(ROOT), device=args.device)
            rows.extend(json.loads((run / "sweep_metrics.json").read_text())["snapshots"])
        output = ROOT / "experiments" / "phase6_9" / "xlstm_mixer" / "reports" / "seed0"
        write_json(output / "summary.json", {"snapshots": rows, "principal_epoch": 50,
            "matched_comparison_complete": False,
            "disclosure": "Full next-eight-hour OHLCV supervision; headline extracts close[t+8]."})
        lines = ["# XM-MV8 seed-0 training diagnostics", "",
            "Full OHLCV supervision at t+1..t+8; headline uses close[t+8]. Epoch 50 is predeclared.", "",
            "This is not a completed fair baseline comparison. Matched H0-D0 and Raw-LSTM reruns remain required.", "",
            "| Walk | Epoch | MAE | RMSE | Movement Rank IC | Full-path MAE |",
            "|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['walk']} | {row['epoch']} | {row['mae']:.6g} | {row['rmse']:.6g} | {row['implied_delta_spearman']:.6g} | {row['full_path_mae']:.6g} |")
        (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"valid": True, "report": str(output / "summary.md")}))
        return 0
    except Exception as exc:
        print(f"XM-MV8 reporting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
