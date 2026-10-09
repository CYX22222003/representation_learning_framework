#!/usr/bin/env python3
"""Report SGN-C against exact-row H0/Raw-LSTM/Raw-MLP controls."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_9_sgn import SNAPSHOTS, phase_root, run_root  # noqa: E402


def control_path(method: str, walk: int, epoch: int) -> Path:
    if method == "h0_d0":
        return ROOT / f"experiments/phase5/downstream/walk{walk}/classification/seed0/e{epoch}/metrics.json"
    return ROOT / f"experiments/phase5/baselines/tasks/classification_h2/{method}/walk{walk}/seed0/e{epoch}/metrics.json"


def main() -> int:
    try:
        rows = []
        for walk in (1, 2):
            for epoch in SNAPSHOTS:
                sgn = json.loads((run_root(ROOT, walk) / f"e{epoch}/metrics.json").read_text())
                metrics = sgn["sgn_c"]
                rows.append({"walk": walk, "epoch": epoch, "method": "sgn_c",
                             "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                             "balanced_accuracy": metrics["balanced_accuracy"], "nll": metrics["nll"],
                             "multiclass_brier": metrics["multiclass_brier"],
                             "predicted_class_counts": metrics["predicted_class_counts"],
                             "hard_group_ids": sgn["hard_group_ids"]})
                for method in ("h0_d0", "raw_ohlcv_lstm", "raw_ohlcv_mlp"):
                    control = json.loads(control_path(method, walk, epoch).read_text())
                    rows.append({"walk": walk, "epoch": epoch, "method": method,
                                 "accuracy": control["accuracy"], "macro_f1": control["macro_f1"],
                                 "balanced_accuracy": control["balanced_accuracy"], "nll": control["nll"],
                                 "multiclass_brier": control["multiclass_brier"]})
        destination = phase_root(ROOT) / "reports/seed0"
        write_json(destination / "matched_comparison.json", {"principal_epoch": 50, "rows": rows})
        principal = [row for row in rows if row["epoch"] == 50]
        lines = [
            "# SGN-C matched classification comparison", "",
            "Seed 0; every method uses the exact original h2/tau=0.001 rows. Epoch 50 is the predeclared principal result.", "",
            "| Walk | Method | Accuracy | Macro-F1 | Balanced accuracy | NLL | Brier |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in principal:
            lines.append(
                f"| {row['walk']} | {row['method']} | {row['accuracy']:.6f} | "
                f"{row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} | "
                f"{row['nll']:.6f} | {row['multiclass_brier']:.6f} |"
            )
        lines += [
            "", "## Interpretation", "",
            "SGN-C does not beat H0-D0 on principal macro-F1 or balanced accuracy in either walk. It also trails Raw LSTM on those metrics in both walks and trails Raw MLP on macro-F1 in both walks. Walk 2 degrades substantially by epoch 50; the stronger epoch-5 result is retained but not selected post hoc.",
            "", "Hard assignments collapse all five variables into one group by the principal checkpoint in both walks. This is reported as a method diagnostic and is not retuned after evaluation.",
            "", "SGN-C is a supervised complete-system comparator, not target-free representation evidence. No universal, multi-seed, significance, or trading claim follows.",
        ]
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "matched_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"valid": True, "principal_rows": len(principal), "report": str(destination)}, indent=2))
        return 0
    except Exception as exc:
        print(f"SGN-C matched reporting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
