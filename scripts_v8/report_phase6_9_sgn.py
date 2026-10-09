#!/usr/bin/env python3
"""Generate the SGN-C-only snapshot report after both walks validate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.phase5_encoder import write_json  # noqa: E402
from training.phase6_9_sgn import SNAPSHOTS, phase_root, run_root  # noqa: E402


def main() -> int:
    try:
        rows = []
        for walk in (1, 2):
            for epoch in SNAPSHOTS:
                metrics = json.loads((run_root(ROOT, walk) / f"e{epoch}/metrics.json").read_text())
                primary = metrics["sgn_c"]
                rows.append({
                    "walk": walk, "epoch": epoch, "macro_f1": primary["macro_f1"],
                    "balanced_accuracy": primary["balanced_accuracy"], "accuracy": primary["accuracy"],
                    "predicted_class_counts": primary["predicted_class_counts"],
                    "hard_group_ids": metrics["hard_group_ids"],
                })
        destination = phase_root(ROOT) / "reports/seed0"
        write_json(destination / "summary.json", {"principal_epoch": 50, "rows": rows})
        lines = [
            "# SGN-C classification report", "",
            "Seed 0, two independent walks, fixed 5/15/50 snapshots; epoch 50 is primary.", "",
            "| Walk | Epoch | Macro-F1 | Balanced accuracy | Accuracy | Predicted D/S/U | Hard groups O/H/L/C/V |",
            "|---:|---:|---:|---:|---:|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['walk']} | {row['epoch']} | {row['macro_f1']:.6f} | "
                f"{row['balanced_accuracy']:.6f} | {row['accuracy']:.6f} | "
                f"{row['predicted_class_counts']} | {row['hard_group_ids']} |"
            )
        lines += ["", "SGN-C is a supervised complete-system comparator, not target-free representation evidence. No universal, multi-seed, significance, or trading claim follows."]
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"valid": True, "rows": len(rows), "report": str(destination)}, indent=2))
        return 0
    except Exception as exc:
        print(f"SGN-C reporting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
