"""Validate and summarize the completed Phase 5 encoder matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_phase5_encoder import dataset_path, default_run_root  # noqa: E402
from training.phase5_encoder import (  # noqa: E402
    ENCODERS,
    validate_encoder_run,
    write_encoder_summary,
    write_json,
)


def main() -> int:
    output = ROOT / "experiments" / "phase5" / "reports" / "encoder_pretraining_seed0"
    try:
        rows = []
        for walk in (1, 2):
            for encoder in ENCODERS:
                run_root = default_run_root(walk, encoder)
                # Completed runs created before summary support are upgraded
                # from their replayed structured artifacts.
                if not (run_root / "summary.md").exists():
                    write_encoder_summary(run_root)
                rows.append(validate_encoder_run(dataset_path(walk), run_root))
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "summary.json", {"runs": rows})
        lines = [
            "# Phase 5 Canonical Encoder Pretraining",
            "",
            "All six seed-0 walk-specific trajectories completed 50 epochs. Epoch 50 was predeclared for downstream feature extraction.",
            "",
            "| Walk | Encoder | Epoch-50 loss | Embedding std | Collapse warning |",
            "|---:|---|---:|---:|:---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['walk']} | {row['encoder']} | {row['epoch50_train_loss']:.8f} | "
                f"{row['epoch50_embedding_std']:.8f} | {str(row['collapse_warning']).lower()} |"
            )
        lines.extend(
            [
                "",
                "These are unsupervised training and representation-health diagnostics, not downstream performance results.",
                "",
            ]
        )
        (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Phase 5 encoder report written: {output}", flush=True)
        return 0
    except Exception as exc:
        print(f"Phase 5 encoder reporting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
