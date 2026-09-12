from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tasks.phase2_classification.data import load_framework_inputs, load_labels_for_rows
from tasks.phase2_classification.models import FrameworkMovementClassifier
from tasks.phase2_classification.runner import RunConfig, parse_budgets, prepare_run_directory, run_classification_experiment


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train the isolated Phase 2 framework movement classifier.")
    result.add_argument("--features-npz", default="data/features/features_4h_seq64_top50_phase1.npz")
    result.add_argument("--labels-npz", required=True)
    result.add_argument("--alignment-npz", default=None)
    result.add_argument("--branches", default=None, help="Comma-separated branch names; default uses every saved branch.")
    result.add_argument("--mode", choices=("concat", "gated"), default="concat")
    result.add_argument("--out-dim", type=int, default=128)
    result.add_argument("--head-hidden-dim", type=int, default=128)
    result.add_argument("--model-id", default="C2_framework")
    result.add_argument("--protocol", choices=("P0", "P1U", "P1O", "P2"), default="P2")
    result.add_argument("--epoch-budgets", default="15,50,100")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=512)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--logit-adjustment-strength", type=float, default=1.0)
    result.add_argument("--device", default="auto")
    result.add_argument("--run-root", required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        labels, provenance = load_labels_for_rows(args.labels_npz, args.alignment_npz)
        X_train, X_test, branch_dims, scaler = load_framework_inputs(args.features_npz, labels, args.branches)
        config = RunConfig(
            model_id=args.model_id, protocol_id=args.protocol, epoch_budgets=parse_budgets(args.epoch_budgets),
            seed=args.seed, batch_size=args.batch_size, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, logit_adjustment_strength=args.logit_adjustment_strength,
            device=args.device,
        )
        run_root = prepare_run_directory(args.run_root, args.overwrite)
        model_spec = {
            "architecture": "RepresentationAggregator+TrendClassifier",
            "branch_dims": branch_dims,
            "mode": args.mode,
            "out_dim": args.out_dim,
            "head_hidden_dim": args.head_hidden_dim,
        }
        run_classification_experiment(
            X_train=X_train, y_train=labels["train_labels"], X_test=X_test, y_test=labels["test_labels"],
            identities=labels,
            model_factory=lambda: FrameworkMovementClassifier(
                branch_dims, mode=args.mode, out_dim=args.out_dim, head_hidden_dim=args.head_hidden_dim
            ),
            model_spec=model_spec, run_root=run_root, config=config,
            dataset_manifest={"features_npz": args.features_npz, "labels_npz": args.labels_npz, **provenance},
            scaler=scaler,
        )
        print(f"Phase 2 framework run completed: {run_root}")
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Phase 2 framework run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
