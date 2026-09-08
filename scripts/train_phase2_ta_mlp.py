from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tasks.phase2_classification.data import load_labels_for_rows, load_ta_inputs
from tasks.phase2_classification.models import make_ta_mlp
from tasks.phase2_classification.runner import RunConfig, parse_budgets, prepare_run_directory, run_classification_experiment


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train the isolated, bundle-aligned Phase 2 TA-MLP.")
    result.add_argument("--labels-npz", required=True)
    result.add_argument("--ta-features-npz", required=True)
    result.add_argument("--protocol", choices=("P0", "P1U", "P1O", "P2"), default="P2")
    result.add_argument("--epoch-budgets", default="15,50,100")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--logit-adjustment-strength", type=float, default=1.0)
    result.add_argument("--device", default="auto")
    result.add_argument("--run-root", required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        labels, provenance = load_labels_for_rows(args.labels_npz, args.ta_features_npz)
        X_train, X_test, scaler = load_ta_inputs(args.ta_features_npz, labels)
        config = RunConfig(
            model_id="C5_ta_mlp", protocol_id=args.protocol,
            epoch_budgets=parse_budgets(args.epoch_budgets), seed=args.seed,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, logit_adjustment_strength=args.logit_adjustment_strength,
            device=args.device,
        )
        run_root = prepare_run_directory(args.run_root, args.overwrite)
        run_classification_experiment(
            X_train=X_train, y_train=labels["train_labels"], X_test=X_test, y_test=labels["test_labels"],
            identities=labels, model_factory=lambda: make_ta_mlp(X_train.shape[1]),
            model_spec={"architecture": "TAMLPClassifier", "input_dim": int(X_train.shape[1]), "n_classes": 3},
            run_root=run_root, config=config,
            dataset_manifest={"ta_features_npz": args.ta_features_npz, "labels_npz": args.labels_npz, **provenance},
            scaler=scaler,
        )
        print(f"Phase 2 TA-MLP run completed: {run_root}")
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Phase 2 TA-MLP run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
