from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tasks.phase2_classification.data import load_labels_for_rows, load_raw_inputs
from tasks.phase2_classification.models import RawMovementMLP
from tasks.phase2_classification.runner import RunConfig, parse_budgets, prepare_run_directory, run_classification_experiment


def _dims(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise ValueError("hidden dims must be positive")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train the isolated Phase 2 Raw-OHLCV MLP classifier.")
    result.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    result.add_argument("--labels-npz", required=True)
    result.add_argument("--alignment-npz", default=None)
    result.add_argument("--hidden-dims", default="512,512,256,256,128")
    result.add_argument("--encoder-output-dim", type=int, default=128)
    result.add_argument("--head-hidden-dim", type=int, default=128)
    result.add_argument("--dropout", type=float, default=0.1)
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
        X_train, X_test = load_raw_inputs(args.processed_npz, labels)
        hidden_dims = _dims(args.hidden_dims)
        config = RunConfig(
            model_id="C1_raw_ohlcv_mlp", protocol_id=args.protocol,
            epoch_budgets=parse_budgets(args.epoch_budgets), seed=args.seed,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, logit_adjustment_strength=args.logit_adjustment_strength,
            device=args.device,
        )
        run_root = prepare_run_directory(args.run_root, args.overwrite)
        model_spec = {
            "architecture": "RawOHLCVMLP+TrendClassifier", "seq_len": int(X_train.shape[1]),
            "n_features": int(X_train.shape[2]), "hidden_dims": hidden_dims,
            "encoder_output_dim": args.encoder_output_dim, "head_hidden_dim": args.head_hidden_dim,
            "dropout": args.dropout,
        }
        run_classification_experiment(
            X_train=X_train, y_train=labels["train_labels"], X_test=X_test, y_test=labels["test_labels"],
            identities=labels,
            model_factory=lambda: RawMovementMLP(
                X_train.shape[1], X_train.shape[2], hidden_dims=hidden_dims,
                encoder_output_dim=args.encoder_output_dim, head_hidden_dim=args.head_hidden_dim,
                dropout=args.dropout,
            ),
            model_spec=model_spec, run_root=run_root, config=config,
            dataset_manifest={"processed_npz": args.processed_npz, "labels_npz": args.labels_npz, **provenance},
        )
        print(f"Phase 2 Raw-OHLCV MLP run completed: {run_root}")
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Phase 2 Raw-OHLCV MLP run failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
