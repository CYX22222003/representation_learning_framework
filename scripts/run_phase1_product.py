"""Run the predeclared Phase-1 Product feature and task experiment matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plot_framework_experiment import plot_run
from prepare_framework_features import build_and_save_framework_features
from train_framework import FrameworkConfig, _prepare_run_root, run_experiment
from validate_feature_store import validate_feature_store


def _parse_stages(text: str) -> set[str]:
    stages = {item.strip() for item in text.split(",") if item.strip()}
    allowed = {"features", "train", "plot"}
    unknown = stages.difference(allowed)
    if unknown or not stages:
        raise ValueError(f"stages must be a non-empty subset of {sorted(allowed)}")
    return stages


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_phase1(args: argparse.Namespace) -> dict:
    stages = _parse_stages(args.stages)
    feature_path = Path(args.features_npz)
    task_root = ROOT / "experiments" / "framework"
    record_path = ROOT / "experiments" / "framework" / "phase1" / args.run_name / "execution_manifest.json"
    if record_path.exists() and not args.overwrite:
        raise FileExistsError(f"{record_path} exists; pass --overwrite to replace this Phase-1 record")

    record: dict = {
        "phase": "Phase-1 Product",
        "processed_npz": args.processed_npz,
        "features_npz": str(feature_path),
        "run_name": args.run_name,
        "stages": sorted(stages),
        "seed": args.seed,
        "epoch_budgets": args.epoch_budgets,
        "mode": "concat",
        "tasks": {},
    }
    if "features" in stages:
        build_and_save_framework_features(
            processed_npz=Path(args.processed_npz), out_path=feature_path,
            vae_checkpoint=Path(args.vae_checkpoint), contrastive_checkpoint=Path(args.contrastive_checkpoint),
            byol_checkpoint=Path(args.byol_checkpoint), ar_order=5, fft_top_k=8, wavelet_levels=3,
            batch_size=args.feature_batch_size, device_name=args.device, progress_every=args.progress_every,
            overwrite=args.overwrite, base_features_npz=Path(args.base_features_npz) if args.base_features_npz else None,
        )
    if "features" in stages or "train" in stages:
        record["feature_validation"] = validate_feature_store(
            feature_path, Path(args.processed_npz),
            {"statistical": 70, "transformed": 55, "vae": 64, "contrastive": 128, "byol": 128}, True,
        )

    labels = {
        "price_prediction": None,
        "trend_classification": args.trend_labels_npz,
        "volatility_prediction": args.volatility_labels_npz,
    }
    if "train" in stages:
        for task, label_path in labels.items():
            run_root = task_root / task / args.run_name
            _prepare_run_root(run_root, args.overwrite)
            config = FrameworkConfig(
                task=task, labels_npz=label_path, mode="concat",
                epoch_budgets=tuple(int(value) for value in args.epoch_budgets.split(",")),
                seed=args.seed, batch_size=args.batch_size, device=args.device,
            )
            metrics = run_experiment(args.processed_npz, str(feature_path), run_root, config)
            record["tasks"][task] = {"run_root": str(run_root), "sweep_metrics": metrics}
    if "plot" in stages:
        for task in labels:
            run_root = task_root / task / args.run_name
            if not run_root.exists():
                raise FileNotFoundError(f"Cannot plot missing task run: {run_root}")
            record["tasks"].setdefault(task, {"run_root": str(run_root)})["plots"] = [str(path) for path in plot_run(run_root)]

    _write_json(record_path, record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute the fixed Phase-1 Product experiment matrix.")
    parser.add_argument("--stages", default="features,train,plot", help="Comma-separated: features,train,plot.")
    parser.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    parser.add_argument("--features-npz", default="data/features/features_4h_seq64_top50_phase1.npz")
    parser.add_argument(
        "--base-features-npz", default="data/features/features_4h_seq64_top50.npz",
        help="Validated four-branch source to reuse; pass an empty value only to recompute all deterministic branches.",
    )
    parser.add_argument("--vae-checkpoint", default="checkpoints/vae_4h_seq64_top50.pth")
    parser.add_argument("--contrastive-checkpoint", default="checkpoints/contrastive_4h_seq64_top50.pth")
    parser.add_argument("--byol-checkpoint", default="checkpoints/byol_4h_seq64_top50.pth")
    parser.add_argument("--trend-labels-npz", default="data/task_labels/trend_classification/triclass_4h_seq64_top50.npz")
    parser.add_argument("--volatility-labels-npz", default="data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz")
    parser.add_argument("--run-name", default="4h_phase1_all5_concat")
    parser.add_argument("--epoch-budgets", default="15,50,100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        record = run_phase1(args)
    except Exception as exc:
        print(f"Phase-1 Product run failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"phase": record["phase"], "tasks": list(record["tasks"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
