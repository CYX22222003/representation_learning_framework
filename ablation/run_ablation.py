"""Plan and execute the ablation study outside the Phase-1/Phase-2 trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_framework_experiment import plot_run
from train_framework import (
    FrameworkConfig,
    _prepare_run_root,
    build_price_data,
    build_trend_data,
    build_volatility_data,
    load_processed_npz,
    load_split_feature_branches,
    run_experiment,
)

from ablation.design import (
    CANONICAL_BRANCHES,
    CANONICAL_TASKS,
    build_variants,
    validate_available_branches,
)
from ablation.report import build_report


def _parse_csv(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def _task_labels(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "price_prediction": None,
        "volatility_prediction": args.volatility_labels_npz,
        "trend_classification": args.trend_labels_npz,
    }


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_task_alignment(
    args: argparse.Namespace,
    tasks: tuple[str, ...],
    train_branches: Mapping[str, object],
    test_branches: Mapping[str, object],
    feature_index: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    train_sequences, test_sequences = load_processed_npz(args.processed_npz)
    if (
        len(train_sequences) != feature_index["train_size"]
        or len(test_sequences) != feature_index["test_size"]
    ):
        raise ValueError("Feature-store split sizes do not match the processed dataset")
    counts: dict[str, dict[str, int]] = {}
    for task in tasks:
        config = FrameworkConfig(task=task, labels_npz=_task_labels(args)[task])
        if task == "price_prediction":
            _, y_train, _, y_test = build_price_data(
                train_sequences, test_sequences, train_branches, test_branches, config
            )
        elif task == "trend_classification":
            _, y_train, _, y_test, _ = build_trend_data(
                train_branches, test_branches, feature_index, config
            )
        else:
            _, y_train, _, y_test, _ = build_volatility_data(
                train_branches, test_branches, feature_index, config
            )
        counts[task] = {"train": len(y_train), "test": len(y_test)}
    return counts


def create_plan(args: argparse.Namespace) -> dict[str, object]:
    tasks = _parse_csv(args.tasks)
    unknown_tasks = set(tasks).difference(CANONICAL_TASKS)
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {sorted(unknown_tasks)}")
    families = _parse_csv(args.families)
    variants = build_variants(CANONICAL_BRANCHES, families, args.include_gated)
    epoch_budgets = tuple(int(value) for value in _parse_csv(args.epoch_budgets))
    seeds = tuple(int(value) for value in _parse_csv(args.seeds))
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    # Reuse the shared configuration validation without loading or training a model.
    for task in tasks:
        FrameworkConfig(
            task=task,
            labels_npz=_task_labels(args)[task],
            mode="concat",
            out_dim=args.out_dim,
            epoch_budgets=epoch_budgets,
            seed=seeds[0],
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            head_hidden_dim=args.head_hidden_dim,
            standardize_clip=args.standardize_clip,
            device=args.device,
        )
    train_branches, test_branches, feature_index = load_split_feature_branches(args.features_npz)
    branch_dims = {name: values.shape[1] for name, values in train_branches.items()}
    validate_available_branches(branch_dims)
    sample_counts = _validate_task_alignment(
        args, tasks, train_branches, test_branches, feature_index
    )
    primary_inputs = [
        args.processed_npz,
        args.features_npz,
        f"{args.features_npz}.index.npz",
    ]
    labels = _task_labels(args)
    primary_inputs.extend(labels[task] for task in tasks if labels[task])
    input_paths: list[str] = []
    for path in primary_inputs:
        input_paths.append(path)
        manifest_path = f"{path}.manifest.json"
        if Path(manifest_path).exists():
            input_paths.append(manifest_path)
    missing = [path for path in input_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing ablation inputs: {missing}")
    return {
        "schema_version": 1,
        "study_name": args.study_name,
        "workstream": "independent_ablation",
        "separation_contract": {
            "code_root": "ablation/",
            "artifact_root": f"ablation/experiments/{args.study_name}/",
            "phase1_artifacts_are_read_only_inputs": True,
            "writes_to_phase1_or_phase2": False,
        },
        "protocol": {
            "global_split": "existing chronological per-contract 80/20 train/test split",
            "validation_split": None,
            "early_stopping": False,
            "test_driven_selection": False,
            "report_all_budgets": True,
        },
        "processed_npz": args.processed_npz,
        "features_npz": args.features_npz,
        "labels": labels,
        "input_sha256": {path: _file_sha256(path) for path in input_paths},
        "feature_index": feature_index,
        "branch_dims": {branch: branch_dims[branch] for branch in CANONICAL_BRANCHES},
        "tasks": list(tasks),
        "sample_counts": sample_counts,
        "variants": [variant.to_dict() for variant in variants],
        "epoch_budgets": list(epoch_budgets),
        "seeds": list(seeds),
        "training": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "head_hidden_dim": args.head_hidden_dim,
            "gated_out_dim": args.out_dim,
            "standardize": True,
            "standardize_clip": args.standardize_clip,
        },
        "status": "planned",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_or_verify_plan(study_root: Path, plan: dict[str, object]) -> dict[str, object]:
    path = study_root / "plan.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {k: v for k, v in plan.items() if k not in {"created_at_utc", "status"}}
        existing_comparable = {k: v for k, v in existing.items() if k not in {"created_at_utc", "status"}}
        if comparable != existing_comparable:
            raise ValueError(
                "The study already has a different locked plan. Use a new --study-name; "
                "do not alter a matrix after test evaluation."
            )
        return existing
    study_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def _run_training(
    args: argparse.Namespace,
    study_root: Path,
    plan: dict[str, object],
) -> None:
    for seed in plan["seeds"]:
        for task in plan["tasks"]:
            for variant in plan["variants"]:
                run_root = study_root / "runs" / task / variant["name"] / f"seed_{seed}"
                _prepare_run_root(run_root, args.overwrite_runs)
                config = FrameworkConfig(
                    task=task,
                    labels_npz=plan["labels"][task],
                    mode=variant["mode"],
                    out_dim=plan["training"]["gated_out_dim"],
                    epoch_budgets=tuple(plan["epoch_budgets"]),
                    seed=seed,
                    batch_size=plan["training"]["batch_size"],
                    learning_rate=plan["training"]["learning_rate"],
                    weight_decay=plan["training"]["weight_decay"],
                    head_hidden_dim=plan["training"]["head_hidden_dim"],
                    standardize_clip=plan["training"]["standardize_clip"],
                    device=args.device,
                )
                run_experiment(
                    plan["processed_npz"], plan["features_npz"], run_root, config,
                    selected_branches=tuple(variant["branches"]),
                )


def _run_plots(study_root: Path, plan: dict[str, object]) -> None:
    for seed in plan["seeds"]:
        for task in plan["tasks"]:
            for variant in plan["variants"]:
                plot_run(study_root / "runs" / task / variant["name"] / f"seed_{seed}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run the independent ablation matrix.")
    parser.add_argument(
        "--stages",
        default="plan",
        help="Comma-separated: plan,train,plot,report",
    )
    parser.add_argument("--study-name", default="4h_all5_v1")
    parser.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    parser.add_argument("--features-npz", default="data/features/features_4h_seq64_top50_phase1.npz")
    parser.add_argument(
        "--trend-labels-npz",
        default="data/task_labels/trend_classification/triclass_4h_seq64_top50.npz",
    )
    parser.add_argument(
        "--volatility-labels-npz",
        default="data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz",
    )
    parser.add_argument("--tasks", default=",".join(CANONICAL_TASKS))
    parser.add_argument("--families", default="single,leave_one_out,full")
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--epoch-budgets", default="15,50,100")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--standardize-clip", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite-runs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        stages = set(_parse_csv(args.stages))
        unknown = stages.difference({"plan", "train", "plot", "report"})
        if unknown:
            raise ValueError(f"Unknown stages: {sorted(unknown)}")
        study_root = ROOT / "ablation" / "experiments" / args.study_name
        plan = _write_or_verify_plan(study_root, create_plan(args))
        if "train" in stages:
            _run_training(args, study_root, plan)
        if "plot" in stages:
            _run_plots(study_root, plan)
        if "report" in stages:
            # The first version supports one seed in the aggregate report. Multi-seed
            # inference is retained in per-seed artifacts pending CI aggregation.
            if len(plan["seeds"]) != 1:
                raise ValueError("aggregate reporting currently requires exactly one seed")
            seed = plan["seeds"][0]
            build_report(study_root, seed=seed)
        print(json.dumps({"study_root": str(study_root), "stages": sorted(stages)}, indent=2))
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ablation study failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
