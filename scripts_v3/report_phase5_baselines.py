#!/usr/bin/env python3
"""Aggregate the matched Phase 5 raw-baseline experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts_v3") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts_v3"))

from bootstrap_phase5_baselines import MATRIX_PATH
from data_processing.phase5_walks import CLASS_NAMES
from tasks.phase2_classification.metrics import probabilistic_classification_metrics
from training.phase5_absolute_price import relative_skill
from training.phase5_baselines import BASELINES, TASKS, validate_baseline_run
from training.phase5_downstream import regression_metrics
from training.phase5_encoder import write_json


OUTPUT = Path("experiments/phase5/baselines/reports/baseline_matrix_seed0")


def _rank_ic(score: np.ndarray, target: np.ndarray) -> float:
    if len(score) < 2 or np.ptp(score) == 0.0 or np.ptp(target) == 0.0:
        return float("nan")
    return float(np.corrcoef(rankdata(score), rankdata(target))[0, 1])


def _cross_sectional_ic(
    score: np.ndarray, target: np.ndarray, timestamp: np.ndarray
) -> dict[str, float | int]:
    values = []
    for value in np.unique(timestamp):
        mask = timestamp == value
        if int(mask.sum()) >= 5:
            ic = _rank_ic(score[mask], target[mask])
            if np.isfinite(ic):
                values.append(ic)
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "icir_unannualized": float("nan"), "positive_fraction": float("nan")}
    std = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": std,
        "icir_unannualized": float(array.mean() / std) if std > 0.0 else float("nan"),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def _rows(matrix: dict[str, object], baseline: str, task: str) -> list[dict[str, object]]:
    return [
        row for row in matrix["runs"]
        if row["baseline"] == baseline and row["task"] == task
    ]


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    validations = []
    pooled: dict[str, object] = {}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for baseline in BASELINES:
        pooled[baseline] = {}
        for task in TASKS:
            values: dict[str, list[np.ndarray]] = {}
            for row in _rows(matrix, baseline, task):
                source = Path(row["dataset_path"])
                root = Path(row["run_root"])
                validations.append(validate_baseline_run(source, root))
                with np.load(root / "e50" / "predictions.npz", allow_pickle=False) as saved:
                    required = {
                        "regression_h2": (
                            "prediction_delta", "target_delta", "condition_ids", "decision_date_ns"
                        ),
                        "classification_h2": (
                            "logits", "targets", "condition_ids", "decision_date_ns"
                        ),
                        "absolute_price_h8": (
                            "prediction_future_price", "target_future_price", "current_close",
                            "prediction_delta_h8", "target_delta_h8", "condition_ids",
                            "decision_date_ns", "last_hour_reversal",
                        ),
                    }[task]
                    for key in required:
                        values.setdefault(key, []).append(np.asarray(saved[key]))
            merged = {key: np.concatenate(parts) for key, parts in values.items()}
            if task == "regression_h2":
                result = regression_metrics(merged["prediction_delta"], merged["target_delta"])
                result["exact_zero_reference"] = regression_metrics(
                    np.zeros_like(merged["target_delta"]), merged["target_delta"]
                )
            elif task == "classification_h2":
                result, _ = probabilistic_classification_metrics(
                    merged["logits"], merged["targets"], CLASS_NAMES
                )
            else:
                price = regression_metrics(
                    merged["prediction_future_price"], merged["target_future_price"]
                )
                persistence = regression_metrics(
                    merged["current_close"], merged["target_future_price"]
                )
                movement = regression_metrics(
                    merged["prediction_delta_h8"], merged["target_delta_h8"]
                )
                result = {
                    "price": price,
                    "persistence_reference": persistence,
                    "persistence_relative_skill": relative_skill(price, persistence),
                    "implied_movement": movement,
                    "implied_movement_cross_sectional_rank_ic": _cross_sectional_ic(
                        merged["prediction_delta_h8"],
                        merged["target_delta_h8"],
                        merged["decision_date_ns"],
                    ),
                    "last_hour_reversal_cross_sectional_rank_ic": _cross_sectional_ic(
                        merged["last_hour_reversal"],
                        merged["target_delta_h8"],
                        merged["decision_date_ns"],
                    ),
                }
            pooled[baseline][task] = result
            np.savez_compressed(OUTPUT / f"{baseline}_{task}_pooled_predictions.npz", **merged)
    framework_h2 = json.loads(
        Path("experiments/phase5/reports/framework_downstream_seed0/summary.json").read_text(encoding="utf-8")
    )["pooled"]
    framework_price = json.loads(
        Path("experiments/phase5/downstream_addons/reports/absolute_price_h8_seed0/summary.json").read_text(encoding="utf-8")
    )["pooled"]
    summary = {
        "phase": 5,
        "scope": "two matched raw-sequence baselines, three tasks, two independent walks, seed 0",
        "principal_epoch": 50,
        "validation": validations,
        "baselines": pooled,
        "framework_references": {
            "regression_h2": framework_h2["regression"]["framework"],
            "classification_h2": framework_h2["classification"]["framework"],
            "absolute_price_h8": framework_price,
        },
    }
    write_json(OUTPUT / "summary.json", summary)
    lines = [
        "# Phase 5 Matched Raw-Sequence Baselines (Seed 0)",
        "",
        "Epoch 50 is the predeclared principal checkpoint. Both walks were fitted independently and pooled only after out-of-future predictions were generated.",
        "",
        "## Two-hour movement regression",
        "",
        "| Model | MAE | RMSE | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    framework_reg = framework_h2["regression"]["framework"]
    lines.append(f"| Five-branch framework | {framework_reg['mae']:.8f} | {framework_reg['rmse']:.8f} | {framework_reg['pearson']:.6f} | {framework_reg['spearman']:.6f} |")
    for baseline in BASELINES:
        row = pooled[baseline]["regression_h2"]
        lines.append(f"| {baseline} | {row['mae']:.8f} | {row['rmse']:.8f} | {row['pearson']:.6f} | {row['spearman']:.6f} |")
    lines.extend([
        "",
        "## Two-hour movement classification",
        "",
        "| Model | Macro-F1 | Balanced accuracy | Accuracy |",
        "|---|---:|---:|---:|",
    ])
    framework_cls = framework_h2["classification"]["framework"]
    lines.append(f"| Five-branch framework | {framework_cls['macro_f1']:.6f} | {framework_cls['balanced_accuracy']:.6f} | {framework_cls['accuracy']:.6f} |")
    for baseline in BASELINES:
        row = pooled[baseline]["classification_h2"]
        lines.append(f"| {baseline} | {row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} | {row['accuracy']:.6f} |")
    lines.extend([
        "",
        "## Eight-hour absolute future price",
        "",
        "| Model | Price MAE | Price Pearson | Implied-movement Spearman | Mean cross-sectional Rank IC |",
        "|---|---:|---:|---:|---:|",
    ])
    fp = framework_price
    lines.append(f"| Five-branch framework | {fp['price']['mae']:.8f} | {fp['price']['pearson']:.6f} | {fp['implied_movement']['spearman']:.6f} | {fp['implied_movement_cross_sectional_rank_ic']['mean']:.6f} |")
    for baseline in BASELINES:
        row = pooled[baseline]["absolute_price_h8"]
        lines.append(f"| {baseline} | {row['price']['mae']:.8f} | {row['price']['pearson']:.6f} | {row['implied_movement']['spearman']:.6f} | {row['implied_movement_cross_sectional_rank_ic']['mean']:.6f} |")
    reversal_ic = pooled[BASELINES[0]]["absolute_price_h8"]["last_hour_reversal_cross_sectional_rank_ic"]["mean"]
    lines.extend([
        "",
        f"The fixed causal last-hour reversal reference has pooled mean cross-sectional Rank IC `{reversal_ic:.6f}` on the same h8 rows.",
        "",
        "These comparisons use identical task rows and source preprocessing. No best-on-evaluation checkpoint selection is performed.",
        "",
    ])
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"complete": True, "report": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
