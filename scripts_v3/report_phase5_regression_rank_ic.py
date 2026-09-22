#!/usr/bin/env python3
"""Report global, cross-sectional, and contract-temporal Phase 5 Rank IC."""

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

from training.phase5_encoder import write_json


TASK_PATHS = {
    "raw_delta_h2": "experiments/phase5/downstream/walk{walk}/regression/seed0/e50/predictions.npz",
    "raw_delta_h8": "experiments/phase5/downstream_addons/tasks/raw_delta_h8/walk{walk}/seed0/e50/predictions.npz",
    "log_return_h2": "experiments/phase5/downstream_addons/tasks/log_return_h2/walk{walk}/seed0/e50/predictions.npz",
}


def rank_ic(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(target, dtype=np.float64)
    if len(pred) < 2 or np.ptp(pred) == 0.0 or np.ptp(actual) == 0.0:
        return float("nan")
    return float(np.corrcoef(rankdata(pred), rankdata(actual))[0, 1])


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    std = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": std,
        "icir_unannualized": float(array.mean() / std) if std > 0 else float("nan"),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def load_task(task: str, walk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(Path(TASK_PATHS[task].format(walk=walk)), allow_pickle=False) as stored:
        if task == "raw_delta_h2":
            prediction = np.asarray(stored["prediction_delta"])
            target = np.asarray(stored["target_delta"])
        else:
            prediction = np.asarray(stored["prediction_scientific_target"])
            target = np.asarray(stored["target_scientific"])
        return (
            prediction,
            target,
            np.asarray(stored["condition_ids"]),
            np.asarray(stored["decision_date_ns"]),
        )


def main() -> None:
    output = Path("experiments/phase5/downstream_addons/reports/regression_rank_ic_seed0")
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "phase": 5,
        "purpose": "regression_rank_ic_diagnostic",
        "principal_epoch": 50,
        "cross_sectional_minimum_contracts": 5,
        "overlap_warning": (
            "hourly stride-one IC observations overlap; ICIR is descriptive and unannualized"
        ),
        "tasks": {},
    }
    arrays: dict[str, np.ndarray] = {}
    for task in TASK_PATHS:
        task_rows: dict[str, object] = {}
        pooled_cross_sectional: list[float] = []
        for walk in (1, 2):
            prediction, target, condition_ids, timestamps = load_task(task, walk)
            cross_sectional = []
            cross_sectional_timestamps = []
            cross_sectional_sizes = []
            constant_target_skips = 0
            for timestamp in np.unique(timestamps):
                mask = timestamps == timestamp
                if int(mask.sum()) < 5:
                    continue
                value = rank_ic(prediction[mask], target[mask])
                if np.isfinite(value):
                    cross_sectional.append(value)
                    cross_sectional_timestamps.append(int(timestamp))
                    cross_sectional_sizes.append(int(mask.sum()))
                else:
                    constant_target_skips += 1
            temporal = []
            temporal_sizes = []
            for condition_id in np.unique(condition_ids):
                mask = condition_ids == condition_id
                value = rank_ic(prediction[mask], target[mask])
                if np.isfinite(value):
                    temporal.append(value)
                    temporal_sizes.append(int(mask.sum()))
            key = f"{task}_walk{walk}"
            arrays[f"{key}_cross_sectional_ic"] = np.asarray(cross_sectional, dtype=np.float64)
            arrays[f"{key}_timestamps_ns"] = np.asarray(cross_sectional_timestamps, dtype=np.int64)
            arrays[f"{key}_contract_counts"] = np.asarray(cross_sectional_sizes, dtype=np.int16)
            pooled_cross_sectional.extend(cross_sectional)
            task_rows[f"walk{walk}"] = {
                "row_count": int(len(target)),
                "global_rank_ic": rank_ic(prediction, target),
                "cross_sectional_rank_ic": {
                    **summarize(cross_sectional),
                    "median_contracts_per_timestamp": float(np.median(cross_sectional_sizes)),
                    "minimum_contracts": int(min(cross_sectional_sizes)),
                    "maximum_contracts": int(max(cross_sectional_sizes)),
                    "constant_target_timestamps_skipped": constant_target_skips,
                },
                "contract_temporal_rank_ic": {
                    **summarize(temporal),
                    "row_weighted_mean": float(np.average(temporal, weights=temporal_sizes)),
                },
            }
        task_rows["pooled_cross_sectional_rank_ic"] = summarize(pooled_cross_sectional)
        report["tasks"][task] = task_rows
    np.savez_compressed(output / "rank_ic_series.npz", **arrays)
    write_json(output / "summary.json", report)
    lines = [
        "# Phase 5 Regression Rank IC Diagnostic (Seed 0, Epoch 50)",
        "",
        "Global Rank IC is the Spearman correlation over all saved rows. Finance-style cross-sectional Rank IC is calculated independently at each decision timestamp with at least five active contracts, then summarized across timestamps.",
        "",
        "| Task | Walk | Global Rank IC | Mean cross-sectional Rank IC | Median IC | Unannualized ICIR | Positive IC fraction | IC timestamps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASK_PATHS:
        for walk in (1, 2):
            row = report["tasks"][task][f"walk{walk}"]
            ic = row["cross_sectional_rank_ic"]
            lines.append(
                f"| {task} | {walk} | {row['global_rank_ic']:.6f} | {ic['mean']:.6f} | "
                f"{ic['median']:.6f} | {ic['icir_unannualized']:.6f} | "
                f"{ic['positive_fraction']:.4f} | {ic['count']} |"
            )
    lines.extend(["", "## Pooled cross-sectional Rank IC", ""])
    for task in TASK_PATHS:
        row = report["tasks"][task]["pooled_cross_sectional_rank_ic"]
        lines.append(
            f"- `{task}`: mean `{row['mean']:.6f}`, median `{row['median']:.6f}`, "
            f"unannualized ICIR `{row['icir_unannualized']:.6f}`, positive fraction "
            f"`{row['positive_fraction']:.4f}`, across `{row['count']}` timestamps."
        )
    lines.extend(
        [
            "",
            "All mean ICs and ICIRs are approximately zero and positive fractions are approximately 50%. The current regression outputs therefore show no stable cross-sectional ranking power. Hourly stride-one rows and targets overlap, so no independent-observation significance claim is made.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"complete": True, "output": str(output), "tasks": report["tasks"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
