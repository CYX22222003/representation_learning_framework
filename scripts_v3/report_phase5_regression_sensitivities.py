#!/usr/bin/env python3
"""Aggregate the Phase 5 eight-hour and log-return regression sensitivities."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phase5")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts_v3") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts_v3"))

from launch_phase5_regression_sensitivities import paths
from training.phase5_downstream import regression_metrics
from training.phase5_encoder import write_json
from training.phase5_regression_sensitivities import validate_sensitivity_run


def main() -> None:
    output = Path("experiments/phase5/reports/regression_sensitivities_seed0")
    output.mkdir(parents=True, exist_ok=True)
    validations = []
    per_walk: dict[str, object] = {}
    pooled: dict[str, object] = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, task in zip(axes, ("raw_delta_h8", "log_return_h2")):
        predictions = []
        targets = []
        reconstructed = []
        future = []
        walks = []
        condition_ids = []
        per_walk[task] = {}
        for walk in (1, 2):
            dataset, feature, scaler, run_root = paths(task, walk)
            validations.append(validate_sensitivity_run(dataset, feature, scaler, run_root))
            metrics = json.loads((run_root / "e50" / "metrics.json").read_text(encoding="utf-8"))
            per_walk[task][f"walk{walk}"] = metrics
            with np.load(run_root / "e50" / "predictions.npz", allow_pickle=False) as values:
                prediction = np.asarray(values["prediction_scientific_target"])
                target = np.asarray(values["target_scientific"])
                predictions.append(prediction)
                targets.append(target)
                reconstructed.append(np.asarray(values["prediction_future_probability"]))
                future.append(np.asarray(values["target_close"]))
                walks.append(np.full(len(target), walk, dtype=np.int8))
                condition_ids.append(np.asarray(values["condition_ids"]))
            with np.load(run_root / "e50" / "history.npz", allow_pickle=False) as history:
                axis.plot(history["epochs"], history["train_loss"], label=f"Walk {walk}")
        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        probability_prediction = np.concatenate(reconstructed)
        probability_target = np.concatenate(future)
        pooled[task] = {
            "framework": regression_metrics(prediction, target),
            "zero_change_reference": regression_metrics(np.zeros_like(target), target),
            "reconstructed_probability": regression_metrics(
                probability_prediction, probability_target
            ),
        }
        np.savez_compressed(
            output / f"pooled_{task}_predictions.npz",
            prediction=prediction,
            target=target,
            prediction_future_probability=probability_prediction,
            target_close=probability_target,
            walk=np.concatenate(walks),
            condition_ids=np.concatenate(condition_ids),
        )
        axis.set_title(task.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Optimization-unit MSE")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "training_losses.png", dpi=180)
    plt.close(fig)

    primary = json.loads(
        Path("experiments/phase5/reports/framework_downstream_seed0/summary.json").read_text(
            encoding="utf-8"
        )
    )["pooled"]["regression"]
    summary = {
        "phase": 5,
        "scope": "exploratory seed-0 regression sensitivities",
        "principal_epoch": 50,
        "primary_raw_delta_h2": primary,
        "per_walk": per_walk,
        "pooled": pooled,
        "validation": validations,
    }
    write_json(output / "summary.json", summary)
    h8 = pooled["raw_delta_h8"]["framework"]
    h8zero = pooled["raw_delta_h8"]["zero_change_reference"]
    log = pooled["log_return_h2"]["framework"]
    logzero = pooled["log_return_h2"]["zero_change_reference"]
    logprob = pooled["log_return_h2"]["reconstructed_probability"]
    primary_framework = primary["framework"]
    lines = [
        "# Phase 5 Exploratory Regression Sensitivities (Seed 0)",
        "",
        "These tasks were frozen after the primary two-hour result and are interpreted as secondary sensitivities, not replacement targets selected from evaluation performance.",
        "",
        "| Task | Walk | MAE | RMSE | Pearson | Spearman | Sign agreement | Reconstructed probability MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in ("raw_delta_h8", "log_return_h2"):
        for walk in (1, 2):
            row = per_walk[task][f"walk{walk}"]["framework"]["overall"]
            reconstruction = per_walk[task][f"walk{walk}"]["reconstructed_probability"]
            lines.append(
                f"| {task} | {walk} | {row['mae']:.8f} | {row['rmse']:.8f} | "
                f"{row['pearson']:.6f} | {row['spearman']:.6f} | {row['sign_agreement']:.6f} | "
                f"{reconstruction['mae']:.8f} |"
            )
    lines.extend(
        [
            "",
            "## Pooled out-of-future observations",
            "",
            f"- Eight-hour raw change: MAE `{h8['mae']:.8f}`, RMSE `{h8['rmse']:.8f}`, Pearson `{h8['pearson']:.6f}`, Spearman `{h8['spearman']:.6f}`, sign agreement `{h8['sign_agreement']:.6f}`.",
            f"- Eight-hour zero-change reference: MAE `{h8zero['mae']:.8f}`, RMSE `{h8zero['rmse']:.8f}`.",
            f"- Two-hour log return: MAE `{log['mae']:.8f}`, RMSE `{log['rmse']:.8f}`, Pearson `{log['pearson']:.6f}`, Spearman `{log['spearman']:.6f}`, sign agreement `{log['sign_agreement']:.6f}`.",
            f"- Log-return zero-change reference: MAE `{logzero['mae']:.8f}`, RMSE `{logzero['rmse']:.8f}`.",
            f"- Log-return reconstructed probability: MAE `{logprob['mae']:.8f}`, RMSE `{logprob['rmse']:.8f}`.",
            "",
            "## Judgement",
            "",
            f"The primary two-hour raw-change Pearson/Spearman were `{primary_framework['pearson']:.6f}/{primary_framework['spearman']:.6f}`. Extending the horizon to eight hours reduced these to `{h8['pearson']:.6f}/{h8['spearman']:.6f}`, with the walk-specific Pearson signs disagreeing. The longer horizon therefore did not expose a stable signed relationship.",
            "",
            f"Ordinary log return also has approximately zero Pearson/Spearman (`{log['pearson']:.6f}/{log['spearman']:.6f}`). Its reconstructed probability MAE `{logprob['mae']:.8f}` is effectively unchanged from the primary model's `{primary_framework['mae']:.8f}`, while reconstructed RMSE is higher. The proportional target changes weighting toward low starting prices but does not recover direction or ordered magnitude.",
            "",
            "Together, the sensitivities do not support horizon length or additive target units as the main explanation for the regression limitation. A learned raw temporal comparator is still needed to separate representation loss from intrinsic short-history unpredictability.",
            "",
            "The existing two-hour raw-change result remains the primary Phase 5 regression experiment.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"complete": True, "report": str(output), "pooled": pooled}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
