#!/usr/bin/env python3
"""Aggregate and interpret the Phase 5 eight-hour absolute-price probe."""

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
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts_v3") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts_v3"))

from launch_phase5_absolute_price_h8 import paths
from training.phase5_absolute_price import relative_skill, validate_absolute_price_run
from training.phase5_downstream import regression_metrics
from training.phase5_encoder import write_json


def rank_ic(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(prediction) < 2 or np.ptp(prediction) == 0.0 or np.ptp(target) == 0.0:
        return float("nan")
    return float(np.corrcoef(rankdata(prediction), rankdata(target))[0, 1])


def cross_sectional_rank_ic(
    score: np.ndarray, target: np.ndarray, timestamps: np.ndarray
) -> tuple[dict[str, float | int], np.ndarray]:
    values = []
    for timestamp in np.unique(timestamps):
        mask = timestamps == timestamp
        if int(mask.sum()) < 5:
            continue
        value = rank_ic(score[mask], target[mask])
        if np.isfinite(value):
            values.append(value)
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1))
    return (
        {
            "count": int(len(array)),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "std": std,
            "icir_unannualized": float(array.mean() / std),
            "positive_fraction": float(np.mean(array > 0.0)),
        },
        array,
    )


def main() -> None:
    output = Path("experiments/phase5/reports/absolute_price_h8_seed0")
    output.mkdir(parents=True, exist_ok=True)
    validations = []
    per_walk: dict[str, object] = {}
    price_predictions = []
    targets = []
    current_values = []
    predicted_deltas = []
    target_deltas = []
    walks = []
    condition_ids = []
    timestamp_ics = []
    reversal_timestamp_ics = []
    fig, axis = plt.subplots(figsize=(7, 4))
    for walk in (1, 2):
        dataset, feature, scaler, run_root = paths(walk)
        validations.append(validate_absolute_price_run(dataset, feature, scaler, run_root))
        metrics = json.loads((run_root / "e50" / "metrics.json").read_text(encoding="utf-8"))
        per_walk[f"walk{walk}"] = metrics
        with np.load(run_root / "e50" / "predictions.npz", allow_pickle=False) as values:
            prediction = np.asarray(values["prediction_future_price"])
            target = np.asarray(values["target_future_price"])
            current = np.asarray(values["current_close"])
            predicted_delta = np.asarray(values["prediction_delta_h8"])
            target_delta = np.asarray(values["target_delta_h8"])
            timestamps = np.asarray(values["decision_date_ns"])
            model_ic, model_ic_values = cross_sectional_rank_ic(
                predicted_delta, target_delta, timestamps
            )
            with np.load(dataset, allow_pickle=False) as source:
                raw_sequences = np.asarray(source["test_raw_sequences"])
            last_hour_reversal = -(
                raw_sequences[:, -1, 3].astype(np.float64)
                - raw_sequences[:, -2, 3].astype(np.float64)
            )
            reversal_ic, reversal_ic_values = cross_sectional_rank_ic(
                last_hour_reversal, target_delta, timestamps
            )
            nonzero = target_delta != 0.0
            metrics["implied_movement_cross_sectional_rank_ic"] = model_ic
            metrics["last_hour_reversal_cross_sectional_rank_ic"] = reversal_ic
            metrics["implied_movement_nonzero_sign_agreement"] = float(
                np.mean(np.sign(predicted_delta[nonzero]) == np.sign(target_delta[nonzero]))
            )
            timestamp_ics.extend(model_ic_values.tolist())
            reversal_timestamp_ics.extend(reversal_ic_values.tolist())
            price_predictions.append(prediction)
            targets.append(target)
            current_values.append(current)
            predicted_deltas.append(predicted_delta)
            target_deltas.append(target_delta)
            walks.append(np.full(len(target), walk, dtype=np.int8))
            condition_ids.append(np.asarray(values["condition_ids"]))
        with np.load(run_root / "e50" / "history.npz", allow_pickle=False) as history:
            axis.plot(history["epochs"], history["train_loss"], label=f"Walk {walk}")
    prediction = np.concatenate(price_predictions)
    target = np.concatenate(targets)
    current = np.concatenate(current_values)
    predicted_delta = np.concatenate(predicted_deltas)
    target_delta = np.concatenate(target_deltas)
    price = regression_metrics(prediction, target)
    persistence = regression_metrics(current, target)
    movement = regression_metrics(predicted_delta, target_delta)
    ic_array = np.asarray(timestamp_ics, dtype=np.float64)
    ic_std = float(ic_array.std(ddof=1))
    cross_sectional_ic = {
        "count": int(len(ic_array)),
        "mean": float(ic_array.mean()),
        "median": float(np.median(ic_array)),
        "std": ic_std,
        "icir_unannualized": float(ic_array.mean() / ic_std),
        "positive_fraction": float(np.mean(ic_array > 0.0)),
    }
    reversal_array = np.asarray(reversal_timestamp_ics, dtype=np.float64)
    reversal_std = float(reversal_array.std(ddof=1))
    reversal_ic = {
        "count": int(len(reversal_array)),
        "mean": float(reversal_array.mean()),
        "median": float(np.median(reversal_array)),
        "std": reversal_std,
        "icir_unannualized": float(reversal_array.mean() / reversal_std),
        "positive_fraction": float(np.mean(reversal_array > 0.0)),
    }
    pooled = {
        "price": price,
        "persistence_reference": persistence,
        "persistence_relative_skill": relative_skill(price, persistence),
        "implied_movement": movement,
        "implied_movement_cross_sectional_rank_ic": cross_sectional_ic,
        "last_hour_reversal_cross_sectional_rank_ic": reversal_ic,
    }
    np.savez_compressed(
        output / "pooled_predictions.npz",
        prediction_future_price=prediction,
        target_future_price=target,
        current_close=current,
        prediction_delta_h8=predicted_delta,
        target_delta_h8=target_delta,
        walk=np.concatenate(walks),
        condition_ids=np.concatenate(condition_ids),
    )
    write_json(
        output / "summary.json",
        {
            "phase": 5,
            "task": "absolute_price_h8",
            "principal_epoch": 50,
            "validation": validations,
            "per_walk": per_walk,
            "pooled": pooled,
        },
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training MSE")
    axis.set_title("Eight-Hour Absolute Future Price")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "training_loss.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Phase 5 Eight-Hour Absolute-Price Probe (Seed 0)",
        "",
        "| Walk | Price MAE | Price RMSE | Price Pearson | Price Spearman | Persistence MAE | Implied-delta Pearson | Implied-delta Spearman |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for walk in (1, 2):
        row = per_walk[f"walk{walk}"]
        lines.append(
            f"| {walk} | {row['price']['overall']['mae']:.8f} | "
            f"{row['price']['overall']['rmse']:.8f} | {row['price']['overall']['pearson']:.6f} | "
            f"{row['price']['overall']['spearman']:.6f} | "
            f"{row['persistence_reference']['overall']['mae']:.8f} | "
            f"{row['implied_movement']['overall']['pearson']:.6f} | "
            f"{row['implied_movement']['overall']['spearman']:.6f} |"
        )
    lines.extend(["", "## Walk-level movement ranking", ""])
    for walk in (1, 2):
        row = per_walk[f"walk{walk}"]
        lines.append(
            f"- Walk {walk}: implied-movement mean cross-sectional Rank IC "
            f"`{row['implied_movement_cross_sectional_rank_ic']['mean']:.6f}`; "
            f"last-hour reversal reference `{row['last_hour_reversal_cross_sectional_rank_ic']['mean']:.6f}`; "
            f"non-zero sign agreement `{row['implied_movement_nonzero_sign_agreement']:.6f}`."
        )
    lines.extend(
        [
            "",
            "## Pooled result",
            "",
            f"- Price level: MAE `{price['mae']:.8f}`, RMSE `{price['rmse']:.8f}`, Pearson `{price['pearson']:.6f}`, Spearman `{price['spearman']:.6f}`.",
            f"- Persistence: MAE `{persistence['mae']:.8f}`, RMSE `{persistence['rmse']:.8f}`.",
            f"- Persistence-relative MSE skill: `{pooled['persistence_relative_skill']['mse']:.6f}`.",
            f"- Implied movement: Pearson `{movement['pearson']:.6f}`, Spearman `{movement['spearman']:.6f}`, sign agreement `{movement['sign_agreement']:.6f}`.",
            f"- Cross-sectional implied-movement Rank IC: mean `{cross_sectional_ic['mean']:.6f}`, median `{cross_sectional_ic['median']:.6f}`, positive fraction `{cross_sectional_ic['positive_fraction']:.4f}` across `{cross_sectional_ic['count']}` timestamps.",
            f"- Last-hour reversal reference cross-sectional Rank IC: mean `{reversal_ic['mean']:.6f}`, median `{reversal_ic['median']:.6f}`, positive fraction `{reversal_ic['positive_fraction']:.4f}`.",
            "",
            "Direct price training recovers a consistent implied-movement ranking signal, but the simple last-hour reversal reference is materially stronger. High price-level correlation remains state reconstruction rather than evidence of superior forecasting.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"complete": True, "report": str(output), "pooled": pooled}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
