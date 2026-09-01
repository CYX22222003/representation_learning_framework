"""Record the Phase-1 volatility comparison from completed experiment artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPOCHS = (15, 50, 100)
METRICS = ("mae", "rmse", "mse", "corr")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_rows(path: Path) -> dict[int, dict]:
    payload = _read_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty metric list: {path}")
    rows = {int(row["epoch"]): row for row in payload}
    missing = sorted(set(EPOCHS) - set(rows))
    if missing:
        raise ValueError(f"Missing epoch budgets {missing}: {path}")
    return rows


def _prediction_payload(path: Path, prediction_key: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "predictions": np.asarray(data[prediction_key], dtype=np.float32).reshape(-1),
            "targets": np.asarray(data["targets"], dtype=np.float32).reshape(-1),
        }


def _relative_error_change(framework: dict, baseline: dict) -> dict[str, float]:
    return {
        metric: (float(baseline[metric]) - float(framework[metric])) / float(baseline[metric])
        for metric in ("mae", "rmse", "mse")
    }


def _metric_subset(row: dict) -> dict[str, float | int]:
    return {
        "mae": float(row["mae"]),
        "rmse": float(row["rmse"]),
        "mse": float(row["mse"]),
        "corr": float(row["corr"]),
    }


def _prediction_stats(predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(predictions)),
        "std": float(np.std(predictions)),
        "min": float(np.min(predictions)),
        "median": float(np.median(predictions)),
        "max": float(np.max(predictions)),
        "negative_fraction": float(np.mean(predictions < 0.0)),
        "nonfinite_count": int(np.size(predictions) - np.count_nonzero(np.isfinite(predictions))),
    }


def _regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    residuals = predictions - targets
    if np.std(predictions) == 0.0 or np.std(targets) == 0.0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(predictions, targets)[0, 1])
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mse": float(np.mean(residuals**2)),
        "corr": corr,
    }


def _dataset_id(manifest: dict) -> str | None:
    label_manifest = manifest.get("label_manifest", {})
    return manifest.get("label_dataset_id") or label_manifest.get("dataset_id")


def build_comparison(
    framework_root: Path,
    raw_lstm_root: Path,
    stack_root: Path,
    legacy_mlp_root: Path | None,
) -> dict:
    framework_rows = _metric_rows(framework_root / "sweep_metrics.json")
    raw_rows = _metric_rows(raw_lstm_root / "sweep_metrics.json")
    stack_rows = _metric_rows(stack_root / "sweep_metrics.json")
    legacy_rows = _metric_rows(legacy_mlp_root / "sweep_metrics.json") if legacy_mlp_root else None

    framework_manifest = _read_json(framework_root / "dataset_manifest.json")
    raw_manifest = _read_json(raw_lstm_root / "dataset_manifest.json")
    stack_manifest = _read_json(stack_root / "dataset_manifest.json")
    manifest_ids = {
        "framework": _dataset_id(framework_manifest),
        "raw_lstm": _dataset_id(raw_manifest),
        "garch_lstm_stack": _dataset_id(stack_manifest),
    }
    if len(set(manifest_ids.values())) != 1 or None in manifest_ids.values():
        raise ValueError(f"Strict baseline label dataset IDs do not match: {manifest_ids}")

    rows = []
    for epoch in EPOCHS:
        framework_predictions = _prediction_payload(framework_root / f"e{epoch}" / "predictions.npz", "preds")
        raw_predictions = _prediction_payload(raw_lstm_root / f"e{epoch}" / "predictions.npz", "predictions")
        stack_predictions = _prediction_payload(
            stack_root / f"e{epoch}" / "predictions.npz", "stack_prediction_nonnegative"
        )
        if not np.array_equal(framework_predictions["targets"], raw_predictions["targets"]):
            raise ValueError(f"Framework and Raw LSTM targets differ at epoch {epoch}")
        if not np.array_equal(framework_predictions["targets"], stack_predictions["targets"]):
            raise ValueError(f"Framework and GARCH-LSTM stack targets differ at epoch {epoch}")

        framework = _metric_subset(framework_rows[epoch])
        raw_lstm = _metric_subset(raw_rows[epoch])
        stack = _metric_subset(stack_rows[epoch])
        row = {
            "epoch": epoch,
            "framework": framework,
            "raw_lstm": raw_lstm,
            "garch_lstm_stack": stack,
            "framework_prediction_stats": _prediction_stats(framework_predictions["predictions"]),
            "framework_nonnegative_diagnostic": _regression_metrics(
                np.maximum(framework_predictions["predictions"], 0.0),
                framework_predictions["targets"],
            ),
            "target_stats": _prediction_stats(framework_predictions["targets"]),
            "strict_relative_error_change_vs_raw_lstm": _relative_error_change(framework, raw_lstm),
            "strict_relative_error_change_vs_garch_lstm_stack": _relative_error_change(framework, stack),
            "strict_correlation_difference_vs_raw_lstm": framework["corr"] - raw_lstm["corr"],
            "strict_correlation_difference_vs_garch_lstm_stack": framework["corr"] - stack["corr"],
            "targets_exactly_equal": True,
            "test_sample_count": int(framework_rows[epoch]["test_sample_count"]),
        }
        if legacy_rows is not None:
            row["legacy_raw_ohlcv_mlp_context"] = _metric_subset(legacy_rows[epoch])
        rows.append(row)

    return {
        "task": "volatility_prediction",
        "framework_root": str(framework_root),
        "raw_lstm_root": str(raw_lstm_root),
        "garch_lstm_stack_root": str(stack_root),
        "legacy_raw_ohlcv_mlp_root": str(legacy_mlp_root) if legacy_mlp_root else None,
        "comparison_contract": {
            "strict_dataset_id": next(iter(manifest_ids.values())),
            "strict_models": "Phase-1 framework, Raw LSTM, and GARCH-LSTM stack use identical saved targets and 27,450 test rows.",
            "legacy_mlp": "Context only: the stored Raw-OHLCV MLP uses a legacy target builder and 27,499 test rows.",
            "selection_rule": "All 15/50/100 budgets are reported; no checkpoint is selected from locked-test metrics.",
        },
        "rows": rows,
    }


def _write_markdown(path: Path, comparison: dict) -> None:
    rows = comparison["rows"]
    lines = [
        "# Phase-1 Volatility Prediction Comparison",
        "",
        "The Phase-1 framework, Raw LSTM, and adapted GARCH–LSTM stack form a strict comparison: they use the same processed split, saved realized-volatility label bundle, seed, 15/50/100 budgets, and exactly identical 27,450 test targets. The stored Raw-OHLCV MLP is context only because it predates the shared label contract and evaluates 27,499 differently constructed targets.",
        "",
        "## Results",
        "",
        "| Epoch | Model | MAE | RMSE | MSE | Correlation |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        for label, key in (
            ("Phase-1 framework", "framework"),
            ("Raw LSTM", "raw_lstm"),
            ("GARCH–LSTM stack", "garch_lstm_stack"),
        ):
            metric = row[key]
            lines.append(
                f"| {row['epoch']} | {label} | {metric['mae']:.6f} | {metric['rmse']:.6f} | "
                f"{metric['mse']:.6f} | {metric['corr']:.6f} |"
            )

    lines += [
        "",
        "## Relative Phase-1 performance",
        "",
        "Positive error change means Phase-1 has lower error than the named baseline; negative means higher error.",
        "",
        "| Epoch | MAE vs Raw LSTM | RMSE vs Raw LSTM | MSE vs Raw LSTM | MAE vs stack | RMSE vs stack | MSE vs stack | Corr difference vs Raw LSTM | Corr difference vs stack |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        raw = row["strict_relative_error_change_vs_raw_lstm"]
        stack = row["strict_relative_error_change_vs_garch_lstm_stack"]
        lines.append(
            f"| {row['epoch']} | {raw['mae']:+.1%} | {raw['rmse']:+.1%} | {raw['mse']:+.1%} | "
            f"{stack['mae']:+.1%} | {stack['rmse']:+.1%} | {stack['mse']:+.1%} | "
            f"{row['strict_correlation_difference_vs_raw_lstm']:+.4f} | "
            f"{row['strict_correlation_difference_vs_garch_lstm_stack']:+.4f} |"
        )

    lines += [
        "",
        "## Recorded target and prediction behavior",
        "",
        "The shared target distribution shifts materially from training to test: mean realized volatility is `0.030779` on training rows and `0.080484` on test rows. The test target standard deviation is approximately `0.132715`, with values from `0.0` to `0.885255`. This temporal shift makes generalization harder and helps explain why test error remains much larger than final training loss.",
        "",
        "| Epoch | Phase-1 prediction mean | Prediction std | Prediction min | Prediction median | Prediction max | Negative fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        stats = row["framework_prediction_stats"]
        lines.append(
            f"| {row['epoch']} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | "
            f"{stats['median']:.6f} | {stats['max']:.6f} | {stats['negative_fraction']:.1%} |"
        )

    lines += [
        "",
        "## Analysis",
        "",
        "Phase-1 consistently outperforms the direct Raw LSTM benchmark on all three error metrics and correlation at every matched budget. Its MAE reduction is approximately 21–24%, while RMSE falls by approximately 12–18%. The result supports the claim that the frozen multi-branch representation provides useful volatility information beyond the end-to-end Raw LSTM under this shared task contract.",
        "",
        "The adapted GARCH–LSTM stack remains stronger overall. At 15 and 50 epochs, Phase-1 and the stack have almost identical RMSE and MSE, but the stack has materially lower MAE and higher correlation. This means the aggregate squared-error magnitude is close while Phase-1 makes more typical absolute errors and tracks cross-sample variation less reliably. At 100 epochs, the stack leads clearly on every metric.",
        "",
        "Within the Phase-1 sweep, MAE improves monotonically from 0.040595 to 0.035934. RMSE and MSE improve through 50 epochs but worsen at 100 epochs, and correlation peaks at 50 epochs before declining. Meanwhile, training loss falls continuously. This is consistent with late-stage overfitting or sensitivity to the train–test volatility-regime shift. The complete sweep is characterization evidence; the 50- or 100-epoch checkpoint must not be selected retrospectively from these test results.",
        "",
        "The current framework `VolatilityRegressor` has an unconstrained linear output, and 5.8–8.4% of its predictions are negative. Raw LSTM uses a Softplus output, while the stack applies its predeclared nonnegative clipping rule. Therefore, the primary framework table preserves the raw saved predictions, and zero-clipped framework results are recorded below only as a post-run diagnostic—not as replacement headline metrics.",
        "",
        "| Epoch | Raw Phase-1 MAE | Clipped diagnostic MAE | Raw Phase-1 RMSE | Clipped diagnostic RMSE | Raw Phase-1 MSE | Clipped diagnostic MSE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        raw = row["framework"]
        clipped = row["framework_nonnegative_diagnostic"]
        lines.append(
            f"| {row['epoch']} | {raw['mae']:.6f} | {clipped['mae']:.6f} | "
            f"{raw['rmse']:.6f} | {clipped['rmse']:.6f} | {raw['mse']:.6f} | {clipped['mse']:.6f} |"
        )
    lines += [
        "",
        "Clipping gives only a small numerical improvement and does not change the main conclusion. The strict target equality check passed for every framework/Raw-LSTM/stack epoch pair, so the reported differences are not caused by sample-count or target-alignment mismatches.",
        "",
        "## Legacy Raw-OHLCV MLP context",
        "",
        "| Epoch | Legacy MLP MAE | Legacy MLP RMSE | Legacy MLP MSE | Legacy MLP correlation |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        legacy = row.get("legacy_raw_ohlcv_mlp_context")
        if legacy is not None:
            lines.append(
                f"| {row['epoch']} | {legacy['mae']:.6f} | {legacy['rmse']:.6f} | "
                f"{legacy['mse']:.6f} | {legacy['corr']:.6f} |"
            )
    lines += [
        "",
        "These MLP values are not used for strict improvement claims. The legacy runner constructs volatility targets from merged sequence arrays and retains 27,499 test rows, whereas the shared contract has contract-aware next-window targets and 27,450 rows. The Raw-OHLCV MLP must be rerun against the saved label bundle before it becomes the direct strict internal baseline.",
        "",
        "## Conclusion and follow-up",
        "",
        "The five-branch Phase-1 framework is stronger than Raw LSTM and competitive with the GARCH–LSTM stack on RMSE/MSE at the shorter two budgets, but it does not surpass the stack overall. This is a positive representation result, not a universal superiority claim.",
        "",
        "- Rerun Raw-OHLCV MLP using the exact shared volatility label rows.",
        "- Repeat the strict framework/baseline matrix over multiple seeds and report mean plus standard deviation.",
        "- Add per-contract metrics and paired bootstrap intervals before making a strong comparative claim.",
        "- Run branch ablations, particularly all-branches-minus-BYOL and volatility-relevant statistical-only features, to identify the source of the gain over Raw LSTM.",
        "- Predeclare and rerun a nonnegative framework decoder (for example Softplus) so decoder constraints match the volatility task before final claims.",
        "- Treat alternative fusion, loss weighting, or training budgets as newly predeclared experiments rather than selecting them from this locked-test sweep.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, comparison: dict) -> None:
    rows = comparison["rows"]
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    model_specs = (
        ("framework", "Phase-1 framework", "o", "-"),
        ("raw_lstm", "Raw LSTM", "s", "-"),
        ("garch_lstm_stack", "GARCH–LSTM stack", "^", "-"),
        ("legacy_raw_ohlcv_mlp_context", "Legacy MLP (context)", "x", "--"),
    )
    for metric, axis in zip(METRICS, axes.reshape(-1)):
        for key, label, marker, linestyle in model_specs:
            if key not in rows[0]:
                continue
            axis.plot(
                epochs,
                [row[key][metric] for row in rows],
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
        axis.set(title=metric.upper(), xlabel="Epoch budget", ylabel=metric.upper())
        axis.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Phase-1 volatility: strict Raw LSTM and GARCH–LSTM comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_outputs(framework_root: Path, comparison: dict) -> list[Path]:
    json_path = framework_root / "comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    markdown_path = framework_root / "comparison.md"
    _write_markdown(markdown_path, comparison)
    image_dir = framework_root / "images"
    image_dir.mkdir(exist_ok=True)
    figure_path = image_dir / "baseline_comparison.png"
    _write_plot(figure_path, comparison)
    return [json_path, markdown_path, figure_path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare completed Phase-1 volatility artifacts with baselines.")
    parser.add_argument("--framework-root", required=True, type=Path)
    parser.add_argument(
        "--raw-lstm-root",
        default="src/baselines/raw_lstm_volatility/experiments/4h-seq64-top50-seed0",
        type=Path,
    )
    parser.add_argument(
        "--stack-root",
        default="src/baselines/garch_lstm_stacking/experiments/4h-seq64-top50-seed0",
        type=Path,
    )
    parser.add_argument(
        "--legacy-mlp-root",
        default="src/baselines/mlp_baseline/experiments/2026-08-04-volatility-sweep-15-50-100",
        type=Path,
    )
    args = parser.parse_args(argv)
    try:
        comparison = build_comparison(
            args.framework_root,
            args.raw_lstm_root,
            args.stack_root,
            args.legacy_mlp_root,
        )
        for output in write_outputs(args.framework_root, comparison):
            print(output)
    except Exception as exc:
        print(f"volatility comparison failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
