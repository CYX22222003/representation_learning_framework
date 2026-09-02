"""Record a fair Phase-1 price comparison with the existing baseline artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_json(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected a non-empty metric list: {path}")
    return value


def _lstm_row(summary_path: Path, epoch: int) -> dict:
    text = summary_path.read_text(encoding="utf-8")
    mae = re.search(r"\*\*Test MAE\*\* \| \*\*([0-9.]+)\*\*", text)
    rmse = re.search(r"\*\*Test RMSE\*\* \| \*\*([0-9.]+)\*\*", text)
    if not mae or not rmse:
        raise ValueError(f"Could not parse MAE/RMSE from {summary_path}")
    return {"epoch": epoch, "mae": float(mae.group(1)), "rmse": float(rmse.group(1))}


def build_comparison(framework_root: Path, mlp_root: Path, lstm_root: Path) -> dict:
    framework = _read_json(framework_root / "sweep_metrics.json")
    mlp = _read_json(mlp_root / "sweep_metrics.json")
    epochs = [15, 50, 100]
    lstm = [_lstm_row(lstm_root / f"2026-06-21-v5-e{epoch}" / "summary.md", epoch) for epoch in epochs]
    framework_by_epoch = {int(row["epoch"]): row for row in framework}
    mlp_by_epoch = {int(row["epoch"]): row for row in mlp}
    rows = []
    for epoch in epochs:
        f, m, l = framework_by_epoch[epoch], mlp_by_epoch[epoch], next(row for row in lstm if row["epoch"] == epoch)
        rows.append({
            "epoch": epoch,
            "framework": {key: f[key] for key in ("mae", "rmse", "mse", "corr", "test_sample_count")},
            "mlp": {key: m[key] for key in ("mae", "rmse", "mse", "corr", "test_sample_count")},
            "lstm_context": l,
            "strict_mlp_relative_error_change": {
                "mae": (m["mae"] - f["mae"]) / m["mae"],
                "rmse": (m["rmse"] - f["rmse"]) / m["rmse"],
                "mse": (m["mse"] - f["mse"]) / m["mse"],
            },
        })
    return {
        "task": "price_prediction",
        "framework_root": str(framework_root),
        "mlp_root": str(mlp_root),
        "lstm_root": str(lstm_root),
        "comparison_contract": {
            "mlp": "Strict: same processed npz, price index/horizon, seed 0, epoch budgets, and 27,499 test rows.",
            "lstm": "Context only: close-only external LSTM reports 27,500 test rows and has a documented one-row target alignment difference.",
        },
        "rows": rows,
    }


def write_outputs(framework_root: Path, comparison: dict) -> list[Path]:
    json_path = framework_root / "comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    lines = [
        "# Phase-1 Price Prediction Baseline Comparison",
        "",
        "The Raw-OHLCV MLP comparison is strict: both runs use the same processed split, price target builder, seed, budgets, and 27,499 test rows. The LSTM is external context only because its close-only artifact uses 27,500 rows and has a documented one-row target-alignment difference.",
        "",
        "| epoch | Phase-1 MAE | MLP MAE | Phase-1 MAE improvement | Phase-1 RMSE | MLP RMSE | Phase-1 RMSE improvement | LSTM MAE (context) | LSTM RMSE (context) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        f, m, l, change = row["framework"], row["mlp"], row["lstm_context"], row["strict_mlp_relative_error_change"]
        lines.append(
            f"| {row['epoch']} | {f['mae']:.6f} | {m['mae']:.6f} | {change['mae']:+.1%} | "
            f"{f['rmse']:.6f} | {m['rmse']:.6f} | {change['rmse']:+.1%} | {l['mae']:.6f} | {l['rmse']:.6f} |"
        )
    lines += ["", "Positive improvement means the Phase-1 framework has lower error than the MLP; negative means higher error. Do not select an epoch from these locked-test results."]
    markdown_path = framework_root / "comparison.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    image_dir = framework_root / "images"
    image_dir.mkdir(exist_ok=True)
    figure_path = image_dir / "baseline_error_comparison.png"
    epochs = [row["epoch"] for row in comparison["rows"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for metric, axis in (("mae", axes[0]), ("rmse", axes[1])):
        axis.plot(epochs, [row["framework"][metric] for row in comparison["rows"]], marker="o", label="Phase-1 framework")
        axis.plot(epochs, [row["mlp"][metric] for row in comparison["rows"]], marker="s", label="Raw-OHLCV MLP")
        axis.plot(epochs, [row["lstm_context"][metric] for row in comparison["rows"]], marker="^", linestyle="--", label="LSTM (context)")
        axis.set(title=metric.upper(), xlabel="Epoch budget", ylabel=metric.upper())
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Phase-1 price prediction: strict MLP comparison; LSTM context only")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return [json_path, markdown_path, figure_path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare completed Phase-1 price results with baseline artifacts.")
    parser.add_argument("--framework-root", required=True, type=Path)
    parser.add_argument("--mlp-root", default="src/baselines/mlp_baseline/experiments/2026-08-04-price-sweep-15-50-100", type=Path)
    parser.add_argument("--lstm-root", default="src/baselines/lstm_baseline/experiments", type=Path)
    args = parser.parse_args(argv)
    try:
        comparison = build_comparison(args.framework_root, args.mlp_root, args.lstm_root)
        for output in write_outputs(args.framework_root, comparison):
            print(output)
    except Exception as exc:
        print(f"price comparison failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
