from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_dir(path: Path) -> Path:
    out = path / "images"
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_training_curve(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    with np.load(budget_dir / "history.npz") as hist:
        epochs = hist["epochs"]
        losses = hist["train_loss"]
    out = _image_dir(budget_dir) / "training_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, color="tab:blue", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train MSE")
    ax.set_title(f"Raw LSTM volatility training curve | epoch {epochs[-1]}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_pred_vs_actual(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    metrics = _read_json(budget_dir / "metrics.json")
    with np.load(budget_dir / "predictions.npz") as data:
        preds = data["predictions"].reshape(-1)
        targets = data["targets"].reshape(-1)
    out = _image_dir(budget_dir) / "pred_vs_actual.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(targets, label="target", linewidth=1.0)
    axes[0].plot(preds, label="prediction", linewidth=1.0, alpha=0.8)
    axes[0].set_title("Time-ordered")
    axes[0].legend()
    low = float(min(preds.min(), targets.min()))
    high = float(max(preds.max(), targets.max()))
    pad = 0.02 * max(high - low, 1e-8)
    lims = [low - pad, high + pad]
    axes[1].scatter(targets, preds, s=5, alpha=0.2)
    axes[1].plot(lims, lims, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_title(f"Scatter | MSE={metrics['mse']:.4g}")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlabel("Target")
    axes[0].set_xlabel("Test row order")
    axes[1].set_ylabel("Prediction")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_residual_distribution(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    with np.load(budget_dir / "predictions.npz") as data:
        residuals = data["predictions"].reshape(-1) - data["targets"].reshape(-1)
    out = _image_dir(budget_dir) / "residual_distribution.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    hist_bins = min(60, max(1, int(np.unique(residuals).shape[0])))
    ax.hist(residuals, bins=hist_bins, alpha=0.75, color="tab:blue")
    ax.axvline(0, color="black", linewidth=1.0)
    ax.axvline(float(np.mean(residuals)), color="tab:red", linewidth=1.0)
    ax.set_xlabel("Prediction minus target")
    ax.set_ylabel("Count")
    ax.set_title("Raw LSTM volatility residual distribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_sweep_metrics(run_root: str | Path) -> Path:
    run_root = Path(run_root)
    sweep = _read_json(run_root / "sweep_metrics.json")
    epochs = [row["epoch"] for row in sweep]
    out = _image_dir(run_root) / "sweep_metrics.png"
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, [row["rmse"] for row in sweep], marker="o", label="RMSE")
    ax1.plot(epochs, [row["mae"] for row in sweep], marker="s", label="MAE")
    ax1.set_xlabel("Epoch budget")
    ax1.set_ylabel("Error")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [np.nan if row["corr"] is None else row["corr"] for row in sweep], marker="^", color="tab:orange", label="corr")
    ax2.set_ylabel("Pearson correlation")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.grid(alpha=0.3)
    ax1.set_title("Predeclared epoch characterization")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_per_contract_mse(run_root: str | Path) -> Path:
    run_root = Path(run_root)
    budget_dirs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("e")], key=lambda p: int(p.name[1:]))
    per_contract = _read_json(budget_dirs[-1] / "per_contract_metrics.json")
    ids = [row["contract_id"] for row in per_contract]
    mse = [row["mse"] for row in per_contract]
    out = _image_dir(run_root) / "per_contract_mse.png"
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(len(ids)), mse, color="tab:blue")
    ax.set_xlabel("Contract id")
    ax.set_ylabel("MSE")
    ax.set_title(f"Per-contract MSE | {budget_dirs[-1].name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_budget(budget_dir: str | Path) -> list[Path]:
    budget_dir = Path(budget_dir)
    return [plot_training_curve(budget_dir), plot_pred_vs_actual(budget_dir), plot_residual_distribution(budget_dir)]


def plot_run(run_root: str | Path) -> list[Path]:
    root = Path(run_root)
    outputs: list[Path] = []
    for budget_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("e")], key=lambda p: int(p.name[1:])):
        outputs.extend(plot_budget(budget_dir))
    outputs.append(plot_sweep_metrics(root))
    outputs.append(plot_per_contract_mse(root))
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot saved Raw LSTM volatility experiment artifacts.")
    parser.add_argument("run_root")
    args = parser.parse_args(argv)
    outputs = plot_run(args.run_root)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
