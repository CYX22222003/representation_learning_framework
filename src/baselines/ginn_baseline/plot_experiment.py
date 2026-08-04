from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_image_dir(parent: Path) -> Path:
    image_dir = parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def _corr_label(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def plot_training_curve(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    with np.load(budget_dir / "history.npz") as history:
        epochs = history["epochs"]
        total = history["train_total_loss"]
        gt = history["train_gt_mse"]
        garch = history["train_garch_mse"]
    out = _ensure_image_dir(budget_dir) / "training_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, total, label="fused loss")
    ax.plot(epochs, gt, label="realised MSE")
    ax.plot(epochs, garch, label="GARCH MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GINN training loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_pred_vs_actual(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    metrics = json.loads((budget_dir / "metrics.json").read_text(encoding="utf-8"))
    with np.load(budget_dir / "predictions.npz") as data:
        preds = data["preds"].reshape(-1)
        targets = data["targets"].reshape(-1)
    out = _ensure_image_dir(budget_dir) / "pred_vs_actual.png"
    low = float(min(np.min(preds), np.min(targets)))
    high = float(max(np.max(preds), np.max(targets)))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets, preds, s=16, alpha=0.7)
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlabel("Realised volatility")
    ax.set_ylabel("Predicted volatility")
    ax.set_title(
        f"Epoch {metrics['epoch']} | MSE={metrics['mse']:.4g} | "
        f"corr={_corr_label(metrics['pearson_corr'])} | N={metrics['sample_count']}"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_error_distribution(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    with np.load(budget_dir / "predictions.npz") as data:
        errors = data["preds"].reshape(-1) - data["targets"].reshape(-1)
    out = _ensure_image_dir(budget_dir) / "error_distribution.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=min(30, max(5, len(errors))), alpha=0.75)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.axvline(float(np.mean(errors)), color="red", linewidth=1)
    ax.set_xlabel("Prediction minus realised volatility")
    ax.set_ylabel("Count")
    ax.set_title(f"Error distribution | mean={np.mean(errors):.4g} | std={np.std(errors):.4g}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_epoch_sweep(run_root: str | Path) -> Path:
    run_root = Path(run_root)
    sweep = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in sweep]
    mses = [row["mse"] for row in sweep]
    corrs = [np.nan if row["pearson_corr"] is None else row["pearson_corr"] for row in sweep]
    out = _ensure_image_dir(run_root) / "epoch_sweep.png"
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, mses, marker="o", color="tab:blue", label="MSE")
    ax1.set_xlabel("Epoch budget")
    ax1.set_ylabel("MSE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, corrs, marker="s", color="tab:orange", label="Pearson corr")
    ax2.set_ylabel("Pearson correlation", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_title("GINN epoch characterization sweep")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_budget(budget_dir: str | Path) -> list[Path]:
    budget_dir = Path(budget_dir)
    required = ["history.npz", "predictions.npz", "metrics.json"]
    missing = [name for name in required if not (budget_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing budget artifacts: {missing}")
    return [
        plot_training_curve(budget_dir),
        plot_pred_vs_actual(budget_dir),
        plot_error_distribution(budget_dir),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot saved GINN experiment artifacts.")
    parser.add_argument("path")
    args = parser.parse_args(argv)
    root = Path(args.path)
    try:
        if (root / "sweep_metrics.json").exists():
            for budget_dir in sorted(
                [path for path in root.iterdir() if path.is_dir() and path.name.startswith("e")],
                key=lambda path: int(path.name[1:]),
            ):
                plot_budget(budget_dir)
            plot_epoch_sweep(root)
        elif (root / "history.npz").exists():
            plot_budget(root)
        else:
            raise FileNotFoundError("expected sweep_metrics.json or history.npz in the supplied directory")
        return 0
    except Exception as exc:
        print(f"GINN plotting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
