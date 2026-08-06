"""Generate training curve and predicted-vs-actual plots for an LSTM run.

Usage:
    python plot_experiment.py experiments/<run-name>

Reads inside the run directory:
    history.npz       (train_loss, val_loss, best_epoch)
    predictions.npz   (preds, targets)

Writes:
    <run-dir>/images/training_curve.png
    <run-dir>/images/pred_vs_actual.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(run_dir: str) -> str:
    history = np.load(os.path.join(run_dir, "history.npz"))
    train = history["train_loss"]
    seed = int(history["seed"]) if "seed" in history.files else None
    epochs = np.arange(len(train))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, label="train RMSE", linewidth=1.8, color="#1f77b4")
    ax.scatter(
        [len(train) - 1], [train[-1]], color="red", zorder=5,
        label=f"final epoch {len(train) - 1}, RMSE={train[-1]:.4f}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train RMSE")
    title = "LSTM Baseline — Training Curve (univariate close, RMSE loss)"
    if seed is not None:
        title += f"\nepochs={len(train)}, seed={seed}"
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = os.path.join(run_dir, "images", "training_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_pred_vs_actual(run_dir: str) -> str:
    data = np.load(os.path.join(run_dir, "predictions.npz"))
    preds = data["preds"]
    targets = data["targets"]

    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, preds, s=2, alpha=0.15, color="#1f77b4")
    lim_lo = float(min(targets.min(), preds.min()))
    lim_hi = float(max(targets.max(), preds.max()))
    pad = 0.02 * (lim_hi - lim_lo)
    lims = [lim_lo - pad, lim_hi + pad]
    ax.plot(lims, lims, color="red", linestyle="--", linewidth=1.2, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual next close")
    ax.set_ylabel("Predicted next close")
    ax.set_title(
        f"LSTM Predicted vs Actual (test set, N={len(targets)})\n"
        f"MAE={mae:.4f}   RMSE={rmse:.4f}"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    out = os.path.join(run_dir, "images", "pred_vs_actual.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dir",
        help="Path to the run directory (e.g. experiments/2026-06-21-v3-rmse).",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory not found: {run_dir}")

    os.makedirs(os.path.join(run_dir, "images"), exist_ok=True)
    print(f"Wrote {plot_training_curve(run_dir)}")
    print(f"Wrote {plot_pred_vs_actual(run_dir)}")


if __name__ == "__main__":
    main()
