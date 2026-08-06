"""Generate training curve and confusion-matrix plots for a TA-MLP run.

Usage:
    python plot_experiment.py experiments/<run-name>

Reads inside the run directory:
    history.npz           (train_loss, epochs, seed)
    confusion_matrix.npz  (cm, class_names)

Writes:
    <run-dir>/images/training_curve.png
    <run-dir>/images/confusion_matrix.png
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
    ax.plot(epochs, train, label="train CE", linewidth=1.8, color="#1f77b4")
    ax.scatter(
        [len(train) - 1], [train[-1]], color="red", zorder=5,
        label=f"final epoch {len(train) - 1}, CE={train[-1]:.4f}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train cross-entropy")
    title = "TA-MLP Baseline — Training Curve (CrossEntropy loss)"
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


def plot_confusion_matrix(run_dir: str) -> str:
    data = np.load(os.path.join(run_dir, "confusion_matrix.npz"))
    cm = data["cm"].astype(np.int64)
    class_names = [str(c) for c in data["class_names"]]
    n = cm.shape[0]

    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums

    accuracy = float(np.trace(cm)) / float(cm.sum()) if cm.sum() > 0 else 0.0
    f1s = []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    macro_f1 = float(np.mean(f1s))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"TA-MLP Confusion Matrix (test, N={cm.sum()})\n"
        f"accuracy={accuracy:.4f}   macro-F1={macro_f1:.4f}"
    )

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)",
                ha="center", va="center", color=color, fontsize=10,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized share")
    fig.tight_layout()

    out = os.path.join(run_dir, "images", "confusion_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dir",
        help="Path to the run directory (e.g. experiments/2026-06-22-v1-e15).",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory not found: {run_dir}")

    os.makedirs(os.path.join(run_dir, "images"), exist_ok=True)
    print(f"Wrote {plot_training_curve(run_dir)}")
    print(f"Wrote {plot_confusion_matrix(run_dir)}")


if __name__ == "__main__":
    main()
