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


def _ensure_image_dir(parent: Path) -> Path:
    image_dir = parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_corr(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:.4f}"


def _budget_sort_key(path: Path) -> int:
    try:
        return int(path.name[1:])
    except ValueError:
        return 0


def plot_training_curve(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    metrics = _read_json(budget_dir / "metrics.json")
    with np.load(budget_dir / "history.npz") as history:
        epochs = history["epochs"]
        train_loss = history["train_loss"]
        seed = int(history["seed"]) if "seed" in history.files else metrics["seed"]

    out = _ensure_image_dir(budget_dir) / "training_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="train loss", color="tab:blue", linewidth=1.8)
    ax.scatter(
        [epochs[-1]],
        [train_loss[-1]],
        color="tab:red",
        zorder=5,
        label=f"epoch {epochs[-1]}, loss={train_loss[-1]:.4g}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss" if metrics["task"] == "trend" else "MSE loss")
    ax.set_title(f"Raw-OHLCV MLP {metrics['task']} training curve | seed={seed}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_pred_vs_actual(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    metrics = _read_json(budget_dir / "metrics.json")
    with np.load(budget_dir / "predictions.npz") as data:
        preds = data["preds"].reshape(-1)
        targets = data["targets"].reshape(-1)

    low = float(min(np.min(preds), np.min(targets)))
    high = float(max(np.max(preds), np.max(targets)))
    pad = 0.02 * max(high - low, 1e-8)
    lims = [low - pad, high + pad]

    out = _ensure_image_dir(budget_dir) / "pred_vs_actual.png"
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(targets, preds, s=4, alpha=0.18, color="tab:blue")
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.2, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(
        f"Raw-OHLCV MLP {metrics['task']} | epoch {metrics['epoch']} | N={metrics['test_sample_count']}\n"
        f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  corr={_format_corr(metrics['corr'])}"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_error_distribution(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    metrics = _read_json(budget_dir / "metrics.json")
    with np.load(budget_dir / "predictions.npz") as data:
        errors = data["preds"].reshape(-1) - data["targets"].reshape(-1)

    out = _ensure_image_dir(budget_dir) / "error_distribution.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=60, color="tab:blue", alpha=0.75)
    ax.axvline(0.0, color="black", linewidth=1.1)
    ax.axvline(float(np.mean(errors)), color="tab:red", linewidth=1.1, label="mean error")
    ax.set_xlabel("Prediction minus target")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Raw-OHLCV MLP {metrics['task']} error distribution | epoch {metrics['epoch']}\n"
        f"mean={np.mean(errors):.4g}  std={np.std(errors):.4g}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _binary_confusion_matrix(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    pred_labels = preds.reshape(-1).astype(np.int64)
    true_labels = targets.reshape(-1).astype(np.int64)
    cm = np.zeros((2, 2), dtype=np.int64)
    for true, pred in zip(true_labels, pred_labels):
        if true in (0, 1) and pred in (0, 1):
            cm[true, pred] += 1
    return cm


def plot_confusion_matrix(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    metrics = _read_json(budget_dir / "metrics.json")
    with np.load(budget_dir / "predictions.npz") as data:
        cm = _binary_confusion_matrix(data["preds"], data["targets"])

    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums
    class_names = ["down", "up"]

    out = _ensure_image_dir(budget_dir) / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Raw-OHLCV MLP trend confusion matrix | epoch {metrics['epoch']}\n"
        f"accuracy={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  N={metrics['test_sample_count']}"
    )
    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized share")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_probability_histogram(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    metrics = _read_json(budget_dir / "metrics.json")
    with np.load(budget_dir / "predictions.npz") as data:
        logits = data["logits"].reshape(-1)
        targets = data["targets"].reshape(-1).astype(np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits))

    out = _ensure_image_dir(budget_dir) / "probability_histogram.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probs[targets == 0], bins=40, range=(0.0, 1.0), alpha=0.65, label="actual down")
    ax.hist(probs[targets == 1], bins=40, range=(0.0, 1.0), alpha=0.65, label="actual up")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Predicted probability of up trend")
    ax.set_ylabel("Count")
    ax.set_title(f"Raw-OHLCV MLP trend probability distribution | epoch {metrics['epoch']}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_epoch_sweep(run_root: str | Path) -> Path:
    run_root = Path(run_root)
    sweep = _read_json(run_root / "sweep_metrics.json")
    if not sweep:
        raise ValueError("sweep_metrics.json is empty")

    task = sweep[0]["task"]
    epochs = [row["epoch"] for row in sweep]
    out = _ensure_image_dir(run_root) / "epoch_sweep.png"
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.set_xlabel("Epoch budget")
    ax1.grid(alpha=0.3)

    if task == "trend":
        ax1.plot(epochs, [row["accuracy"] for row in sweep], marker="o", color="tab:blue", label="accuracy")
        ax1.plot(epochs, [row["f1"] for row in sweep], marker="s", color="tab:orange", label="F1")
        ax1.set_ylabel("Score")
        ax1.set_ylim(0.0, 1.0)
        ax1.legend()
    else:
        ax1.plot(epochs, [row["rmse"] for row in sweep], marker="o", color="tab:blue", label="RMSE")
        ax1.plot(epochs, [row["mae"] for row in sweep], marker="s", color="tab:green", label="MAE")
        ax1.set_ylabel("Error", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(epochs, [row["corr"] for row in sweep], marker="^", color="tab:orange", label="corr")
        ax2.set_ylabel("Correlation", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(f"Raw-OHLCV MLP {task} epoch characterization sweep")
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

    metrics = _read_json(budget_dir / "metrics.json")
    outputs = [plot_training_curve(budget_dir)]
    if metrics["task"] == "trend":
        outputs += [plot_confusion_matrix(budget_dir), plot_probability_histogram(budget_dir)]
    else:
        outputs += [plot_pred_vs_actual(budget_dir), plot_error_distribution(budget_dir)]
    return outputs


def plot_run(root: str | Path) -> list[Path]:
    root = Path(root)
    outputs: list[Path] = []
    if (root / "sweep_metrics.json").exists():
        budget_dirs = sorted(
            [path for path in root.iterdir() if path.is_dir() and path.name.startswith("e")],
            key=_budget_sort_key,
        )
        for budget_dir in budget_dirs:
            outputs.extend(plot_budget(budget_dir))
        outputs.append(plot_epoch_sweep(root))
        return outputs
    if (root / "history.npz").exists():
        return plot_budget(root)
    raise FileNotFoundError("expected sweep_metrics.json or history.npz in the supplied directory")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot saved Raw-OHLCV MLP baseline artifacts.")
    parser.add_argument("path", help="Run root or individual e* budget directory.")
    args = parser.parse_args(argv)
    try:
        for path in plot_run(args.path):
            print(f"Wrote {path}")
        return 0
    except Exception as exc:
        print(f"MLP plotting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
