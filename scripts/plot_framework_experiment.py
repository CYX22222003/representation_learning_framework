"""Plot and summarize saved framework task-head experiment artifacts."""

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


def _read_json(path: Path) -> dict | list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_dir(path: Path) -> Path:
    out = path / "images"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _budget_dirs(run_root: Path) -> list[Path]:
    return sorted(
        (path for path in run_root.iterdir() if path.is_dir() and path.name.startswith("e")),
        key=lambda path: int(path.name[1:]),
    )


def plot_training_curve(budget_dir: Path) -> Path:
    with np.load(budget_dir / "history.npz") as history:
        epochs, loss = history["epochs"], history["train_loss"]
    metrics = _read_json(budget_dir / "metrics.json")
    out = _image_dir(budget_dir) / "training_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, loss, color="tab:blue", linewidth=1.8)
    ax.scatter(epochs[-1], loss[-1], color="tab:red", zorder=3)
    ax.set(title=f"{metrics['task'].replace('_', ' ').title()} training loss", xlabel="Epoch", ylabel="MSE / cross-entropy")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def plot_predictions(budget_dir: Path) -> Path:
    metrics = _read_json(budget_dir / "metrics.json")
    with np.load(budget_dir / "predictions.npz") as data:
        predictions, targets = data["preds"], data["targets"]
    out = _image_dir(budget_dir) / "predictions.png"
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if metrics["task"] == "trend_classification":
        matrix_path = budget_dir / "confusion_matrix.npz"
        with np.load(matrix_path) as data:
            matrix = data["confusion_matrix"]
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, label="Rows")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
        ax.set(title="Test confusion matrix", xlabel="Predicted class", ylabel="True class")
    else:
        lower = float(min(predictions.min(), targets.min()))
        upper = float(max(predictions.max(), targets.max()))
        ax.scatter(targets, predictions, s=5, alpha=0.25, color="tab:blue", rasterized=True)
        ax.plot([lower, upper], [lower, upper], "--", color="tab:red", linewidth=1)
        ax.set(title="Locked-test predictions", xlabel="Target", ylabel="Prediction")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def plot_sweep(run_root: Path) -> Path:
    sweep = _read_json(run_root / "sweep_metrics.json")
    if not isinstance(sweep, list) or not sweep:
        raise ValueError("sweep_metrics.json is empty")
    task = str(sweep[0]["task"])
    epochs = [row["epoch"] for row in sweep]
    out = _image_dir(run_root) / "epoch_sweep.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    if task == "trend_classification":
        ax.plot(epochs, [row["accuracy"] for row in sweep], marker="o", label="accuracy")
        ax.plot(epochs, [row["macro_f1"] for row in sweep], marker="s", label="macro-F1")
        ax.set_ylabel("Score")
    else:
        ax.plot(epochs, [row["mse"] for row in sweep], marker="o", label="MSE")
        ax.plot(epochs, [row["rmse"] for row in sweep], marker="s", label="RMSE")
        ax.set_ylabel("Error")
    ax.set(title=f"{task.replace('_', ' ').title()} fixed-budget characterization", xlabel="Epoch budget")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def write_report(run_root: Path) -> Path:
    sweep = _read_json(run_root / "sweep_metrics.json")
    manifest = _read_json(run_root / "dataset_manifest.json")
    if not isinstance(sweep, list) or not sweep:
        raise ValueError("sweep_metrics.json is empty")
    task = str(sweep[0]["task"])
    lines = [
        f"# Framework {task.replace('_', ' ').title()} Report",
        "",
        "All fixed epoch budgets are retained as a characterization sweep; no checkpoint is selected from locked-test performance.",
        "",
        "## Data and representation",
        "",
        f"- feature store: `{manifest['features_npz']}`",
        f"- branches: `{manifest['branch_dims']}`",
        f"- train/test samples: `{manifest['train_sample_count']}` / `{manifest['test_sample_count']}`",
        f"- label bundle: `{manifest['labels_npz']}`",
        "",
        "## Results",
        "",
    ]
    if task == "trend_classification":
        lines += ["| epoch | accuracy | macro-F1 | weighted-F1 |", "|---:|---:|---:|---:|"]
        lines += [f"| {row['epoch']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['weighted_f1']:.6f} |" for row in sweep]
    else:
        lines += ["| epoch | MAE | RMSE | MSE | Pearson correlation |", "|---:|---:|---:|---:|---:|"]
        lines += [f"| {row['epoch']} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['mse']:.6f} | {row['corr']:.6f} |" for row in sweep]
    lines += ["", "Images are stored under the run-level and per-budget `images/` folders."]
    out = run_root / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def plot_run(run_root: str | Path) -> list[Path]:
    root = Path(run_root)
    outputs: list[Path] = []
    for budget_dir in _budget_dirs(root):
        outputs.extend([plot_training_curve(budget_dir), plot_predictions(budget_dir)])
    outputs.extend([plot_sweep(root), write_report(root)])
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot saved framework task experiments.")
    parser.add_argument("run_root", help="experiments/framework/<task>/<run-name>")
    args = parser.parse_args(argv)
    try:
        for path in plot_run(args.run_root):
            print(path)
    except Exception as exc:
        print(f"framework plotting failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

