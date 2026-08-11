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


def _ensure_image_dir(parent: Path) -> Path:
    image_dir = parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


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
        train_recon = history["train_recon"]
        train_kld = history["train_kld"]

    out = _ensure_image_dir(budget_dir) / "training_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, color="tab:blue", linewidth=1.8, label="total loss")
    ax.plot(epochs, train_recon, color="tab:green", linewidth=1.5, label="reconstruction MSE")
    ax.plot(epochs, train_kld, color="tab:orange", linewidth=1.5, label="KL divergence")
    ax.scatter(
        [epochs[-1]],
        [train_loss[-1]],
        color="tab:red",
        zorder=5,
        label=f"epoch {epochs[-1]}, loss={train_loss[-1]:.4g}",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        "VAE encoder training curve | "
        f"seed={metrics['seed']} | beta={metrics['training_config']['beta']}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_loss_components(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    with np.load(budget_dir / "history.npz") as history:
        epochs = history["epochs"]
        train_recon = history["train_recon"]
        train_kld = history["train_kld"]

    out = _ensure_image_dir(budget_dir) / "loss_components.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stackplot(
        epochs,
        train_recon,
        train_kld,
        labels=["reconstruction MSE", "KL divergence"],
        colors=["tab:green", "tab:orange"],
        alpha=0.75,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Component value")
    ax.set_title("VAE loss components")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_epoch_times(budget_dir: str | Path) -> Path:
    budget_dir = Path(budget_dir)
    with np.load(budget_dir / "history.npz") as history:
        epochs = history["epochs"]
        epoch_seconds = history["epoch_seconds"]

    out = _ensure_image_dir(budget_dir) / "epoch_times.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(epochs, epoch_seconds, color="tab:purple", alpha=0.75)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.set_title("VAE encoder epoch runtime")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_sweep(run_root: str | Path) -> Path:
    run_root = Path(run_root)
    sweep = _read_json(run_root / "sweep_metrics.json")
    if not isinstance(sweep, list) or not sweep:
        raise ValueError("sweep_metrics.json is empty")

    epochs = [row["epoch"] for row in sweep]
    train_loss = [row["train_loss"] for row in sweep]
    train_recon = [row["train_recon"] for row in sweep]
    train_kld = [row["train_kld"] for row in sweep]

    out = _ensure_image_dir(run_root) / "epoch_sweep.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, marker="o", color="tab:blue", label="total loss")
    ax.plot(epochs, train_recon, marker="s", color="tab:green", label="reconstruction MSE")
    ax.plot(epochs, train_kld, marker="^", color="tab:orange", label="KL divergence")
    ax.set_xlabel("Epoch budget")
    ax.set_ylabel("Loss")
    ax.set_title("VAE encoder fixed-budget characterization")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def write_report(run_root: str | Path) -> Path:
    run_root = Path(run_root)
    sweep = _read_json(run_root / "sweep_metrics.json")
    config = _read_json(run_root / "config.json")
    manifest = _read_json(run_root / "dataset_manifest.json")
    if not isinstance(sweep, list) or not sweep:
        raise ValueError("sweep_metrics.json is empty")

    final = sweep[-1]
    lines = [
        "# VAE Encoder Experiment Report",
        "",
        "This report covers unsupervised VAE pretraining only. Test sequences are recorded in the manifest for traceability but are not used for training, early stopping, or checkpoint selection.",
        "",
        "## Dataset",
        "",
        f"- processed npz: `{manifest['processed_npz']}`",
        f"- train shape: `{manifest['train_sequence_shape']}`",
        f"- test shape: `{manifest['test_sequence_shape']}`",
        "",
        "## Configuration",
        "",
        f"- seed: `{config['seed']}`",
        f"- batch size: `{config['batch_size']}`",
        f"- learning rate: `{config['learning_rate']}`",
        f"- weight decay: `{config['weight_decay']}`",
        f"- latent dim: `{config['latent_dim']}`",
        f"- hidden dim: `{config['hidden_dim']}`",
        f"- beta: `{config['beta']}`",
        f"- device request: `{config['device']}`",
        "",
        "## Epoch Budgets",
        "",
        "| epoch | total loss | reconstruction MSE | KL divergence | elapsed seconds | checkpoint |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in sweep:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.10f} | {row['train_recon']:.10f} | "
            f"{row['train_kld']:.10f} | {row['elapsed_seconds']:.2f} | "
            f"`{row['checkpoint_path']}` |"
        )
    lines += [
        "",
        "## Final Budget",
        "",
        f"- epoch: `{final['epoch']}`",
        f"- final total loss: `{final['train_loss']:.10f}`",
        f"- final reconstruction MSE: `{final['train_recon']:.10f}`",
        f"- final KL divergence: `{final['train_kld']:.10f}`",
        "",
        "Generated images are saved under each `e*/images/` directory and under the run-level `images/` directory.",
    ]
    out = run_root / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def plot_budget(budget_dir: str | Path) -> list[Path]:
    budget_dir = Path(budget_dir)
    required = ["history.npz", "metrics.json"]
    missing = [name for name in required if not (budget_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing budget artifacts: {missing}")
    return [plot_training_curve(budget_dir), plot_loss_components(budget_dir), plot_epoch_times(budget_dir)]


def plot_run(run_root: str | Path) -> list[Path]:
    run_root = Path(run_root)
    if not (run_root / "sweep_metrics.json").exists():
        return plot_budget(run_root)

    outputs: list[Path] = []
    budget_dirs = sorted(
        [path for path in run_root.iterdir() if path.is_dir() and path.name.startswith("e")],
        key=_budget_sort_key,
    )
    for budget_dir in budget_dirs:
        outputs.extend(plot_budget(budget_dir))
    outputs.append(plot_sweep(run_root))
    outputs.append(write_report(run_root))
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot and summarize VAE encoder pretraining artifacts.")
    parser.add_argument("path", help="Run root or individual e* budget directory.")
    args = parser.parse_args(argv)
    try:
        outputs = plot_run(args.path)
    except Exception as exc:
        print(f"vae plot failed: {exc}")
        return 1
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
