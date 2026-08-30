from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _budget_dirs(run_root: Path) -> list[Path]:
    return sorted([path for path in run_root.glob("e[0-9]*") if path.is_dir()], key=lambda p: int(p.name[1:]))


def plot_oof_base_predictions(run_root: Path, out_dir: Path) -> Path:
    with np.load(run_root / "oof" / "garch_predictions.npz") as g:
        garch = g["prediction_guarded"]
        targets = g["targets"]
    first_lstm = sorted((run_root / "oof").glob("lstm_predictions_e*.npz"))[0]
    with np.load(first_lstm) as l:
        lstm = l["predictions"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(targets, garch, s=8, alpha=0.5)
    axes[0].set_title("OOF GARCH")
    axes[0].set_xlabel("target")
    axes[0].set_ylabel("prediction")
    axes[1].scatter(targets, lstm, s=8, alpha=0.5)
    axes[1].set_title(f"OOF LSTM {first_lstm.stem.split('_')[-1]}")
    axes[1].set_xlabel("target")
    axes[1].set_ylabel("prediction")
    fig.tight_layout()
    path = out_dir / "oof_base_predictions.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_raw_lstm_vs_stack(run_root: Path, out_dir: Path) -> Path:
    budget = _budget_dirs(run_root)[-1]
    with np.load(budget / "predictions.npz") as data:
        targets = data["targets"]
        lstm = data["lstm_prediction"]
        stack = data["stack_prediction_nonnegative"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(targets, lstm, s=8, alpha=0.5, label="Raw LSTM")
    axes[0].scatter(targets, stack, s=8, alpha=0.5, label="Stack")
    axes[0].set_xlabel("target")
    axes[0].set_ylabel("prediction")
    axes[0].legend()
    axes[1].hist(lstm - targets, bins=40, alpha=0.55, label="Raw LSTM")
    axes[1].hist(stack - targets, bins=40, alpha=0.55, label="Stack")
    axes[1].set_title("Residuals")
    axes[1].legend()
    fig.tight_layout()
    path = out_dir / "raw_lstm_vs_stack.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_sweep_comparison(run_root: Path, out_dir: Path) -> Path:
    rows = _load_json(run_root / "comparison_with_raw_lstm.json")
    epochs = [int(row["epoch"]) for row in rows]  # type: ignore[index]
    raw = [float(row["raw_lstm_mse"]) for row in rows]  # type: ignore[index]
    stack = [float(row["stack_mse"]) for row in rows]  # type: ignore[index]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, raw, marker="o", label="Raw LSTM")
    ax.plot(epochs, stack, marker="o", label="GARCH--LSTM stack")
    ax.set_xlabel("LSTM epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "sweep_comparison.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_meta_coefficients(run_root: Path, out_dir: Path) -> Path:
    budgets = _budget_dirs(run_root)
    names = ["garch", "lstm", "interaction"]
    values = []
    epochs = []
    for budget in budgets:
        payload = _load_json(budget / "meta_model.json")
        coeffs = payload["coefficients"]  # type: ignore[index]
        values.append([float(coeffs[name]) for name in names])
        epochs.append(int(budget.name[1:]))
    arr = np.asarray(values)
    x = np.arange(len(epochs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(names):
        ax.bar(x + (i - 1) * width, arr[:, i], width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in epochs])
    ax.set_xlabel("LSTM epoch")
    ax.set_ylabel("coefficient")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "meta_coefficients.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_garch_diagnostics(run_root: Path, out_dir: Path) -> Path:
    payload = _load_json(run_root / "garch_diagnostics.json")
    rows = payload.get("test", [])  # type: ignore[union-attr]
    contracts = [str(row["contract_id"]) for row in rows]
    fallback = [int(row.get("fallback_forecast_count", 0)) for row in rows]
    capped = [int(row.get("capped_forecast_count", 0)) for row in rows]
    fig, ax = plt.subplots(figsize=(max(7, len(contracts) * 0.25), 4))
    x = np.arange(len(contracts))
    ax.bar(x, fallback, label="fallback")
    ax.bar(x, capped, bottom=fallback, label="capped")
    ax.set_xticks(x)
    ax.set_xticklabels(contracts, rotation=90)
    ax.set_ylabel("row count")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "garch_diagnostics.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_per_contract_delta_mse(run_root: Path, out_dir: Path) -> Path:
    budget = _budget_dirs(run_root)[-1]
    rows = _load_json(budget / "per_contract_metrics.json")
    contracts = [str(row["contract_id"]) for row in rows]  # type: ignore[index]
    stack_mse = np.asarray([float(row["mse"]) for row in rows], dtype=np.float64)  # type: ignore[index]
    with np.load(budget / "predictions.npz") as data:
        targets = data["targets"]
        lstm = data["lstm_prediction"]
        cids = data["contract_ids"]
    raw_mse = []
    for cid in [int(c) for c in contracts]:
        mask = cids == cid
        raw_mse.append(float(np.mean((lstm[mask] - targets[mask]) ** 2)))
    delta = stack_mse - np.asarray(raw_mse)
    fig, ax = plt.subplots(figsize=(max(7, len(contracts) * 0.25), 4))
    ax.bar(np.arange(len(contracts)), delta)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(contracts)))
    ax.set_xticklabels(contracts, rotation=90)
    ax.set_ylabel("stack MSE minus Raw LSTM MSE")
    fig.tight_layout()
    path = out_dir / "per_contract_delta_mse.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_all(run_root: str | Path) -> list[Path]:
    root = Path(run_root)
    out_dir = root / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    required = [root / "oof" / "garch_predictions.npz", root / "comparison_with_raw_lstm.json", root / "garch_diagnostics.json"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"missing required artifact: {path}")
    paths = [
        plot_oof_base_predictions(root, out_dir),
        plot_raw_lstm_vs_stack(root, out_dir),
        plot_sweep_comparison(root, out_dir),
        plot_meta_coefficients(root, out_dir),
        plot_garch_diagnostics(root, out_dir),
        plot_per_contract_delta_mse(root, out_dir),
    ]
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot a saved GARCH--LSTM stacking experiment.")
    parser.add_argument("run_root")
    args = parser.parse_args(argv)
    try:
        paths = plot_all(args.run_root)
        print("Generated plots:")
        for path in paths:
            print(path)
        return 0
    except Exception as exc:
        print(f"plotting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
