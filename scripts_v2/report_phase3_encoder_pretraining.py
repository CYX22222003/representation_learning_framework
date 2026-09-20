"""Validate, aggregate, and plot completed Phase 3 encoder-pretraining runs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_encoder_common import (  # noqa: E402
    DEFAULT_ENCODER_ROOT,
    ENCODER_VARIANTS,
    EPOCH_BUDGETS,
    PHASE3_EXPERIMENT_ROOT,
    read_json,
    require_phase3_path,
    sha256_file,
    write_json,
)


def _run_root(root: Path, variant: str) -> Path:
    family, backbone = variant.split("_", 1)
    return root / family / backbone / "seed0"


def validate_run(run_root: Path, variant: str) -> tuple[list[dict], dict[str, np.ndarray]]:
    required = (
        "config.json", "environment.json", "dataset_manifest.json",
        "architecture_manifest.json", "sweep_metrics.json", "training_complete.json",
    )
    missing = [name for name in required if not (run_root / name).is_file()]
    if missing:
        raise ValueError(f"{variant}: missing run artifacts {missing}")
    completion = read_json(run_root / "training_complete.json")
    if completion.get("complete") is not True or completion.get("variant") != variant:
        raise ValueError(f"{variant}: invalid completion marker")
    if completion.get("budgets") != list(EPOCH_BUDGETS):
        raise ValueError(f"{variant}: budget mismatch")
    config = read_json(run_root / "config.json")
    if config.get("variant") != variant or config.get("seed") != 0:
        raise ValueError(f"{variant}: configuration mismatch")
    dataset = read_json(run_root / "dataset_manifest.json")
    metrics_payload = read_json(run_root / "sweep_metrics.json")
    snapshots = metrics_payload.get("snapshots")
    if not isinstance(snapshots, list) or [row.get("epoch") for row in snapshots] != list(EPOCH_BUDGETS):
        raise ValueError(f"{variant}: incomplete snapshot metrics")
    for epoch, row in zip(EPOCH_BUDGETS, snapshots, strict=True):
        budget = run_root / f"e{epoch}"
        checkpoint = budget / "checkpoint.pth"
        history_path = budget / "history.npz"
        metrics_path = budget / "metrics.json"
        if not checkpoint.is_file() or not history_path.is_file() or not metrics_path.is_file():
            raise ValueError(f"{variant}: incomplete e{epoch} artifacts")
        if sha256_file(checkpoint) != row.get("checkpoint_sha256"):
            raise ValueError(f"{variant}: e{epoch} checkpoint hash mismatch")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("phase") != 3 or saved.get("variant") != variant or saved.get("completed_epoch") != epoch:
            raise ValueError(f"{variant}: e{epoch} checkpoint metadata mismatch")
        if saved.get("processed_npz_sha256") != dataset.get("processed_npz_sha256"):
            raise ValueError(f"{variant}: e{epoch} dataset provenance mismatch")
    with np.load(run_root / "e50" / "history.npz", allow_pickle=False) as history_npz:
        history = {name: np.asarray(history_npz[name]) for name in history_npz.files}
    if not np.array_equal(history["epochs"], np.arange(1, 51)):
        raise ValueError(f"{variant}: history is not one uninterrupted 50-epoch trajectory")
    for name, values in history.items():
        if name != "epochs" and not np.isfinite(values).all():
            raise ValueError(f"{variant}: non-finite history values in {name}")
    return snapshots, history


def _plot_family(histories: dict[str, dict[str, np.ndarray]], family: str, out_dir: Path) -> list[str]:
    variants = [name for name in ENCODER_VARIANTS if name.startswith(f"{family}_")]
    outputs = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant in variants:
        history = histories[variant]
        ax.plot(history["epochs"], history["loss"], label=variant.split("_", 1)[1])
    ax.set(title=f"Phase 3 {family} training loss", xlabel="Epoch", ylabel="Training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{family}_training_loss.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path))

    if family == "vae":
        history = histories["vae_mlp"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["recon"], label="reconstruction MSE")
        ax.plot(history["epochs"], history["kld"], label="KL divergence")
        ax.set(title="Phase 3 VAE loss components", xlabel="Epoch", ylabel="Loss component")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out_dir / "vae_loss_components.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(str(path))
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        for variant in variants:
            history = histories[variant]
            ax.plot(history["epochs"], history["embedding_std"], label=variant.split("_", 1)[1])
        ax.axhline(1e-3, color="black", linestyle="--", linewidth=1, label="collapse threshold")
        ax.set(title=f"Phase 3 {family} representation health", xlabel="Epoch", ylabel="Mean embedding std")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"{family}_embedding_std.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def report(encoder_root: Path, out_dir: Path) -> dict:
    require_phase3_path(encoder_root, PHASE3_EXPERIMENT_ROOT / "encoder_pretraining")
    require_phase3_path(out_dir, PHASE3_EXPERIMENT_ROOT / "reports")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty report directory: {out_dir}")
    rows: list[dict] = []
    histories: dict[str, dict[str, np.ndarray]] = {}
    dataset_hashes = set()
    for variant in ENCODER_VARIANTS:
        run_root = _run_root(encoder_root, variant)
        snapshots, history = validate_run(run_root, variant)
        histories[variant] = history
        dataset_hashes.add(read_json(run_root / "dataset_manifest.json")["processed_npz_sha256"])
        for row in snapshots:
            rows.append(row)
    if len(dataset_hashes) != 1:
        raise ValueError("encoder runs were trained from different processed bundles")

    out_dir.mkdir(parents=True, exist_ok=False)
    fieldnames = sorted({name for row in rows for name in row})
    with (out_dir / "encoder_snapshot_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "encoder_snapshot_metrics.json", {"snapshots": rows})

    plots = []
    for family in ("vae", "contrastive", "byol"):
        plots.extend(_plot_family(histories, family, out_dir))

    lines = [
        "# Phase 3 encoder-pretraining report", "",
        "All values are training-side diagnostics from seed 0. The runs use one uninterrupted",
        "50-epoch trajectory with snapshots at epochs 5, 15, and 50. Epoch 50 was",
        "predeclared for downstream feature extraction; no test metric selected it.", "",
        "| variant | epoch | loss | embedding std | parameters | seconds | collapse |", 
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['epoch']} | {row['train_loss']:.8f} | "
            f"{row['embedding_std']:.8f} | {row['trainable_parameter_count']} | "
            f"{row['elapsed_seconds']:.2f} | {row['collapse_warning']} |"
        )
    lines += ["", "## Visuals", ""] + [f"- `{Path(path).name}`" for path in plots]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "valid": True,
        "encoder_count": len(ENCODER_VARIANTS),
        "snapshot_count": len(rows),
        "processed_npz_sha256": next(iter(dataset_hashes)),
        "plots": plots,
    }
    write_json(out_dir / "report_manifest.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-root", type=Path, default=DEFAULT_ENCODER_ROOT)
    parser.add_argument(
        "--out-dir", type=Path,
        default=PHASE3_EXPERIMENT_ROOT / "reports" / "encoder_pretraining_seed0",
    )
    args = parser.parse_args(argv)
    try:
        result = report(args.encoder_root, args.out_dir)
        print(f"Phase 3 encoder report complete: {result}")
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Phase 3 encoder reporting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
