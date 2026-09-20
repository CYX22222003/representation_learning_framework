"""Pretrain Phase-2 LSTM/Transformer BYOL encoder candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.encoder_variants import (  # noqa: E402
    BYOL_VARIANTS,
    TemporalBackboneConfig,
    build_byol_variant,
    trainable_parameter_count,
)
from training.train_byol import train_byol_epoch  # noqa: E402


@dataclass(frozen=True)
class Phase2BYOLConfig:
    variant: str
    epoch_budgets: tuple[int, ...] = (15, 50, 100)
    seed: int = 0
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    projection_dim: int = 128
    predictor_hidden_dim: int = 128
    target_decay: float = 0.99
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0
    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    transformer_feedforward_dim: int = 256
    transformer_dropout: float = 0.1
    transformer_norm_first: bool = False
    max_seq_len: int = 512
    collapse_std_threshold: float = 1e-3
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.variant not in BYOL_VARIANTS:
            raise ValueError(f"variant must be one of {BYOL_VARIANTS}")
        if not self.epoch_budgets or any(value <= 0 for value in self.epoch_budgets):
            raise ValueError("epoch budgets must be non-empty and positive")
        if tuple(sorted(set(self.epoch_budgets))) != self.epoch_budgets:
            raise ValueError("epoch budgets must be unique and sorted")
        if self.batch_size <= 1 or self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimiser configuration")
        if self.hidden_dim != 128 or self.projection_dim != 128:
            raise ValueError("primary BYOL variants must retain 128-d backbone/projector outputs")
        if self.predictor_hidden_dim <= 0 or not 0.0 <= self.target_decay <= 1.0:
            raise ValueError("invalid BYOL predictor or EMA configuration")
        if self.collapse_std_threshold < 0.0:
            raise ValueError("collapse threshold must be non-negative")
        self.backbone_config()

    def backbone_config(self) -> TemporalBackboneConfig:
        return TemporalBackboneConfig(
            hidden_dim=self.hidden_dim,
            lstm_num_layers=self.lstm_num_layers,
            lstm_dropout=self.lstm_dropout,
            transformer_num_layers=self.transformer_num_layers,
            transformer_num_heads=self.transformer_num_heads,
            transformer_feedforward_dim=self.transformer_feedforward_dim,
            transformer_dropout=self.transformer_dropout,
            transformer_norm_first=self.transformer_norm_first,
            max_seq_len=self.max_seq_len,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["epoch_budgets"] = list(self.epoch_budgets)
        return payload


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_train(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    with np.load(path, allow_pickle=False) as data:
        train = np.asarray(data["train"], dtype=np.float32)
        test_shape = tuple(int(value) for value in data["test"].shape)
    if train.ndim != 3 or len(test_shape) != 3 or train.shape[1:] != test_shape[1:]:
        raise ValueError("processed train/test arrays must share [N, sequence, features] shape")
    if not np.isfinite(train).all():
        raise ValueError("training sequences contain non-finite values")
    return train, test_shape


def _history_payload(history: list[dict[str, float]], seconds: list[float]) -> dict[str, np.ndarray]:
    return {
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        "train_loss": np.asarray([row["loss"] for row in history], dtype=np.float32),
        "view_cosine": np.asarray([row["view_cosine"] for row in history], dtype=np.float32),
        "embedding_std": np.asarray([row["embedding_std"] for row in history], dtype=np.float32),
        "embedding_norm": np.asarray([row["embedding_norm"] for row in history], dtype=np.float32),
        "epoch_seconds": np.asarray(seconds, dtype=np.float32),
    }


def _write_summary(run_root: Path, rows: list[dict]) -> None:
    lines = [
        "# Phase 2 temporal BYOL encoder pretraining",
        "",
        "Training-only diagnostics; no downstream test result is used for selection.",
        "",
        "| epoch | BYOL loss | view cosine | embedding std | collapse | parameters | seconds |",
        "|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.8f} | {row['view_cosine']:.8f} | "
            f"{row['embedding_std']:.8f} | {row['collapse_warning']} | "
            f"{row['trainable_parameter_count']} | {row['elapsed_seconds']:.2f} |"
        )
        (run_root / f"e{row['epoch']}" / "summary.md").write_text(
            f"# {row['variant']} epoch {row['epoch']}\n\n"
            f"- train BYOL loss: `{row['train_loss']:.10f}`\n"
            f"- view cosine: `{row['view_cosine']:.10f}`\n"
            f"- embedding std: `{row['embedding_std']:.10f}`\n"
            f"- collapse warning: `{row['collapse_warning']}`\n"
            f"- trainable parameters: `{row['trainable_parameter_count']}`\n"
            f"- elapsed seconds: `{row['elapsed_seconds']:.2f}`\n",
            encoding="utf-8",
        )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    processed_npz: Path,
    run_root: Path,
    checkpoint_path: Path,
    config: Phase2BYOLConfig,
) -> list[dict]:
    train, test_shape = _load_train(processed_npz)
    if train.shape[1] > config.max_seq_len:
        raise ValueError("processed sequence length exceeds max_seq_len")
    _set_seed(config.seed)
    device = _device(config.device)
    model = build_byol_variant(
        config.variant,
        input_dim=int(train.shape[2]),
        projection_dim=config.projection_dim,
        predictor_hidden_dim=config.predictor_hidden_dim,
        backbone_config=config.backbone_config(),
    ).to(device)
    parameter_count = trainable_parameter_count(model)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train)), batch_size=config.batch_size,
        shuffle=True, drop_last=True, generator=generator,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    dataset_sha = _sha256(processed_npz)
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "config.json", config.to_dict())
    _write_json(run_root / "dataset_manifest.json", {
        "processed_npz": str(processed_npz), "processed_npz_sha256": dataset_sha,
        "train_sequence_shape": list(train.shape), "test_sequence_shape": list(test_shape),
        "test_usage": "shape recorded only; test rows are not used for encoder training",
    })
    _write_json(run_root / "architecture_manifest.json", {
        "variant": config.variant, "objective": "BYOL",
        "augmentation_source": "models.contrastive.make_views",
        "downstream_embedding": "unnormalized online backbone state h",
        "downstream_embedding_dim": config.hidden_dim,
        "projection_dim": config.projection_dim,
        "predictor_hidden_dim": config.predictor_hidden_dim,
        "target_decay": config.target_decay,
        "trainable_parameter_count": parameter_count,
        "backbone_config": config.backbone_config().to_dict(),
    })

    history: list[dict[str, float]] = []
    seconds: list[float] = []
    snapshots: list[dict] = []
    started = time.perf_counter()
    budgets = set(config.epoch_budgets)
    for epoch in range(1, max(config.epoch_budgets) + 1):
        epoch_started = time.perf_counter()
        diagnostics = train_byol_epoch(
            model, loader, optimizer, device=str(device), target_decay=config.target_decay
        )
        history.append(diagnostics)
        seconds.append(time.perf_counter() - epoch_started)
        print(
            f"{config.variant} epoch={epoch} loss={diagnostics['loss']:.8f} "
            f"embedding_std={diagnostics['embedding_std']:.8f} seconds={seconds[-1]:.2f}",
            flush=True,
        )
        if epoch not in budgets:
            continue
        budget_dir = run_root / f"e{epoch}"
        budget_dir.mkdir(parents=True, exist_ok=True)
        budget_checkpoint = budget_dir / "checkpoint.pth"
        checkpoint = {
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "model_family": "temporal_byol_encoder", "variant": config.variant,
            "completed_epoch": epoch, "seq_len": int(train.shape[1]),
            "input_dim": int(train.shape[2]), "hidden_dim": config.hidden_dim,
            "projection_dim": config.projection_dim,
            "predictor_hidden_dim": config.predictor_hidden_dim,
            "backbone_config": config.backbone_config().to_dict(),
            "training_config": config.to_dict(), "dataset_path": str(processed_npz),
            "dataset_sha256": dataset_sha, "trainable_parameter_count": parameter_count,
        }
        torch.save(checkpoint, budget_checkpoint)
        np.savez_compressed(budget_dir / "history.npz", **_history_payload(history, seconds))
        row = {
            "variant": config.variant, "epoch": epoch, "seed": config.seed,
            "device": str(device), "train_loss": diagnostics["loss"],
            "best_train_loss": min(value["loss"] for value in history),
            "view_cosine": diagnostics["view_cosine"],
            "embedding_std": diagnostics["embedding_std"],
            "embedding_norm": diagnostics["embedding_norm"],
            "collapse_warning": diagnostics["embedding_std"] < config.collapse_std_threshold,
            "trainable_parameter_count": parameter_count,
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint_path": str(budget_checkpoint),
        }
        _write_json(budget_dir / "metrics.json", row)
        snapshots.append(row)
        if epoch == max(config.epoch_budgets):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(budget_checkpoint, checkpoint_path)
    _write_json(run_root / "sweep_metrics.json", snapshots)
    _write_summary(run_root, snapshots)
    return snapshots


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("expected at least one epoch budget")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--variant", choices=BYOL_VARIANTS, required=True)
    result.add_argument("--processed-npz", type=Path, default=Path("data/processed/market_4h_seq64_top50.npz"))
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--checkpoint-path", type=Path, required=True)
    result.add_argument("--epoch-budgets", default="15,50,100")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--target-decay", type=float, default=0.99)
    result.add_argument("--device", default="auto")
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if ((args.run_root.exists() and any(args.run_root.iterdir())) or args.checkpoint_path.exists()) and not args.overwrite:
            raise FileExistsError("BYOL variant artifacts already exist; pass --overwrite to replace them")
        if args.overwrite:
            if args.run_root.exists():
                shutil.rmtree(args.run_root)
            if args.checkpoint_path.exists():
                args.checkpoint_path.unlink()
        config = Phase2BYOLConfig(
            variant=args.variant, epoch_budgets=_csv_ints(args.epoch_budgets),
            seed=args.seed, batch_size=args.batch_size, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, target_decay=args.target_decay, device=args.device,
        )
        run_experiment(args.processed_npz, args.run_root, args.checkpoint_path, config)
        print(f"Phase 2 BYOL variant completed: {args.run_root}", flush=True)
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Phase 2 BYOL variant failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
