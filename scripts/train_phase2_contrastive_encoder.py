"""Pretrain Phase-2 LSTM/Transformer contrastive encoder candidates."""

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
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.encoder_variants import (  # noqa: E402
    CONTRASTIVE_VARIANTS,
    TemporalBackboneConfig,
    build_contrastive_variant,
    trainable_parameter_count,
)
from training.train_encoder_variants import train_temporal_contrastive_epoch  # noqa: E402


@dataclass(frozen=True)
class Phase2ContrastiveConfig:
    variant: str
    epoch_budgets: tuple[int, ...] = (15, 50, 100)
    seed: int = 0
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 128
    temperature: float = 0.2
    hidden_dim: int = 128
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
        if self.variant not in CONTRASTIVE_VARIANTS:
            raise ValueError(f"variant must be one of {CONTRASTIVE_VARIANTS}")
        if not self.epoch_budgets or any(epoch <= 0 for epoch in self.epoch_budgets):
            raise ValueError("epoch budgets must be non-empty and positive")
        if tuple(sorted(set(self.epoch_budgets))) != self.epoch_budgets:
            raise ValueError("epoch budgets must be unique and sorted")
        if self.batch_size <= 1:
            raise ValueError("batch_size must be greater than 1 for NT-Xent")
        if self.learning_rate <= 0.0 or self.temperature <= 0.0:
            raise ValueError("learning_rate and temperature must be positive")
        if self.weight_decay < 0.0 or self.collapse_std_threshold < 0.0:
            raise ValueError("weight_decay and collapse_std_threshold must be non-negative")
        if self.hidden_dim != 128:
            raise ValueError("Phase-2 primary encoder candidates must retain a 128-d downstream representation")
        if self.embedding_dim != 128:
            raise ValueError("Phase-2 primary candidates retain the reference 128-d projector output")
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


def _resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_data(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    with np.load(path) as bundle:
        if "train" not in bundle or "test" not in bundle:
            raise ValueError("processed npz must contain train and test arrays")
        train = np.asarray(bundle["train"], dtype=np.float32)
        test_shape = tuple(int(value) for value in bundle["test"].shape)
    if train.ndim != 3 or len(test_shape) != 3 or tuple(train.shape[1:]) != tuple(test_shape[1:]):
        raise ValueError("processed train/test arrays must share shape [N, sequence, features]")
    if not np.isfinite(train).all():
        raise ValueError("training sequences contain non-finite values")
    return train, test_shape


def _make_loader(train: np.ndarray, config: Phase2ContrastiveConfig) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(train)),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )


def _history_payload(history: list[dict[str, float]], seconds: list[float]) -> dict[str, np.ndarray]:
    return {
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        "train_loss": np.asarray([row["loss"] for row in history], dtype=np.float32),
        "embedding_std": np.asarray([row["embedding_std"] for row in history], dtype=np.float32),
        "embedding_norm": np.asarray([row["embedding_norm"] for row in history], dtype=np.float32),
        "projected_std": np.asarray([row["projected_std"] for row in history], dtype=np.float32),
        "epoch_seconds": np.asarray(seconds, dtype=np.float32),
    }


def _write_summaries(run_root: Path, snapshots: list[dict]) -> None:
    lines = [
        "# Phase 2 contrastive encoder pretraining",
        "",
        "Training-side fixed-budget diagnostics only; no downstream test result is used for selection.",
        "",
        "| epoch | NT-Xent | embedding std | projected std | collapse | parameters | seconds |",
        "|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in snapshots:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.8f} | {row['embedding_std']:.8f} | "
            f"{row['projected_std']:.8f} | {row['collapse_warning']} | "
            f"{row['trainable_parameter_count']} | {row['elapsed_seconds']:.2f} |"
        )
        budget_lines = [
            f"# {row['variant']} epoch {row['epoch']}",
            "",
            "This checkpoint was trained without labels on the locked training split only.",
            "",
            f"- train NT-Xent loss: `{row['train_loss']:.10f}`",
            f"- backbone embedding std: `{row['embedding_std']:.10f}`",
            f"- projected embedding std: `{row['projected_std']:.10f}`",
            f"- collapse warning: `{row['collapse_warning']}`",
            f"- trainable parameters: `{row['trainable_parameter_count']}`",
            f"- elapsed seconds: `{row['elapsed_seconds']:.2f}`",
            "",
            "Artifacts: `checkpoint.pth`, `history.npz`, and `metrics.json`.",
        ]
        (run_root / f"e{row['epoch']}" / "summary.md").write_text(
            "\n".join(budget_lines) + "\n", encoding="utf-8"
        )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    processed_npz: Path,
    run_root: Path,
    checkpoint_path: Path,
    config: Phase2ContrastiveConfig,
) -> list[dict]:
    train, test_shape = _load_data(processed_npz)
    if train.shape[1] > config.max_seq_len:
        raise ValueError("processed sequence length exceeds max_seq_len")
    device = _resolve_device(config.device)
    _set_seed(config.seed)
    model = build_contrastive_variant(
        config.variant,
        input_dim=int(train.shape[2]),
        embedding_dim=config.embedding_dim,
        backbone_config=config.backbone_config(),
    ).to(device)
    parameter_count = trainable_parameter_count(model)
    loader = _make_loader(train, config)
    if len(loader) == 0:
        raise ValueError("training split is too small for batch_size with drop_last=True")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    dataset_sha256 = _sha256(processed_npz)

    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "config.json", config.to_dict())
    _write_json(
        run_root / "dataset_manifest.json",
        {
            "processed_npz": str(processed_npz),
            "processed_npz_sha256": dataset_sha256,
            "train_sequence_shape": list(train.shape),
            "test_sequence_shape": list(test_shape),
            "test_usage": "shape recorded only; test rows are not loaded into encoder training",
        },
    )
    _write_json(
        run_root / "architecture_manifest.json",
        {
            "variant": config.variant,
            "objective": "NT-Xent",
            "augmentation_source": "models.contrastive.make_views",
            "downstream_embedding": "unnormalized backbone state h",
            "downstream_embedding_dim": config.hidden_dim,
            "projector_embedding_dim": config.embedding_dim,
            "trainable_parameter_count": parameter_count,
            "backbone_config": config.backbone_config().to_dict(),
        },
    )

    history: list[dict[str, float]] = []
    epoch_seconds: list[float] = []
    snapshots: list[dict] = []
    started_at = time.perf_counter()
    budgets = set(config.epoch_budgets)
    for epoch in range(1, max(config.epoch_budgets) + 1):
        epoch_start = time.perf_counter()
        diagnostics = train_temporal_contrastive_epoch(
            model, loader, optimizer, device=str(device), temperature=config.temperature
        )
        history.append(diagnostics)
        epoch_seconds.append(time.perf_counter() - epoch_start)
        print(
            f"{config.variant} epoch={epoch} loss={diagnostics['loss']:.8f} "
            f"embedding_std={diagnostics['embedding_std']:.8f} seconds={epoch_seconds[-1]:.2f}",
            flush=True,
        )
        if epoch not in budgets:
            continue

        budget_dir = run_root / f"e{epoch}"
        budget_dir.mkdir(parents=True, exist_ok=True)
        budget_checkpoint = budget_dir / "checkpoint.pth"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_family": "temporal_contrastive_encoder",
            "variant": config.variant,
            "completed_epoch": epoch,
            "seq_len": int(train.shape[1]),
            "input_dim": int(train.shape[2]),
            "hidden_dim": config.hidden_dim,
            "embedding_dim": config.embedding_dim,
            "backbone_config": config.backbone_config().to_dict(),
            "training_config": config.to_dict(),
            "dataset_path": str(processed_npz),
            "dataset_sha256": dataset_sha256,
            "trainable_parameter_count": parameter_count,
        }
        torch.save(checkpoint, budget_checkpoint)
        np.savez_compressed(budget_dir / "history.npz", **_history_payload(history, epoch_seconds))
        metrics = {
            "variant": config.variant,
            "epoch": epoch,
            "seed": config.seed,
            "device": str(device),
            "train_loss": diagnostics["loss"],
            "best_train_loss": min(row["loss"] for row in history),
            "embedding_std": diagnostics["embedding_std"],
            "embedding_norm": diagnostics["embedding_norm"],
            "projected_std": diagnostics["projected_std"],
            "collapse_warning": diagnostics["embedding_std"] < config.collapse_std_threshold,
            "trainable_parameter_count": parameter_count,
            "elapsed_seconds": time.perf_counter() - started_at,
            "checkpoint_path": str(budget_checkpoint),
        }
        _write_json(budget_dir / "metrics.json", metrics)
        snapshots.append(metrics)
        if epoch == max(config.epoch_budgets):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(budget_checkpoint, checkpoint_path)

    _write_json(run_root / "sweep_metrics.json", snapshots)
    _write_summaries(run_root, snapshots)
    return snapshots


def _parse_budgets(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one epoch budget")
    return values


def _safe_run_name(name: str) -> str:
    path = Path(name)
    if not name or path.is_absolute() or len(path.parts) != 1 or name in {".", ".."}:
        raise ValueError("run-name must be one simple relative directory name")
    return name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=CONTRASTIVE_VARIANTS, required=True)
    parser.add_argument("--processed-npz", type=Path, default=Path("data/processed/market_4h_seq64_top50.npz"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--epoch-budgets", default="15,50,100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lstm-num-layers", type=int, default=1)
    parser.add_argument("--lstm-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-num-layers", type=int, default=2)
    parser.add_argument("--transformer-num-heads", type=int, default=4)
    parser.add_argument("--transformer-feedforward-dim", type=int, default=256)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--transformer-norm-first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--collapse-std-threshold", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_name = _safe_run_name(args.run_name)
        config = Phase2ContrastiveConfig(
            variant=args.variant,
            epoch_budgets=_parse_budgets(args.epoch_budgets),
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            embedding_dim=args.embedding_dim,
            temperature=args.temperature,
            hidden_dim=args.hidden_dim,
            lstm_num_layers=args.lstm_num_layers,
            lstm_dropout=args.lstm_dropout,
            transformer_num_layers=args.transformer_num_layers,
            transformer_num_heads=args.transformer_num_heads,
            transformer_feedforward_dim=args.transformer_feedforward_dim,
            transformer_dropout=args.transformer_dropout,
            transformer_norm_first=args.transformer_norm_first,
            max_seq_len=args.max_seq_len,
            collapse_std_threshold=args.collapse_std_threshold,
            device=args.device,
        )
        run_root = args.run_root or ROOT / "experiments/framework/phase2/encoder_refinement/pretraining" / run_name
        checkpoint_path = args.checkpoint_path or ROOT / "checkpoints/phase2" / f"{run_name}.pth"
        occupied = (run_root.exists() and any(run_root.iterdir())) or checkpoint_path.exists()
        if occupied and not args.overwrite:
            raise FileExistsError("Phase-2 run artifacts already exist; pass --overwrite to replace them")
        if args.overwrite:
            if run_root.exists():
                shutil.rmtree(run_root)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        run_experiment(args.processed_npz, run_root, checkpoint_path, config)
        print(f"completed {config.variant} pretraining: run={run_root} checkpoint={checkpoint_path}")
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Phase-2 contrastive encoder training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
