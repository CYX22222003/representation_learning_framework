from __future__ import annotations

import argparse
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

from models.contrastive import ContrastiveEncoder
from training.train_contrastive import train_contrastive_epoch


@dataclass(frozen=True)
class ContrastiveConfig:
    epoch_budgets: tuple[int, ...] = (15, 20, 25, 50, 100)
    seed: int = 0
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    embedding_dim: int = 128
    temperature: float = 0.2
    device: str = "auto"

    def __post_init__(self) -> None:
        if any(epoch <= 0 for epoch in self.epoch_budgets):
            raise ValueError("epoch budgets must be positive")
        if tuple(sorted(set(self.epoch_budgets))) != tuple(self.epoch_budgets):
            raise ValueError("epoch budgets must be unique and sorted")
        if self.batch_size <= 1:
            raise ValueError("batch_size must be greater than 1 for contrastive learning")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["epoch_budgets"] = list(self.epoch_budgets)
        return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_train_sequences(path: str | Path) -> tuple[np.ndarray, tuple[int, ...]]:
    with np.load(path) as data:
        if "train" not in data or "test" not in data:
            raise ValueError("Processed .npz must contain 'train' and 'test'")
        train = np.asarray(data["train"], dtype=np.float32)
        test_shape = tuple(int(v) for v in data["test"].shape)
    if train.ndim != 3:
        raise ValueError(f"Expected train array shaped [N, seq_len, features], got {train.shape}")
    return train, test_shape


def make_loader(train_sequences: np.ndarray, config: ContrastiveConfig) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        TensorDataset(torch.tensor(train_sequences, dtype=torch.float32)),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )


def _json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _history_payload(history: list[float], epoch_seconds: list[float], seed: int) -> dict[str, np.ndarray]:
    return {
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        "train_loss": np.asarray(history, dtype=np.float32),
        "epoch_seconds": np.asarray(epoch_seconds, dtype=np.float32),
        "seed": np.asarray(seed, dtype=np.int32),
    }


def _write_budget_summary(path: Path, metrics: dict) -> None:
    lines = [
        f"# Contrastive encoder epoch {metrics['epoch']}",
        "",
        "This is an unsupervised pretraining run. The held-out test split is not used for training or model selection.",
        "",
        f"- dataset: `{metrics['dataset_path']}`",
        f"- seed: `{metrics['seed']}`",
        f"- device: `{metrics['device']}`",
        f"- train sequences: `{metrics['train_sequence_count']}`",
        f"- batch size: `{metrics['training_config']['batch_size']}`",
        f"- temperature: `{metrics['training_config']['temperature']}`",
        f"- final train NT-Xent loss: `{metrics['train_loss']:.10f}`",
        f"- best observed train NT-Xent loss: `{metrics['best_train_loss']:.10f}`",
        f"- elapsed seconds: `{metrics['elapsed_seconds']:.2f}`",
        "",
        "Artifacts: `checkpoint.pth`, `history.npz`, `metrics.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_summary(run_root: Path, sweep: list[dict]) -> None:
    lines = [
        "# Contrastive encoder pretraining sweep",
        "",
        "All epoch budgets are reported as a fixed-budget characterization sweep. No checkpoint is selected using test metrics.",
        "",
        "| epoch | train NT-Xent loss | best train loss so far | elapsed seconds |",
        "|---:|---:|---:|---:|",
    ]
    for row in sweep:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.10f} | "
            f"{row['best_train_loss']:.10f} | {row['elapsed_seconds']:.2f} |"
        )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_canonical_checkpoint(checkpoint_path: Path, canonical_path: Path) -> None:
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, canonical_path)


def train_and_snapshot(
    train_sequences: np.ndarray,
    processed_npz: str | Path,
    run_root: Path,
    config: ContrastiveConfig,
    canonical_checkpoint: Path | None = None,
) -> list[dict]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    _, seq_len, input_dim = train_sequences.shape
    loader = make_loader(train_sequences, config)
    if len(loader) == 0:
        raise ValueError("Training split is too small for batch_size with drop_last=True")

    model = ContrastiveEncoder(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[float] = []
    epoch_seconds: list[float] = []
    sweep: list[dict] = []
    budgets = set(config.epoch_budgets)
    started_at = time.time()

    for epoch in range(1, max(config.epoch_budgets) + 1):
        epoch_start = time.time()
        loss = train_contrastive_epoch(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            device=str(device),
            temperature=config.temperature,
        )
        history.append(loss)
        epoch_seconds.append(time.time() - epoch_start)
        print(
            f"contrastive epoch completed: epoch={epoch} train_loss={loss:.8f} "
            f"seconds={epoch_seconds[-1]:.2f}",
            flush=True,
        )

        if epoch not in budgets:
            continue

        budget_dir = run_root / f"e{epoch}"
        budget_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = budget_dir / "checkpoint.pth"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_epoch": epoch,
            "seq_len": int(seq_len),
            "input_dim": int(input_dim),
            "hidden_dim": config.hidden_dim,
            "embedding_dim": config.embedding_dim,
            "training_config": config.to_dict(),
            "seed": config.seed,
            "dataset_path": str(processed_npz),
        }
        torch.save(checkpoint, checkpoint_path)
        np.savez(budget_dir / "history.npz", **_history_payload(history, epoch_seconds, config.seed))

        metrics = {
            "model": "contrastive_encoder",
            "epoch": epoch,
            "seed": config.seed,
            "device": str(device),
            "train_loss": float(history[-1]),
            "best_train_loss": float(min(history)),
            "elapsed_seconds": float(time.time() - started_at),
            "train_sequence_count": int(train_sequences.shape[0]),
            "dataset_path": str(processed_npz),
            "checkpoint_path": str(checkpoint_path),
            "training_config": config.to_dict(),
        }
        _json_write(budget_dir / "metrics.json", metrics)
        _write_budget_summary(budget_dir / "summary.md", metrics)
        sweep.append(metrics)
        print(f"contrastive checkpoint saved: epoch={epoch} path={checkpoint_path}", flush=True)

        if canonical_checkpoint is not None and epoch == max(config.epoch_budgets):
            _copy_canonical_checkpoint(checkpoint_path, canonical_checkpoint)
            print(f"canonical contrastive checkpoint saved: path={canonical_checkpoint}", flush=True)

    _json_write(run_root / "sweep_metrics.json", sweep)
    _write_root_summary(run_root, sweep)
    return sweep


def run_experiment(
    processed_npz: str | Path,
    run_root: Path,
    config: ContrastiveConfig,
    canonical_checkpoint: Path | None = None,
) -> list[dict]:
    train_sequences, test_shape = load_train_sequences(processed_npz)
    run_root.mkdir(parents=True, exist_ok=True)
    _json_write(run_root / "config.json", config.to_dict())
    _json_write(
        run_root / "dataset_manifest.json",
        {
            "processed_npz": str(processed_npz),
            "train_sequence_shape": list(train_sequences.shape),
            "test_sequence_shape": list(test_shape),
            "train_sequence_count": int(train_sequences.shape[0]),
            "test_sequence_count": int(test_shape[0]) if test_shape else 0,
            "test_usage": "not used for contrastive pretraining",
        },
    )
    return train_and_snapshot(train_sequences, processed_npz, run_root, config, canonical_checkpoint)


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _default_run_root(run_name: str) -> Path:
    path = Path(run_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("run-name must be a simple relative directory name")
    return ROOT / "experiments" / "contrastive_encoder" / run_name


def _prepare_run_root(run_root: Path, overwrite: bool) -> None:
    if run_root.exists() and any(run_root.iterdir()):
        if not overwrite:
            raise FileExistsError("run directory exists and is not empty; pass --overwrite")
        shutil.rmtree(run_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain the contrastive encoder on the locked train split.")
    parser.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    parser.add_argument("--run-name", default="contrastive-4h-seq64-top50")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--epoch-budgets", default="15,20,25,50,100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--canonical-checkpoint",
        default=None,
        help="Optional path for a copy of the final budget checkpoint, e.g. checkpoints/contrastive_4h_seq64_top50.pth.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        run_root = Path(args.run_root) if args.run_root else _default_run_root(args.run_name)
        config = ContrastiveConfig(
            epoch_budgets=_parse_int_tuple(args.epoch_budgets),
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            temperature=args.temperature,
            device=args.device,
        )
        canonical_checkpoint = Path(args.canonical_checkpoint) if args.canonical_checkpoint else None
        _prepare_run_root(run_root, args.overwrite)
        run_experiment(args.processed_npz, run_root, config, canonical_checkpoint)
        print(
            "contrastive encoder experiment completed: "
            f"epochs={','.join(str(epoch) for epoch in config.epoch_budgets)} "
            f"run_dir={run_root}",
            flush=True,
        )
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"contrastive encoder experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
