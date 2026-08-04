from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baselines.mlp_baseline.mlp_model import RawOHLCVMLP
from evaluation.metrics import classification_metrics, mse_and_corr, regression_metrics
from tasks.price_prediction import PriceRegressor, build_price_prediction_targets
from tasks.trend_classification import TrendClassifier, build_trend_labels
from tasks.volatility_prediction import VolatilityRegressor, build_volatility_targets


@dataclass(frozen=True)
class ExperimentConfig:
    task: str = "price"
    epoch_budgets: tuple[int, ...] = (15, 20, 25, 50, 100)
    seed: int = 0
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256, 128)
    encoder_output_dim: int = 128
    head_hidden_dim: int = 128
    dropout: float = 0.1
    price_index: int = 3
    horizon: int = 1
    trend_threshold: float = 0.0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.task not in {"price", "volatility", "trend"}:
            raise ValueError("task must be one of: price, volatility, trend")
        if any(epoch <= 0 for epoch in self.epoch_budgets):
            raise ValueError("epoch budgets must be positive")
        if tuple(sorted(set(self.epoch_budgets))) != tuple(self.epoch_budgets):
            raise ValueError("epoch budgets must be unique and sorted")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions")
        if self.encoder_output_dim <= 0:
            raise ValueError("encoder_output_dim must be positive")
        if self.head_hidden_dim <= 0:
            raise ValueError("head_hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["epoch_budgets"] = list(self.epoch_budgets)
        data["hidden_dims"] = list(self.hidden_dims)
        return data


class MLPBaseline(nn.Module):
    def __init__(self, seq_len: int, n_features: int, config: ExperimentConfig) -> None:
        super().__init__()
        self.encoder = RawOHLCVMLP(
            seq_len=seq_len,
            n_features=n_features,
            hidden_dims=list(config.hidden_dims),
            output_dim=config.encoder_output_dim,
            dropout=config.dropout,
        )
        if config.task == "price":
            self.head = PriceRegressor(self.encoder.output_dim, config.head_hidden_dim)
        elif config.task == "volatility":
            self.head = VolatilityRegressor(self.encoder.output_dim, config.head_hidden_dim)
        elif config.task == "trend":
            self.head = TrendClassifier(self.encoder.output_dim, config.head_hidden_dim)
        else:
            raise ValueError(f"Unsupported task: {config.task}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


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


def load_processed_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        train = np.asarray(data["train"], dtype=np.float32)
        test = np.asarray(data["test"], dtype=np.float32)
    if train.ndim != 3 or test.ndim != 3:
        raise ValueError(f"Expected train/test arrays shaped [N, seq_len, features], got {train.shape}, {test.shape}")
    if train.shape[1:] != test.shape[1:]:
        raise ValueError(f"Train/test sequence shapes differ: {train.shape[1:]} vs {test.shape[1:]}")
    return train, test


def build_task_data(sequences: np.ndarray, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.task == "price":
        return build_price_prediction_targets(sequences, config.price_index, config.horizon)
    if config.task == "volatility":
        return build_volatility_targets(sequences, config.price_index, config.horizon)
    if config.task == "trend":
        return build_trend_labels(sequences, config.price_index, config.horizon, config.trend_threshold)
    raise ValueError(f"Unsupported task: {config.task}")


def make_loader(X: np.ndarray, y: np.ndarray, config: ExperimentConfig) -> DataLoader:
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )


def make_model(seq_len: int, n_features: int, config: ExperimentConfig) -> MLPBaseline:
    return MLPBaseline(seq_len, n_features, config)


def loss_fn(config: ExperimentConfig) -> nn.Module:
    if config.task == "trend":
        return nn.BCEWithLogitsLoss()
    return nn.MSELoss()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        batch_size = x_batch.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    preds = []
    targets = []
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)),
        batch_size=config.batch_size,
        shuffle=False,
    )
    for x_batch, y_batch in loader:
        logits_or_pred = model(x_batch.to(device)).cpu()
        preds.append(logits_or_pred)
        targets.append(y_batch)
    pred_tensor = torch.cat(preds, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    if config.task == "trend":
        metrics = classification_metrics(pred_tensor, target_tensor)
        pred_array = (torch.sigmoid(pred_tensor.reshape(-1)) >= 0.5).to(torch.int64).numpy()
        return {
            "metrics": metrics,
            "logits": pred_tensor.reshape(-1).numpy().astype(np.float32),
            "predictions": pred_array,
            "targets": target_tensor.reshape(-1).numpy().astype(np.float32),
        }

    metrics = regression_metrics(pred_tensor, target_tensor)
    metrics.update(mse_and_corr(pred_tensor, target_tensor))
    return {
        "metrics": metrics,
        "predictions": pred_tensor.reshape(-1).numpy().astype(np.float32),
        "targets": target_tensor.reshape(-1).numpy().astype(np.float32),
    }


def _json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _history_arrays(history: list[float], seed: int) -> dict[str, np.ndarray]:
    return {
        "train_loss": np.asarray(history, dtype=np.float32),
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        "seed": np.asarray(seed, dtype=np.int32),
    }


def _write_budget_summary(path: Path, metrics: dict, final_train_loss: float) -> None:
    lines = [
        f"# Raw-OHLCV MLP {metrics['task']} epoch {metrics['epoch']}",
        "",
        f"- dataset: `{metrics['dataset_path']}`",
        f"- seed: `{metrics['seed']}`",
        f"- train samples: `{metrics['train_sample_count']}`",
        f"- test samples: `{metrics['test_sample_count']}`",
        f"- final train loss: `{final_train_loss:.10f}`",
    ]
    if metrics["task"] == "trend":
        lines += [
            f"- accuracy: `{metrics['accuracy']:.10f}`",
            f"- f1: `{metrics['f1']:.10f}`",
        ]
    else:
        corr = "nan" if np.isnan(metrics["corr"]) else f"{metrics['corr']:.10f}"
        lines += [
            f"- MAE: `{metrics['mae']:.10f}`",
            f"- RMSE: `{metrics['rmse']:.10f}`",
            f"- MSE: `{metrics['mse']:.10f}`",
            f"- Pearson correlation: `{corr}`",
        ]
    lines += ["", "Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_summary(run_root: Path, sweep: list[dict], task: str) -> None:
    lines = [
        f"# Raw-OHLCV MLP {task} sweep",
        "",
        "All epoch budgets are reported as a characterization sweep; no checkpoint is selected from test performance.",
        "",
    ]
    if task == "trend":
        lines += [
            "| epoch | accuracy | f1 | train_loss |",
            "|---:|---:|---:|---:|",
        ]
        for row in sweep:
            lines.append(f"| {row['epoch']} | {row['accuracy']:.10f} | {row['f1']:.10f} | {row['train_loss']:.10f} |")
    else:
        lines += [
            "| epoch | mae | rmse | mse | corr | train_loss |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in sweep:
            corr = "nan" if np.isnan(row["corr"]) else f"{row['corr']:.10f}"
            lines.append(
                f"| {row['epoch']} | {row['mae']:.10f} | {row['rmse']:.10f} | "
                f"{row['mse']:.10f} | {corr} | {row['train_loss']:.10f} |"
            )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_and_snapshot(
    X_train: np.ndarray,
    y_train: np.ndarray,
    run_root: Path,
    dataset_path: str,
    config: ExperimentConfig,
) -> list[Path]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    model = make_model(X_train.shape[1], X_train.shape[2], config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = loss_fn(config)
    loader = make_loader(X_train, y_train, config)
    history: list[float] = []
    checkpoints = []
    budgets = set(config.epoch_budgets)
    for epoch in range(1, max(config.epoch_budgets) + 1):
        history.append(train_one_epoch(model, loader, optimizer, criterion, device))
        if epoch in budgets:
            budget_dir = run_root / f"e{epoch}"
            budget_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = budget_dir / "checkpoint.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "completed_epoch": epoch,
                    "seq_len": X_train.shape[1],
                    "n_features": X_train.shape[2],
                    "training_config": config.to_dict(),
                    "seed": config.seed,
                    "dataset_path": dataset_path,
                },
                checkpoint_path,
            )
            np.savez(budget_dir / "history.npz", **_history_arrays(history, config.seed))
            checkpoints.append(checkpoint_path)
    return checkpoints


def evaluate_snapshots(
    checkpoint_paths: list[Path],
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_sample_count: int,
    run_root: Path,
    config: ExperimentConfig,
) -> list[dict]:
    device = resolve_device(config.device)
    sweep = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        loaded_config = ExperimentConfig(**checkpoint["training_config"])
        model = make_model(checkpoint["seq_len"], checkpoint["n_features"], loaded_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        result = evaluate_model(model, X_test, y_test, loaded_config, device)
        budget_dir = checkpoint_path.parent
        prediction_payload = {
            "preds": result["predictions"],
            "targets": result["targets"],
        }
        if "logits" in result:
            prediction_payload["logits"] = result["logits"]
        np.savez(budget_dir / "predictions.npz", **prediction_payload)
        with np.load(budget_dir / "history.npz") as history_npz:
            train_loss = float(history_npz["train_loss"][-1])
        metrics = dict(result["metrics"])
        metrics.update(
            {
                "task": loaded_config.task,
                "epoch": int(checkpoint["completed_epoch"]),
                "seed": int(checkpoint["seed"]),
                "train_loss": train_loss,
                "train_sample_count": int(train_sample_count),
                "test_sample_count": int(X_test.shape[0]),
                "dataset_path": checkpoint["dataset_path"],
                "training_config": checkpoint["training_config"],
            }
        )
        _json_write(budget_dir / "metrics.json", metrics)
        _write_budget_summary(budget_dir / "summary.md", metrics, train_loss)
        sweep.append(metrics)
    _json_write(run_root / "sweep_metrics.json", sweep)
    _write_root_summary(run_root, sweep, config.task)
    return sweep


def run_experiment(processed_npz: str | Path, run_root: Path, config: ExperimentConfig) -> list[dict]:
    train_sequences, test_sequences = load_processed_npz(processed_npz)
    X_train, y_train = build_task_data(train_sequences, config)
    X_test, y_test = build_task_data(test_sequences, config)
    run_root.mkdir(parents=True, exist_ok=True)
    _json_write(run_root / "config.json", config.to_dict())
    _json_write(
        run_root / "dataset_manifest.json",
        {
            "processed_npz": str(processed_npz),
            "train_sequence_shape": list(train_sequences.shape),
            "test_sequence_shape": list(test_sequences.shape),
            "train_sample_count": int(X_train.shape[0]),
            "test_sample_count": int(X_test.shape[0]),
            "task": config.task,
            "horizon": config.horizon,
            "price_index": config.price_index,
        },
    )
    checkpoint_paths = train_and_snapshot(X_train, y_train, run_root, str(processed_npz), config)
    return evaluate_snapshots(checkpoint_paths, X_test, y_test, len(X_train), run_root, config)


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _default_run_root(run_name: str) -> Path:
    path = Path(run_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("run-name must be a simple relative directory name")
    return Path(__file__).resolve().parent / "experiments" / run_name


def _prepare_run_root(run_root: Path, overwrite: bool) -> None:
    if run_root.exists() and any(run_root.iterdir()):
        if not overwrite:
            raise FileExistsError("run directory exists and is not empty; pass --overwrite")
        shutil.rmtree(run_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Raw-OHLCV MLP baseline experiment.")
    parser.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    parser.add_argument("--run-name", default="2026-08-04-v1")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--task", choices=("price", "volatility", "trend"), default="price")
    parser.add_argument("--epoch-budgets", default="15,20,25,50,100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dims", default="512,512,256,256,128")
    parser.add_argument("--encoder-output-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--price-index", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--trend-threshold", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        run_root = Path(args.run_root) if args.run_root else _default_run_root(args.run_name)
        config = ExperimentConfig(
            task=args.task,
            epoch_budgets=_parse_int_tuple(args.epoch_budgets),
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dims=_parse_int_tuple(args.hidden_dims),
            encoder_output_dim=args.encoder_output_dim,
            head_hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
            price_index=args.price_index,
            horizon=args.horizon,
            trend_threshold=args.trend_threshold,
            device=args.device,
        )
        _prepare_run_root(run_root, args.overwrite)
        run_experiment(args.processed_npz, run_root, config)
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"MLP baseline experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
