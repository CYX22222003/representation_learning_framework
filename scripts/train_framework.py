# Train and evaluate the framework MVP on downstream tasks.
#
# MVP scope: downstream probing with frozen branch-aware features and simple
# MLP task heads. The script keeps the feature extractors frozen; only the
# aggregator (if learnable) and task head are trained.

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from aggregation.aggregator import RepresentationAggregator
from evaluation.metrics import mse_and_corr, multiclass_classification_metrics, regression_metrics
from features.feature_store import NpzFeatureStore
from tasks.price_prediction import PriceRegressor, build_price_prediction_targets
from tasks.trend_classification import TrendClassifier
from tasks.volatility_labels import load_volatility_label_bundle, validate_volatility_label_bundle
from tasks.volatility_prediction import VolatilityRegressor


@dataclass(frozen=True)
class FrameworkConfig:
    task: str = "price_prediction"
    labels_npz: str | None = None
    mode: str = "concat"
    out_dim: int = 128
    epoch_budgets: tuple[int, ...] = (15, 50, 100)
    seed: int = 0
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    head_hidden_dim: int = 128
    n_classes: int = 3
    price_index: int = 3
    horizon: int = 1
    standardize: bool = True
    standardize_clip: float = 10.0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.task not in {"price_prediction", "trend_classification", "volatility_prediction"}:
            raise ValueError("task must be one of: price_prediction, trend_classification, volatility_prediction")
        if self.task in {"trend_classification", "volatility_prediction"} and not self.labels_npz:
            raise ValueError(f"{self.task} requires --labels-npz")
        if self.mode not in {"concat", "gated"}:
            raise ValueError("mode must be 'concat' or 'gated'")
        if self.out_dim <= 0:
            raise ValueError("out_dim must be positive")
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
        if self.head_hidden_dim <= 0:
            raise ValueError("head_hidden_dim must be positive")
        if self.n_classes <= 1:
            raise ValueError("n_classes must be greater than 1")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if self.standardize_clip < 0.0:
            raise ValueError("standardize_clip must be non-negative")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["epoch_budgets"] = list(self.epoch_budgets)
        return data


class BranchDataset(Dataset):
    def __init__(
        self,
        branches: Mapping[str, np.ndarray],
        targets: np.ndarray,
        target_kind: str,
    ) -> None:
        lengths = {name: len(values) for name, values in branches.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Branch row counts differ: {lengths}")
        if next(iter(lengths.values())) != len(targets):
            raise ValueError(
                f"Feature rows and target rows differ: {lengths} vs targets={len(targets)}"
            )
        self.branch_names = list(branches)
        self.branches = {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in branches.items()
        }
        if target_kind == "regression":
            self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
        elif target_kind == "multiclass":
            self.targets = torch.tensor(targets, dtype=torch.long).reshape(-1)
        else:
            raise ValueError(f"Unsupported target_kind: {target_kind}")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return {name: self.branches[name][idx] for name in self.branch_names}, self.targets[idx]


class FrameworkTaskModel(nn.Module):
    def __init__(
        self,
        branch_dims: Mapping[str, int],
        mode: str,
        out_dim: int,
        head_hidden_dim: int,
        task: str,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.aggregator = RepresentationAggregator(
            dict(branch_dims),
            out_dim=out_dim,
            mode=mode,
        )
        if task == "price_prediction":
            self.head = PriceRegressor(self.aggregator.output_dim, head_hidden_dim)
        elif task == "trend_classification":
            self.head = TrendClassifier(
                self.aggregator.output_dim,
                hidden_dim=head_hidden_dim,
                n_classes=n_classes,
            )
        elif task == "volatility_prediction":
            self.head = VolatilityRegressor(self.aggregator.output_dim, head_hidden_dim)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def forward(self, branch_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        embedding, _ = self.aggregator(branch_inputs)
        return self.head(embedding)


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


def _json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one integer")
    return values


def load_processed_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "train" not in data or "test" not in data:
            raise ValueError("Processed .npz must contain train and test arrays")
        train = np.asarray(data["train"], dtype=np.float32)
        test = np.asarray(data["test"], dtype=np.float32)
    if train.ndim != 3 or test.ndim != 3:
        raise ValueError(f"Expected [N, seq_len, features], got {train.shape}, {test.shape}")
    if train.shape[1:] != test.shape[1:]:
        raise ValueError(f"Train/test sequence shapes differ: {train.shape} vs {test.shape}")
    return train, test


def load_split_feature_branches(
    feature_npz: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    feature_path = Path(feature_npz)
    index_path = Path(f"{feature_path}.index.npz")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing feature split index: {index_path}")
    with np.load(index_path) as data:
        train_size = int(data["train_size"])
        test_size = int(data["test_size"])
    bundle = NpzFeatureStore(str(feature_path)).load()
    branches = bundle.as_branch_dict()
    train: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}
    for name, values in branches.items():
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Feature branch {name!r} must be 2D, got {values.shape}")
        if len(values) != train_size + test_size:
            raise ValueError(
                f"Branch {name!r} rows={len(values)} do not match split sizes "
                f"{train_size}+{test_size}"
            )
        train[name] = values[:train_size]
        test[name] = values[train_size:]
    return train, test, {"train_size": train_size, "test_size": test_size}


def fit_standardizer(branches: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for name, values in branches.items():
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = values.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        stats[name] = {"mean": mean, "std": std}
    return stats


def apply_standardizer(
    branches: Mapping[str, np.ndarray],
    stats: Mapping[str, Mapping[str, np.ndarray]],
    clip: float,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, values in branches.items():
        scaled = (values - stats[name]["mean"]) / stats[name]["std"]
        if clip > 0.0:
            scaled = np.clip(scaled, -clip, clip)
        out[name] = scaled.astype(np.float32)
    return out


def save_standardizer(path: Path, stats: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    payload = {}
    for name, branch_stats in stats.items():
        payload[f"{name}__mean"] = branch_stats["mean"]
        payload[f"{name}__std"] = branch_stats["std"]
    np.savez(path, **payload)


def slice_for_horizon(
    branches: Mapping[str, np.ndarray],
    horizon: int,
) -> dict[str, np.ndarray]:
    return {name: values[:-horizon] for name, values in branches.items()}


def select_branch_rows(
    branches: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {name: values[indices] for name, values in branches.items()}


def target_kind(task: str) -> str:
    if task == "price_prediction":
        return "regression"
    if task == "trend_classification":
        return "multiclass"
    if task == "volatility_prediction":
        return "regression"
    raise ValueError(f"Unsupported task: {task}")


def build_price_data(
    train_sequences: np.ndarray,
    test_sequences: np.ndarray,
    train_branches: Mapping[str, np.ndarray],
    test_branches: Mapping[str, np.ndarray],
    config: FrameworkConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    _, y_train = build_price_prediction_targets(
        train_sequences,
        price_index=config.price_index,
        horizon=config.horizon,
    )
    _, y_test = build_price_prediction_targets(
        test_sequences,
        price_index=config.price_index,
        horizon=config.horizon,
    )
    return (
        slice_for_horizon(train_branches, config.horizon),
        y_train,
        slice_for_horizon(test_branches, config.horizon),
        y_test,
    )


def load_trend_labels(
    labels_npz: str | Path,
    train_size: int,
    test_size: int,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    label_path = Path(labels_npz)
    if not label_path.exists():
        raise FileNotFoundError(f"Missing trend label bundle: {label_path}")
    with np.load(label_path) as data:
        required = {"train_indices", "train_labels", "test_indices", "test_labels", "class_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Trend label bundle missing keys: {sorted(missing)}")
        train_indices = np.asarray(data["train_indices"], dtype=np.int64)
        train_labels = np.asarray(data["train_labels"], dtype=np.int64)
        test_indices = np.asarray(data["test_indices"], dtype=np.int64)
        test_labels = np.asarray(data["test_labels"], dtype=np.int64)
        class_names = [str(item) for item in data["class_names"].tolist()]

    if len(class_names) != n_classes:
        raise ValueError(f"Label bundle has {len(class_names)} classes, config has {n_classes}")
    if len(train_indices) != len(train_labels) or len(test_indices) != len(test_labels):
        raise ValueError("Trend label index/label row counts differ")
    if np.any(train_indices < 0) or np.any(train_indices >= train_size):
        raise ValueError("Trend train indices are outside the feature train split")
    if np.any(test_indices < 0) or np.any(test_indices >= test_size):
        raise ValueError("Trend test indices are outside the feature test split")
    if np.any(train_labels < 0) or np.any(train_labels >= n_classes):
        raise ValueError("Trend train labels are outside the configured class range")
    if np.any(test_labels < 0) or np.any(test_labels >= n_classes):
        raise ValueError("Trend test labels are outside the configured class range")

    manifest_path = Path(f"{label_path}.manifest.json")
    manifest: dict[str, object] = {
        "labels_npz": str(label_path),
        "class_names": class_names,
    }
    if manifest_path.exists():
        manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))

    return train_indices, train_labels, test_indices, test_labels, manifest


def build_trend_data(
    train_branches: Mapping[str, np.ndarray],
    test_branches: Mapping[str, np.ndarray],
    feature_index: Mapping[str, int],
    config: FrameworkConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    if not config.labels_npz:
        raise ValueError("trend_classification requires a label bundle")
    train_indices, y_train, test_indices, y_test, label_manifest = load_trend_labels(
        config.labels_npz,
        train_size=feature_index["train_size"],
        test_size=feature_index["test_size"],
        n_classes=config.n_classes,
    )
    return (
        select_branch_rows(train_branches, train_indices),
        y_train,
        select_branch_rows(test_branches, test_indices),
        y_test,
        label_manifest,
    )


def build_volatility_data(
    train_branches: Mapping[str, np.ndarray],
    test_branches: Mapping[str, np.ndarray],
    feature_index: Mapping[str, int],
    config: FrameworkConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    """Select only contract-safe rows from the saved realised-volatility bundle."""
    if not config.labels_npz:
        raise ValueError("volatility_prediction requires a label bundle")
    bundle, label_manifest = load_volatility_label_bundle(config.labels_npz)
    validate_volatility_label_bundle(
        bundle,
        train_size=feature_index["train_size"],
        test_size=feature_index["test_size"],
        metadata=label_manifest,
    )
    train_indices = np.asarray(bundle["train_row_indices"], dtype=np.int64)
    test_indices = np.asarray(bundle["test_row_indices"], dtype=np.int64)
    y_train = np.asarray(bundle["train_labels"], dtype=np.float32)
    y_test = np.asarray(bundle["test_labels"], dtype=np.float32)
    return (
        select_branch_rows(train_branches, train_indices),
        y_train,
        select_branch_rows(test_branches, test_indices),
        y_test,
        dict(label_manifest),
    )


def make_loader(
    branches: Mapping[str, np.ndarray],
    targets: np.ndarray,
    config: FrameworkConfig,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        BranchDataset(branches, targets, target_kind=target_kind(config.task)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )


def make_model(branch_dims: Mapping[str, int], config: FrameworkConfig) -> FrameworkTaskModel:
    return FrameworkTaskModel(
        branch_dims=branch_dims,
        mode=config.mode,
        out_dim=config.out_dim,
        head_hidden_dim=config.head_hidden_dim,
        task=config.task,
        n_classes=config.n_classes,
    )


def make_criterion(config: FrameworkConfig) -> nn.Module:
    if config.task in {"price_prediction", "volatility_prediction"}:
        return nn.MSELoss()
    if config.task == "trend_classification":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported task: {config.task}")


def _to_device(branches: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: values.to(device) for name, values in branches.items()}


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
    for branches, targets in loader:
        branches = _to_device(branches, device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(branches)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    branches: Mapping[str, np.ndarray],
    targets: np.ndarray,
    config: FrameworkConfig,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    loader = DataLoader(
        BranchDataset(branches, targets, target_kind=target_kind(config.task)),
        batch_size=config.batch_size,
        shuffle=False,
    )
    preds = []
    y = []
    for batch_branches, batch_targets in loader:
        pred = model(_to_device(batch_branches, device)).cpu()
        preds.append(pred)
        y.append(batch_targets)
    pred_tensor = torch.cat(preds, dim=0)
    target_tensor = torch.cat(y, dim=0)
    if config.task == "trend_classification":
        metrics = multiclass_classification_metrics(
            pred_tensor,
            target_tensor,
            n_classes=config.n_classes,
        )
        probabilities = torch.softmax(pred_tensor, dim=1)
        predictions = torch.argmax(pred_tensor, dim=1)
        return {
            "metrics": metrics,
            "logits": pred_tensor.numpy().astype(np.float32),
            "probabilities": probabilities.numpy().astype(np.float32),
            "predictions": predictions.numpy().astype(np.int64),
            "targets": target_tensor.numpy().astype(np.int64),
        }

    metrics = regression_metrics(pred_tensor, target_tensor)
    metrics.update(mse_and_corr(pred_tensor, target_tensor))
    return {
        "metrics": metrics,
        "predictions": pred_tensor.reshape(-1).numpy().astype(np.float32),
        "targets": target_tensor.reshape(-1).numpy().astype(np.float32),
    }


def _history_arrays(history: list[float], seed: int) -> dict[str, np.ndarray]:
    return {
        "train_loss": np.asarray(history, dtype=np.float32),
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        "seed": np.asarray(seed, dtype=np.int32),
    }


def _write_budget_summary(path: Path, metrics: dict, final_train_loss: float) -> None:
    task_title = metrics["task"].replace("_", " ")
    lines = [
        f"# Framework {task_title} epoch {metrics['epoch']}",
        "",
        "Frozen feature branches are probed with the framework aggregator and a simple MLP head.",
        "",
        f"- processed dataset: `{metrics['processed_npz']}`",
        f"- feature store: `{metrics['features_npz']}`",
        f"- seed: `{metrics['seed']}`",
        f"- mode: `{metrics['mode']}`",
        f"- train samples: `{metrics['train_sample_count']}`",
        f"- test samples: `{metrics['test_sample_count']}`",
        f"- final train loss: `{final_train_loss:.10f}`",
    ]
    if metrics["task"] == "trend_classification":
        lines += [
            f"- accuracy: `{metrics['accuracy']:.10f}`",
            f"- macro-F1: `{metrics['macro_f1']:.10f}`",
            f"- weighted-F1: `{metrics['weighted_f1']:.10f}`",
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
    if metrics["task"] == "trend_classification":
        lines[-1] = (
            "Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, "
            "`confusion_matrix.npz`, `metrics.json`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_summary(run_root: Path, sweep: list[dict]) -> None:
    task = sweep[0]["task"] if sweep else "framework"
    task_title = task.replace("_", " ")
    lines = [
        f"# Framework {task_title} sweep",
        "",
        "All epoch budgets are reported as a characterization sweep; no checkpoint is selected from test performance.",
        "",
    ]
    if task == "trend_classification":
        lines += [
            "| epoch | accuracy | macro_f1 | weighted_f1 | train_loss |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in sweep:
            lines.append(
                f"| {row['epoch']} | {row['accuracy']:.10f} | {row['macro_f1']:.10f} | "
                f"{row['weighted_f1']:.10f} | {row['train_loss']:.10f} |"
            )
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
    train_branches: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    branch_dims: Mapping[str, int],
    run_root: Path,
    processed_npz: str | Path,
    features_npz: str | Path,
    config: FrameworkConfig,
) -> list[Path]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    model = make_model(branch_dims, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = make_criterion(config)
    loader = make_loader(train_branches, y_train, config)
    history: list[float] = []
    checkpoints: list[Path] = []
    budgets = set(config.epoch_budgets)
    for epoch in range(1, max(config.epoch_budgets) + 1):
        train_loss = train_one_epoch(model, loader, optimizer, criterion, device)
        history.append(train_loss)
        print(f"framework epoch completed: epoch={epoch} train_loss={train_loss:.10f}", flush=True)
        if epoch not in budgets:
            continue

        budget_dir = run_root / f"e{epoch}"
        budget_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = budget_dir / "checkpoint.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "completed_epoch": epoch,
                "branch_dims": dict(branch_dims),
                "training_config": config.to_dict(),
                "seed": config.seed,
                "processed_npz": str(processed_npz),
                "features_npz": str(features_npz),
            },
            checkpoint_path,
        )
        np.savez(budget_dir / "history.npz", **_history_arrays(history, config.seed))
        checkpoints.append(checkpoint_path)
        print(f"framework checkpoint saved: epoch={epoch} path={checkpoint_path}", flush=True)
    return checkpoints


def evaluate_snapshots(
    checkpoint_paths: list[Path],
    test_branches: Mapping[str, np.ndarray],
    y_test: np.ndarray,
    train_sample_count: int,
    run_root: Path,
    config: FrameworkConfig,
) -> list[dict]:
    device = resolve_device(config.device)
    sweep: list[dict] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        loaded_config = FrameworkConfig(**checkpoint["training_config"])
        model = make_model(checkpoint["branch_dims"], loaded_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        result = evaluate_model(model, test_branches, y_test, loaded_config, device)
        budget_dir = checkpoint_path.parent
        if loaded_config.task == "trend_classification":
            np.savez(
                budget_dir / "predictions.npz",
                preds=result["predictions"],
                targets=result["targets"],
                logits=result["logits"],
                probabilities=result["probabilities"],
            )
            np.savez(
                budget_dir / "confusion_matrix.npz",
                confusion_matrix=np.asarray(result["metrics"]["confusion_matrix"], dtype=np.int64),
            )
        else:
            np.savez(
                budget_dir / "predictions.npz",
                preds=result["predictions"],
                targets=result["targets"],
            )
        with np.load(budget_dir / "history.npz") as history_npz:
            train_loss = float(history_npz["train_loss"][-1])
        metrics = dict(result["metrics"])
        metrics.update(
            {
                "task": loaded_config.task,
                "epoch": int(checkpoint["completed_epoch"]),
                "seed": int(checkpoint["seed"]),
                "mode": loaded_config.mode,
                "train_loss": train_loss,
                "train_sample_count": int(train_sample_count),
                "test_sample_count": int(y_test.shape[0]),
                "processed_npz": checkpoint["processed_npz"],
                "features_npz": checkpoint["features_npz"],
                "labels_npz": loaded_config.labels_npz,
                "branch_dims": checkpoint["branch_dims"],
                "training_config": checkpoint["training_config"],
            }
        )
        _json_write(budget_dir / "metrics.json", metrics)
        _write_budget_summary(budget_dir / "summary.md", metrics, train_loss)
        sweep.append(metrics)
        print(
            f"framework checkpoint evaluated: epoch={metrics['epoch']} metrics={budget_dir / 'metrics.json'}",
            flush=True,
        )
    _json_write(run_root / "sweep_metrics.json", sweep)
    _write_root_summary(run_root, sweep)
    return sweep


def run_experiment(
    processed_npz: str | Path,
    features_npz: str | Path,
    run_root: Path,
    config: FrameworkConfig,
) -> list[dict]:
    train_sequences, test_sequences = load_processed_npz(processed_npz)
    train_raw, test_raw, feature_index = load_split_feature_branches(features_npz)
    if feature_index["train_size"] != len(train_sequences) or feature_index["test_size"] != len(test_sequences):
        raise ValueError(
            "Feature split index does not match processed sequence split: "
            f"{feature_index} vs train={len(train_sequences)} test={len(test_sequences)}"
        )

    label_manifest: dict[str, object] | None = None
    if config.task == "price_prediction":
        train_branches, y_train, test_branches, y_test = build_price_data(
            train_sequences=train_sequences,
            test_sequences=test_sequences,
            train_branches=train_raw,
            test_branches=test_raw,
            config=config,
        )
    elif config.task == "trend_classification":
        train_branches, y_train, test_branches, y_test, label_manifest = build_trend_data(
            train_branches=train_raw,
            test_branches=test_raw,
            feature_index=feature_index,
            config=config,
        )
    elif config.task == "volatility_prediction":
        train_branches, y_train, test_branches, y_test, label_manifest = build_volatility_data(
            train_branches=train_raw,
            test_branches=test_raw,
            feature_index=feature_index,
            config=config,
        )
    else:
        raise ValueError(f"Unsupported task: {config.task}")
    branch_dims = {name: values.shape[1] for name, values in train_branches.items()}
    run_root.mkdir(parents=True, exist_ok=True)

    if config.standardize:
        standardizer = fit_standardizer(train_branches)
        train_branches = apply_standardizer(
            train_branches,
            standardizer,
            clip=config.standardize_clip,
        )
        test_branches = apply_standardizer(
            test_branches,
            standardizer,
            clip=config.standardize_clip,
        )
        save_standardizer(run_root / "feature_standardizer.npz", standardizer)
    else:
        standardizer = None

    _json_write(run_root / "config.json", config.to_dict())
    _json_write(
        run_root / "dataset_manifest.json",
        {
            "processed_npz": str(processed_npz),
            "features_npz": str(features_npz),
            "feature_index": feature_index,
            "train_sequence_shape": list(train_sequences.shape),
            "test_sequence_shape": list(test_sequences.shape),
            "train_sample_count": int(y_train.shape[0]),
            "test_sample_count": int(y_test.shape[0]),
            "branch_dims": branch_dims,
            "task": config.task,
            "labels_npz": config.labels_npz,
            "label_manifest": label_manifest,
            "n_classes": config.n_classes if config.task == "trend_classification" else None,
            "horizon": config.horizon,
            "price_index": config.price_index,
            "standardized": bool(standardizer is not None),
            "standardize_clip": config.standardize_clip if standardizer is not None else None,
            "standardizer": "feature_standardizer.npz" if standardizer is not None else None,
        },
    )
    checkpoint_paths = train_and_snapshot(
        train_branches=train_branches,
        y_train=y_train,
        branch_dims=branch_dims,
        run_root=run_root,
        processed_npz=processed_npz,
        features_npz=features_npz,
        config=config,
    )
    return evaluate_snapshots(
        checkpoint_paths=checkpoint_paths,
        test_branches=test_branches,
        y_test=y_test,
        train_sample_count=len(y_train),
        run_root=run_root,
        config=config,
    )


def _default_run_root(task: str, run_name: str) -> Path:
    path = Path(run_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("run-name must be a simple relative directory name")
    return ROOT / "experiments" / "framework" / task / run_name


def _prepare_run_root(run_root: Path, overwrite: bool) -> None:
    if run_root.exists() and any(run_root.iterdir()):
        if not overwrite:
            raise FileExistsError("run directory exists and is not empty; pass --overwrite")
        shutil.rmtree(run_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the framework MVP task head.")
    parser.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    parser.add_argument("--features-npz", default="data/features/features_4h_seq64_top50.npz")
    parser.add_argument("--run-name", default="4h_stat_transform_vae_contrastive_concat")
    parser.add_argument("--run-root", default=None)
    parser.add_argument(
        "--task",
        choices=("price_prediction", "trend_classification", "volatility_prediction"),
        default="price_prediction",
    )
    parser.add_argument(
        "--labels-npz",
        default=None,
        help="Required for trend/volatility; use the saved split-safe task label bundle.",
    )
    parser.add_argument("--mode", choices=("concat", "gated"), default="concat")
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--epoch-budgets", default="15,50,100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--price-index", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument(
        "--standardize-clip",
        type=float,
        default=10.0,
        help="Clip standardized branch values to +/- this bound; 0 disables clipping.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = FrameworkConfig(
            task=args.task,
            labels_npz=args.labels_npz,
            mode=args.mode,
            out_dim=args.out_dim,
            epoch_budgets=_parse_int_tuple(args.epoch_budgets),
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            head_hidden_dim=args.head_hidden_dim,
            n_classes=args.n_classes,
            price_index=args.price_index,
            horizon=args.horizon,
            standardize=not args.no_standardize,
            standardize_clip=args.standardize_clip,
            device=args.device,
        )
        run_root = Path(args.run_root) if args.run_root else _default_run_root(config.task, args.run_name)
        _prepare_run_root(run_root, args.overwrite)
        run_experiment(args.processed_npz, args.features_npz, run_root, config)
        print(
            f"framework experiment completed: task={config.task} "
            f"epochs={','.join(str(epoch) for epoch in config.epoch_budgets)} "
            f"run_dir={run_root}",
            flush=True,
        )
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"framework experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
