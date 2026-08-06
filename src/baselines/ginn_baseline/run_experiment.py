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

from baselines.ginn_baseline.ginn_data import GinnDataConfig, GinnCache, load_and_validate_cache
from baselines.ginn_baseline.ginn_model import (
    GinnTensorDataset,
    LSTMVariancePredictor,
    evaluate_model,
    make_train_loader,
    train_one_epoch,
)


@dataclass(frozen=True)
class ExperimentConfig:
    epoch_budgets: tuple[int, ...] = (15, 20, 25, 50, 100)
    seed: int = 0
    batch_size: int = 64
    learning_rate: float = 1e-4
    lambda_garch: float = 0.3
    input_size: int = 5
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    output_transform: str = "linear"
    device: str = "auto"

    def __post_init__(self) -> None:
        if any(epoch <= 0 for epoch in self.epoch_budgets):
            raise ValueError("epoch budgets must be positive")
        if tuple(sorted(set(self.epoch_budgets))) != tuple(self.epoch_budgets):
            raise ValueError("epoch budgets must be unique and sorted")
        if self.batch_size <= 1:
            raise ValueError("batch_size must be greater than 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.lambda_garch <= 1.0:
            raise ValueError("lambda_garch must be in [0, 1]")
        if self.output_transform not in {"linear", "softplus"}:
            raise ValueError("output_transform must be 'linear' or 'softplus'")

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


def model_config(config: ExperimentConfig) -> dict[str, int | float]:
    return {
        "input_size": config.input_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "output_transform": config.output_transform,
    }


def _new_model(config_dict: dict) -> LSTMVariancePredictor:
    return LSTMVariancePredictor(
        input_size=int(config_dict["input_size"]),
        hidden_size=int(config_dict["hidden_size"]),
        num_layers=int(config_dict["num_layers"]),
        dropout=float(config_dict["dropout"]),
        output_transform=str(config_dict.get("output_transform", "linear")),
    )


def _json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _history_arrays(history: dict[str, list[float]], seed: int) -> dict[str, np.ndarray]:
    epochs = np.arange(1, len(history["total"]) + 1, dtype=np.int32)
    return {
        "train_total_loss": np.asarray(history["total"], dtype=np.float32),
        "train_gt_mse": np.asarray(history["gt_mse"], dtype=np.float32),
        "train_garch_mse": np.asarray(history["garch_mse"], dtype=np.float32),
        "epochs": epochs,
        "seed": np.asarray(seed, dtype=np.int32),
    }


def _write_budget_summary(path: Path, metrics: dict, history: dict[str, list[float]]) -> None:
    lines = [
        f"# GINN epoch {metrics['epoch']}",
        "",
        f"- dataset_id: `{metrics['dataset_id']}`",
        f"- seed: `{metrics['seed']}`",
        f"- samples: `{metrics['sample_count']}`",
        f"- MSE: `{metrics['mse']:.10f}`",
        f"- Pearson correlation: `{metrics['pearson_corr']}`",
        f"- MAE: `{metrics['mae']:.10f}`",
        f"- RMSE: `{metrics['rmse']:.10f}`",
        f"- final fused loss: `{history['total'][-1]:.10f}`",
        f"- negative prediction fraction: `{metrics['negative_prediction_fraction']:.10f}`",
        "",
        "Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_summary(run_root: Path, sweep: list[dict]) -> None:
    lines = [
        "# GINN baseline sweep",
        "",
        "All epoch budgets are reported as a characterization sweep; no checkpoint is selected from test performance.",
        "",
        "| epoch | mse | pearson_corr | mae | rmse | negative_prediction_fraction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sweep:
        corr = "null" if row["pearson_corr"] is None else f"{row['pearson_corr']:.10f}"
        lines.append(
            f"| {row['epoch']} | {row['mse']:.10f} | {corr} | {row['mae']:.10f} | "
            f"{row['rmse']:.10f} | {row['negative_prediction_fraction']:.10f} |"
        )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_and_snapshot(
    X_train: np.ndarray,
    y_gt_train: np.ndarray,
    y_garch_train: np.ndarray,
    dataset_id: str,
    run_root: Path,
    config: ExperimentConfig,
) -> list[Path]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    model = LSTMVariancePredictor(**model_config(config))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    dataset = GinnTensorDataset(X_train, y_gt_train, y_garch_train)
    loader = make_train_loader(dataset, batch_size=config.batch_size, seed=config.seed)
    history = {"total": [], "gt_mse": [], "garch_mse": []}
    checkpoints = []
    budgets = set(config.epoch_budgets)
    for epoch in range(1, max(config.epoch_budgets) + 1):
        losses = train_one_epoch(model, loader, optimizer, device, config.lambda_garch)
        for key, value in losses.items():
            history[key].append(value)
        if epoch in budgets:
            budget_dir = run_root / f"e{epoch}"
            budget_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = budget_dir / "checkpoint.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "completed_epoch": epoch,
                    "model_config": model_config(config),
                    "training_config": config.to_dict(),
                    "seed": config.seed,
                    "dataset_id": dataset_id,
                },
                checkpoint_path,
            )
            np.savez(budget_dir / "history.npz", **_history_arrays(history, config.seed))
            checkpoints.append(checkpoint_path)
    return checkpoints


def evaluate_snapshots(
    checkpoint_paths: list[Path],
    cache: GinnCache,
    run_root: Path,
    config: ExperimentConfig,
) -> list[dict]:
    device = resolve_device(config.device)
    sweep = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = _new_model(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        result = evaluate_model(
            model,
            cache.X_test,
            cache.y_gt_test,
            cache.y_garch_test,
            cache.contract_id_test,
            device,
            batch_size=config.batch_size,
        )
        budget_dir = checkpoint_path.parent
        np.savez(
            budget_dir / "predictions.npz",
            preds=result["predictions"],
            targets=result["targets"],
            garch_targets=result["garch_targets"],
            contract_ids=result["contract_ids"],
        )
        metrics = dict(result["metrics"])
        metrics.update(
            {
                "epoch": int(checkpoint["completed_epoch"]),
                "seed": int(checkpoint["seed"]),
                "sample_count": int(cache.X_test.shape[0]),
                "dataset_id": checkpoint["dataset_id"],
                "model_config": checkpoint["model_config"],
                "training_config": checkpoint["training_config"],
            }
        )
        _json_write(budget_dir / "metrics.json", metrics)
        with np.load(budget_dir / "history.npz") as history_npz:
            history = {
                "total": history_npz["train_total_loss"].astype(float).tolist(),
                "gt_mse": history_npz["train_gt_mse"].astype(float).tolist(),
                "garch_mse": history_npz["train_garch_mse"].astype(float).tolist(),
            }
        _write_budget_summary(budget_dir / "summary.md", metrics, history)
        sweep.append(metrics)
    _json_write(run_root / "sweep_metrics.json", sweep)
    _write_root_summary(run_root, sweep)
    return sweep


def run_loaded_experiment(
    cache: GinnCache,
    run_root: Path,
    config: ExperimentConfig,
    cache_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> list[dict]:
    run_root.mkdir(parents=True, exist_ok=True)
    _json_write(run_root / "config.json", config.to_dict())
    _json_write(run_root / "dataset_manifest.json", cache.manifest)
    checkpoint_paths = train_and_snapshot(
        cache.X_train,
        cache.y_gt_train,
        cache.y_garch_train,
        cache.manifest["dataset_id"],
        run_root,
        config,
    )
    sweep = evaluate_snapshots(checkpoint_paths, cache, run_root, config)
    if cache_path is not None and manifest_path is not None:
        verify_run(run_root, cache_path, manifest_path, device=config.device)
    return sweep


def _assert_close(name: str, actual: np.ndarray, expected: np.ndarray, rtol: float, atol: float) -> None:
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(f"{name} mismatch")


def verify_run(
    run_root: str | Path,
    cache_path: str | Path,
    manifest_path: str | Path,
    device: str = "auto",
) -> None:
    run_root = Path(run_root)
    config_data = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    config = ExperimentConfig(
        epoch_budgets=tuple(config_data["epoch_budgets"]),
        seed=config_data["seed"],
        batch_size=config_data["batch_size"],
        learning_rate=config_data["learning_rate"],
        lambda_garch=config_data["lambda_garch"],
        input_size=config_data["input_size"],
        hidden_size=config_data["hidden_size"],
        num_layers=config_data["num_layers"],
        dropout=config_data["dropout"],
        output_transform=config_data.get("output_transform", "linear"),
        device=device,
    )
    cache = load_and_validate_cache(cache_path, manifest_path, GinnDataConfig())
    manifest_copy = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest_copy["dataset_id"] != cache.manifest["dataset_id"]:
        raise ValueError("dataset_id mismatch")
    loaded_device = resolve_device(device)
    sweep = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))
    if [row["epoch"] for row in sweep] != list(config.epoch_budgets):
        raise ValueError("sweep epoch ordering mismatch")
    for epoch in config.epoch_budgets:
        budget_dir = run_root / f"e{epoch}"
        checkpoint = torch.load(budget_dir / "checkpoint.pth", map_location=loaded_device)
        if checkpoint["dataset_id"] != cache.manifest["dataset_id"]:
            raise ValueError(f"e{epoch} checkpoint dataset_id mismatch")
        model = _new_model(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        result = evaluate_model(
            model,
            cache.X_test,
            cache.y_gt_test,
            cache.y_garch_test,
            cache.contract_id_test,
            loaded_device,
            batch_size=config.batch_size,
        )
        with np.load(budget_dir / "predictions.npz") as saved:
            _assert_close("prediction", saved["preds"], result["predictions"], 1e-6, 1e-7)
            _assert_close("target", saved["targets"], result["targets"], 1e-6, 1e-7)
            _assert_close("garch target", saved["garch_targets"], result["garch_targets"], 1e-6, 1e-7)
            if not np.array_equal(saved["contract_ids"], result["contract_ids"]):
                raise ValueError("contract_id mismatch")
        metrics = json.loads((budget_dir / "metrics.json").read_text(encoding="utf-8"))
        for key, value in result["metrics"].items():
            if isinstance(value, float):
                if not np.isclose(metrics[key], value, rtol=1e-7, atol=1e-9):
                    raise ValueError(f"{key} metric mismatch")
            elif key != "per_contract":
                if metrics[key] != value:
                    raise ValueError(f"{key} metric mismatch")
        for relative in ("checkpoint.pth", "history.npz", "predictions.npz", "metrics.json", "summary.md"):
            if not (budget_dir / relative).exists():
                raise ValueError(f"missing artifact {budget_dir / relative}")


def _parse_budgets(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _default_run_root(run_name: str) -> Path:
    if Path(run_name).is_absolute() or ".." in Path(run_name).parts:
        raise ValueError("run-name must be a simple relative directory name")
    return Path(__file__).resolve().parent / "experiments" / run_name


def _prepare_run_root(run_root: Path, overwrite: bool) -> None:
    if run_root.exists() and any(run_root.iterdir()):
        if not overwrite:
            raise FileExistsError("run directory exists and is not empty; pass --overwrite")
        shutil.rmtree(run_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the framework-aligned GINN sweep.")
    parser.add_argument("--cache-path", default="data/processed/ginn_4h_seq64_top50.npz")
    parser.add_argument("--manifest-path", default="data/processed/ginn_4h_seq64_top50.manifest.json")
    parser.add_argument("--run-name", default="2026-08-04-v1")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--epoch-budgets", default="15,20,25,50,100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-garch", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--output-transform",
        choices=("linear", "softplus"),
        default="linear",
        help="Final scalar transform. Use softplus to constrain volatility predictions to be non-negative.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-run", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        run_root = Path(args.verify_run) if args.verify_run else (
            Path(args.run_root) if args.run_root else _default_run_root(args.run_name)
        )
        if args.verify_run:
            verify_run(run_root, args.cache_path, args.manifest_path, args.device)
            return 0
        config = ExperimentConfig(
            epoch_budgets=_parse_budgets(args.epoch_budgets),
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lambda_garch=args.lambda_garch,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            output_transform=args.output_transform,
            device=args.device,
        )
        cache = load_and_validate_cache(args.cache_path, args.manifest_path, GinnDataConfig())
        _prepare_run_root(run_root, args.overwrite)
        run_loaded_experiment(cache, run_root, config, args.cache_path, args.manifest_path)
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"GINN experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
