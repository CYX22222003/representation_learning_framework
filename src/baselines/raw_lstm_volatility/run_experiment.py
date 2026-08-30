from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.raw_lstm_volatility.model import (
    RawLSTMVolatility,
    VolatilitySequenceDataset,
    count_parameters,
    make_loader,
    predict,
    set_seed,
    train_one_epoch,
)
from evaluation.metrics import mse_and_corr, regression_metrics
from tasks.volatility_labels import TARGET_DEFINITION, load_volatility_label_bundle, sha256_file


@dataclass(frozen=True)
class ExperimentConfig:
    epoch_budgets: tuple[int, ...] = (15, 50, 100)
    seed: int = 0
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    input_size: int = 5
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    device: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch_budgets", tuple(int(e) for e in self.epoch_budgets))
        if tuple(sorted(set(self.epoch_budgets))) != self.epoch_budgets or any(e <= 0 for e in self.epoch_budgets):
            raise ValueError("epoch_budgets must be positive, unique, and sorted")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["epoch_budgets"] = list(self.epoch_budgets)
        return data


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_processed_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        train = np.asarray(data["train"], dtype=np.float32)
        test = np.asarray(data["test"], dtype=np.float32)
    if train.ndim != 3 or test.ndim != 3 or train.shape[1:] != test.shape[1:]:
        raise ValueError("processed arrays must be train/test [N, seq_len, features] with matching sequence shape")
    return train, test


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _history_arrays(history: list[float], seed: int) -> dict[str, np.ndarray]:
    return {
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        "train_loss": np.asarray(history, dtype=np.float32),
        "seed": np.asarray(seed, dtype=np.int32),
    }


def make_model(config: ExperimentConfig) -> RawLSTMVolatility:
    return RawLSTMVolatility(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


def _assert_label_match(processed_npz: Path, train: np.ndarray, test: np.ndarray, manifest: dict[str, object]) -> None:
    if manifest.get("label_mode") != TARGET_DEFINITION and manifest.get("target_definition") != TARGET_DEFINITION:
        raise ValueError("labels bundle target definition does not match raw LSTM volatility benchmark")
    shapes = manifest.get("processed_shapes", {})
    if shapes:
        if list(train.shape) != list(shapes.get("train", [])) or list(test.shape) != list(shapes.get("test", [])):
            raise ValueError("processed array shapes do not match label manifest")
    expected_sha = manifest.get("processed_npz_sha256")
    if expected_sha and sha256_file(processed_npz) != expected_sha:
        raise ValueError("processed NPZ digest does not match label manifest")
    if int(manifest.get("horizon", 1)) != 1:
        raise ValueError("raw LSTM volatility benchmark requires horizon=1 labels")


def _metric_block(preds: np.ndarray, targets: np.ndarray) -> tuple[dict[str, object], list[str]]:
    pred_t = torch.tensor(preds.reshape(-1, 1), dtype=torch.float32)
    target_t = torch.tensor(targets.reshape(-1, 1), dtype=torch.float32)
    metrics = regression_metrics(pred_t, target_t)
    metrics.update(mse_and_corr(pred_t, target_t))
    warnings: list[str] = []
    if not np.isfinite(metrics["corr"]):
        warnings.append("Pearson correlation is non-finite, likely due to a constant prediction or target series")
        metrics["corr"] = None
    return metrics, warnings


def _quantiles(values: np.ndarray) -> dict[str, float]:
    qs = np.quantile(values.astype(np.float64), [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {name: float(value) for name, value in zip(["min", "q05", "q25", "median", "q75", "q95", "max"], qs)}


def per_contract_metrics(preds: np.ndarray, targets: np.ndarray, contract_ids: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for cid in np.unique(contract_ids):
        mask = contract_ids == cid
        p = preds[mask]
        y = targets[mask]
        metrics, warnings = _metric_block(p, y)
        rows.append(
            {
                "contract_id": int(cid),
                "sample_count": int(mask.sum()),
                **metrics,
                "target_mean": float(np.mean(y)),
                "target_std": float(np.std(y)),
                "prediction_mean": float(np.mean(p)),
                "prediction_std": float(np.std(p)),
                "constant_target": bool(np.std(y) <= 1e-12),
                "constant_prediction": bool(np.std(p) <= 1e-12),
                "warnings": warnings,
            }
        )
    return rows


def evaluate_snapshot(
    checkpoint_path: Path,
    test_dataset: VolatilitySequenceDataset,
    bundle: dict[str, np.ndarray],
    run_root: Path,
    config: ExperimentConfig,
    write_artifacts: bool = True,
) -> dict[str, object]:
    device = resolve_device(config.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    loaded = ExperimentConfig(**checkpoint["training_config"])
    model = make_model(loaded).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = make_loader(test_dataset, loaded.batch_size, loaded.seed, shuffle=False)
    start = time.perf_counter()
    preds, targets = predict(model, loader, device)
    inference_seconds = time.perf_counter() - start
    metrics, warnings = _metric_block(preds, targets)
    residuals = preds - targets
    diagnostics = {
        "negative_prediction_fraction": float(np.mean(preds < 0.0)),
        "nonfinite_prediction_count": int(np.sum(~np.isfinite(preds))),
        "prediction_quantiles": _quantiles(preds),
        "target_quantiles": _quantiles(targets),
        "residual_quantiles": _quantiles(residuals),
        "prediction_mean": float(np.mean(preds)),
        "prediction_std": float(np.std(preds)),
        "target_mean": float(np.mean(targets)),
        "target_std": float(np.std(targets)),
        "inference_seconds": float(inference_seconds),
    }
    budget_dir = checkpoint_path.parent
    if write_artifacts:
        np.savez_compressed(
            budget_dir / "predictions.npz",
            predictions=preds,
            targets=targets,
            contract_ids=bundle["test_contract_ids"],
            processed_row_indices=bundle["test_row_indices"],
            window_starts=bundle["test_window_starts"],
        )
    with np.load(budget_dir / "history.npz") as hist:
        train_loss = float(hist["train_loss"][-1])
    row = {
        "task": "volatility_prediction",
        "epoch": int(checkpoint["completed_epoch"]),
        "seed": int(loaded.seed),
        "train_loss": train_loss,
        "train_sample_count": int(bundle["train_labels"].shape[0]),
        "test_sample_count": int(bundle["test_labels"].shape[0]),
        "parameter_count": int(checkpoint["parameter_count"]),
        **metrics,
        **diagnostics,
        "warnings": warnings,
    }
    if write_artifacts:
        write_json(budget_dir / "metrics.json", row)
        write_json(budget_dir / "per_contract_metrics.json", per_contract_metrics(preds, targets, bundle["test_contract_ids"]))
        _write_budget_summary(budget_dir / "summary.md", row)
    else:
        row["_predictions"] = preds
        row["_targets"] = targets
    return row


def _write_budget_summary(path: Path, row: dict[str, object]) -> None:
    corr = "null" if row["corr"] is None else f"{float(row['corr']):.10f}"
    lines = [
        f"# Raw LSTM Volatility Epoch {row['epoch']}",
        "",
        f"- train loss: `{row['train_loss']:.10f}`",
        f"- MAE: `{row['mae']:.10f}`",
        f"- RMSE: `{row['rmse']:.10f}`",
        f"- MSE: `{row['mse']:.10f}`",
        f"- Pearson correlation: `{corr}`",
        f"- test rows: `{row['test_sample_count']}`",
        f"- negative prediction fraction: `{row['negative_prediction_fraction']:.10f}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_summary(run_root: Path, sweep: list[dict[str, object]]) -> None:
    lines = [
        "# Raw LSTM Volatility Sweep",
        "",
        "All epoch budgets are predeclared characterization points; no checkpoint is selected from test performance.",
        "",
        "| Epoch | MAE | RMSE | MSE | Pearson corr. | Negative fraction | Test rows |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sweep:
        corr = "null" if row["corr"] is None else f"{float(row['corr']):.10f}"
        lines.append(
            f"| {row['epoch']} | {row['mae']:.10f} | {row['rmse']:.10f} | {row['mse']:.10f} | "
            f"{corr} | {row['negative_prediction_fraction']:.10f} | {row['test_sample_count']} |"
        )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_and_snapshot(
    train_dataset: VolatilitySequenceDataset,
    run_root: Path,
    processed_npz: Path,
    labels_npz: Path,
    config: ExperimentConfig,
) -> list[Path]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    model = make_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = make_loader(train_dataset, config.batch_size, config.seed, shuffle=True)
    history: list[float] = []
    checkpoints: list[Path] = []
    for epoch in range(1, max(config.epoch_budgets) + 1):
        history.append(train_one_epoch(model, loader, optimizer, device, grad_clip=config.grad_clip))
        if epoch in set(config.epoch_budgets):
            budget_dir = run_root / f"e{epoch}"
            budget_dir.mkdir(parents=True, exist_ok=True)
            path = budget_dir / "checkpoint.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "completed_epoch": epoch,
                    "training_config": config.to_dict(),
                    "processed_npz": str(processed_npz),
                    "labels_npz": str(labels_npz),
                    "parameter_count": count_parameters(model),
                },
                path,
            )
            np.savez_compressed(budget_dir / "history.npz", **_history_arrays(history, config.seed))
            checkpoints.append(path)
            print(f"Raw LSTM checkpoint saved: epoch={epoch} path={path}", flush=True)
    return checkpoints


def run_experiment(processed_npz: str | Path, labels_npz: str | Path, run_root: Path, config: ExperimentConfig) -> list[dict[str, object]]:
    processed_path = Path(processed_npz)
    labels_path = Path(labels_npz)
    train, test = load_processed_npz(processed_path)
    bundle, label_manifest = load_volatility_label_bundle(labels_path)
    _assert_label_match(processed_path, train, test, label_manifest)
    if train.shape[2] != config.input_size:
        raise ValueError(f"model input_size={config.input_size} does not match processed feature count {train.shape[2]}")

    train_dataset = VolatilitySequenceDataset(train, bundle["train_labels"], bundle["train_row_indices"])
    test_dataset = VolatilitySequenceDataset(test, bundle["test_labels"], bundle["test_row_indices"])
    run_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    model_for_count = make_model(config)
    write_json(run_root / "config.json", {**config.to_dict(), "parameter_count": count_parameters(model_for_count)})
    write_json(
        run_root / "dataset_manifest.json",
        {
            "processed_npz": str(processed_path),
            "labels_npz": str(labels_path),
            "processed_sha256": sha256_file(processed_path),
            "label_dataset_id": label_manifest.get("dataset_id"),
            "label_manifest": label_manifest,
            "train_sequence_shape": list(train.shape),
            "test_sequence_shape": list(test.shape),
            "train_sample_count": len(train_dataset),
            "test_sample_count": len(test_dataset),
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "test_evaluation_policy": "snapshots evaluated only after max epoch training completes",
        },
    )
    checkpoint_paths = train_and_snapshot(train_dataset, run_root, processed_path, labels_path, config)
    sweep = [evaluate_snapshot(path, test_dataset, bundle, run_root, config) for path in checkpoint_paths]
    write_json(run_root / "sweep_metrics.json", sweep)
    _write_root_summary(run_root, sweep)
    return sweep


def verify_run(run_root: str | Path, processed_npz: str | Path, labels_npz: str | Path, device_name: str = "auto") -> bool:
    run_root = Path(run_root)
    config_payload = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    config_payload.pop("parameter_count", None)
    config = ExperimentConfig(**(config_payload | {"device": device_name}))
    _, test = load_processed_npz(processed_npz)
    bundle, _ = load_volatility_label_bundle(labels_npz)
    test_dataset = VolatilitySequenceDataset(test, bundle["test_labels"], bundle["test_row_indices"])
    ok = True
    for epoch in config.epoch_budgets:
        budget_dir = run_root / f"e{epoch}"
        if not (budget_dir / "predictions.npz").exists() or not (budget_dir / "metrics.json").exists():
            print(f"verify-run missing evaluated artifacts for epoch {epoch}", file=sys.stderr)
            ok = False
            continue
        with np.load(budget_dir / "predictions.npz") as saved_pred:
            saved_predictions = saved_pred["predictions"].copy()
            saved_targets = saved_pred["targets"].copy()
            saved_rows = saved_pred["processed_row_indices"].copy()
        saved_metrics = json.loads((budget_dir / "metrics.json").read_text(encoding="utf-8"))
        row = evaluate_snapshot(budget_dir / "checkpoint.pth", test_dataset, bundle, run_root, config, write_artifacts=False)
        preds_match = np.allclose(saved_predictions, row.pop("_predictions"), rtol=1e-6, atol=1e-7)
        targets_match = np.array_equal(saved_targets, row.pop("_targets"))
        rows_match = np.array_equal(saved_rows, bundle["test_row_indices"])
        metric_keys = ("mae", "rmse", "mse", "negative_prediction_fraction", "test_sample_count")
        metrics_match = all(
            np.isclose(float(saved_metrics[key]), float(row[key]), rtol=1e-6, atol=1e-8)
            for key in metric_keys
        )
        ok = ok and preds_match and targets_match and rows_match and metrics_match
    print(f"verify-run {'succeeded' if ok else 'failed'}: {run_root}", flush=True)
    return ok


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one epoch budget")
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
    parser = argparse.ArgumentParser(description="Run the Raw LSTM volatility benchmark.")
    parser.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    parser.add_argument("--labels-npz", default="data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz")
    parser.add_argument("--run-name", default="4h-seq64-top50-seed0")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--epoch-budgets", default="15,50,100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-run", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verify_run:
            return 0 if verify_run(args.verify_run, args.processed_npz, args.labels_npz, args.device) else 1
        config = ExperimentConfig(
            epoch_budgets=_parse_int_tuple(args.epoch_budgets),
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
        )
        run_root = Path(args.run_root) if args.run_root else _default_run_root(args.run_name)
        _prepare_run_root(run_root, args.overwrite)
        run_experiment(args.processed_npz, args.labels_npz, run_root, config)
        print(f"Raw LSTM volatility experiment completed: run_dir={run_root}", flush=True)
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"raw LSTM volatility experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
