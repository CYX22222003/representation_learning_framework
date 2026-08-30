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

from baselines.garch_lstm_stacking.crossfit import CrossfitPlan, assert_oof_integrity, assert_same_row_identity, make_expanding_folds
from baselines.garch_lstm_stacking.garch import GuardedGarchForecaster
from baselines.garch_lstm_stacking.meta import StackingMetaModel, build_meta_features, clipped_predictions, clipping_diagnostics
from baselines.raw_lstm_volatility.model import RawLSTMVolatility, VolatilitySequenceDataset, make_loader, predict, set_seed, train_one_epoch
from baselines.raw_lstm_volatility.run_experiment import ExperimentConfig as RawLSTMConfig
from baselines.raw_lstm_volatility.run_experiment import load_processed_npz, per_contract_metrics, resolve_device, write_json
from evaluation.metrics import mse_and_corr, regression_metrics
from tasks.volatility_labels import TARGET_DEFINITION, load_volatility_label_bundle, sha256_file


@dataclass(frozen=True)
class StackConfig:
    epoch_budgets: tuple[int, ...] = (15, 50, 100)
    crossfit_folds: int = 5
    seed: int = 0
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    elasticnet_alpha: float = 1e-4
    elasticnet_l1_ratio: float = 0.5
    device: str = "auto"
    price_index: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch_budgets", tuple(int(e) for e in self.epoch_budgets))
        if tuple(sorted(set(self.epoch_budgets))) != self.epoch_budgets or any(e <= 0 for e in self.epoch_budgets):
            raise ValueError("epoch_budgets must be positive, unique, and sorted")
        if self.crossfit_folds <= 0:
            raise ValueError("crossfit_folds must be positive")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["epoch_budgets"] = list(self.epoch_budgets)
        return payload


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _metric_block(preds: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    pred_t = torch.tensor(np.asarray(preds).reshape(-1, 1), dtype=torch.float32)
    target_t = torch.tensor(np.asarray(targets).reshape(-1, 1), dtype=torch.float32)
    out = regression_metrics(pred_t, target_t)
    out.update(mse_and_corr(pred_t, target_t))
    if not np.isfinite(out["corr"]):
        out["corr"] = None
    return out


def _load_raw_lstm_run(raw_lstm_run: Path, processed_npz: Path, labels_npz: Path, config: StackConfig) -> dict[int, dict[str, np.ndarray]]:
    if not raw_lstm_run.exists():
        raise FileNotFoundError(f"Raw LSTM run does not exist: {raw_lstm_run}")
    run_config_path = raw_lstm_run / "config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Raw LSTM run is missing config.json: {raw_lstm_run}")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    raw_epochs = tuple(int(e) for e in run_config.get("epoch_budgets", []))
    if raw_epochs != config.epoch_budgets:
        raise ValueError(f"Raw LSTM epoch budgets {raw_epochs} do not match stack {config.epoch_budgets}")
    if int(run_config.get("seed", config.seed)) != int(config.seed):
        raise ValueError("Raw LSTM seed does not match stack seed")

    payload: dict[int, dict[str, np.ndarray]] = {}
    for epoch in config.epoch_budgets:
        path = raw_lstm_run / f"e{epoch}" / "predictions.npz"
        if not path.exists():
            raise FileNotFoundError(f"Raw LSTM predictions are missing for epoch {epoch}: {path}")
        with np.load(path) as data:
            payload[epoch] = {key: data[key].copy() for key in data.files}
        for key in ("predictions", "targets", "contract_ids", "processed_row_indices", "window_starts"):
            if key not in payload[epoch]:
                raise ValueError(f"Raw LSTM predictions for epoch {epoch} are missing {key}")
    manifest_path = raw_lstm_run / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest.get("processed_sha256")
        if expected_sha and expected_sha != sha256_file(processed_npz):
            raise ValueError("Raw LSTM processed dataset digest does not match stack request")
        if manifest.get("labels_npz") and Path(str(manifest["labels_npz"])).name != labels_npz.name:
            raise ValueError("Raw LSTM label bundle path does not match stack request")
        if manifest.get("label_manifest", {}).get("target_definition") not in (None, TARGET_DEFINITION):
            raise ValueError("Raw LSTM target definition does not match stack target")
    first = payload[config.epoch_budgets[0]]
    for epoch in config.epoch_budgets[1:]:
        assert_same_row_identity(
            {
                "targets": first["targets"],
                "contract_ids": first["contract_ids"],
                "processed_row_indices": first["processed_row_indices"],
                "window_starts": first["window_starts"],
            },
            {
                "targets": payload[epoch]["targets"],
                "contract_ids": payload[epoch]["contract_ids"],
                "processed_row_indices": payload[epoch]["processed_row_indices"],
                "window_starts": payload[epoch]["window_starts"],
            },
        )
    return payload


def _make_raw_model() -> RawLSTMVolatility:
    return RawLSTMVolatility(input_size=5, hidden_size=128, num_layers=2, dropout=0.1)


def _train_lstm_oof(
    train_sequences: np.ndarray,
    bundle: dict[str, np.ndarray],
    plan: CrossfitPlan,
    run_root: Path,
    config: StackConfig,
) -> dict[int, np.ndarray]:
    device = resolve_device(config.device)
    oof_by_epoch = {epoch: np.full(len(plan.records), np.nan, dtype=np.float32) for epoch in config.epoch_budgets}
    train_contracts = bundle["train_contract_ids"]
    train_starts = bundle["train_window_starts"]
    train_rows = bundle["train_row_indices"]
    train_labels = bundle["train_labels"]
    fold_ids = plan.fold_ids()
    for fold_id in range(1, config.crossfit_folds + 1):
        pred_positions = np.flatnonzero(fold_ids == fold_id)
        if pred_positions.size == 0:
            continue
        allowed = np.zeros(train_labels.shape[0], dtype=bool)
        for record in np.asarray(plan.records, dtype=object)[pred_positions]:
            mask = (train_contracts == record.contract_id) & (train_starts <= record.max_training_start)
            allowed |= mask
        pred_rows = plan.prediction_row_indices()[pred_positions]
        pred_mask = np.isin(train_rows, pred_rows)
        if not np.any(allowed):
            raise ValueError(f"fold {fold_id} has no training rows")
        set_seed(config.seed + fold_id)
        model = _make_raw_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        train_ds = VolatilitySequenceDataset(train_sequences, train_labels[allowed], train_rows[allowed])
        pred_ds = VolatilitySequenceDataset(train_sequences, train_labels[pred_mask], train_rows[pred_mask])
        loader = make_loader(train_ds, config.batch_size, config.seed + fold_id, shuffle=True)
        history: list[float] = []
        for epoch in range(1, max(config.epoch_budgets) + 1):
            history.append(train_one_epoch(model, loader, optimizer, device, grad_clip=config.grad_clip))
            if epoch in config.epoch_budgets:
                pred_loader = make_loader(pred_ds, config.batch_size, config.seed, shuffle=False)
                preds, _ = predict(model, pred_loader, device)
                row_to_pred = {int(row): float(pred) for row, pred in zip(train_rows[pred_mask], preds)}
                for pos in pred_positions:
                    oof_by_epoch[epoch][pos] = row_to_pred[int(plan.records[pos].processed_row_index)]
        np.savez_compressed(run_root / "oof" / f"lstm_fold{fold_id}_history.npz", train_loss=np.asarray(history, dtype=np.float32))
    for epoch, values in oof_by_epoch.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite OOF LSTM predictions for epoch {epoch}")
    return oof_by_epoch


def _contract_close_from_sequences(sequences: np.ndarray, row_indices: np.ndarray, price_index: int) -> tuple[np.ndarray, dict[int, int]]:
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    if rows.size == 0:
        return np.asarray([], dtype=np.float64), {}
    seq = np.asarray(sequences[rows], dtype=np.float64)
    close = list(seq[0, :, price_index])
    row_to_local = {int(rows[0]): 0}
    last_row = int(rows[0])
    for raw_row, window in zip(rows[1:], seq[1:]):
        raw_row = int(raw_row)
        gap = raw_row - last_row
        if gap <= 0:
            raise ValueError("row_indices must be strictly increasing per contract")
        if gap == 1:
            close.append(float(window[-1, price_index]))
        else:
            close.extend(float(v) for v in window[-gap:, price_index])
        row_to_local[raw_row] = raw_row - int(rows[0])
        last_row = raw_row
    return np.asarray(close, dtype=np.float64), row_to_local


def _contract_close_from_row_range(
    sequences: np.ndarray,
    first_row: int,
    last_row_inclusive: int,
    price_index: int,
) -> tuple[np.ndarray, dict[int, int]]:
    rows = np.arange(int(first_row), int(last_row_inclusive) + 1, dtype=np.int64)
    return _contract_close_from_sequences(sequences, rows, price_index)


def _garch_for_split(
    sequences: np.ndarray,
    labels_bundle: dict[str, np.ndarray],
    split: str,
    target_rows: np.ndarray,
    config: StackConfig,
    *,
    fit_row_limit_by_contract: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    out_raw = np.empty(target_rows.size, dtype=np.float32)
    out_guarded = np.empty(target_rows.size, dtype=np.float32)
    out_fallback = np.empty(target_rows.size, dtype=bool)
    diagnostics: list[dict[str, object]] = []
    all_rows = labels_bundle[f"{split}_row_indices"]
    all_contracts = labels_bundle[f"{split}_contract_ids"]
    seq_len = int(sequences.shape[1])
    forecaster = GuardedGarchForecaster()
    for cid in np.unique(all_contracts):
        contract_mask = all_contracts == cid
        contract_rows = all_rows[contract_mask]
        first = int(contract_rows.min())
        last = int(contract_rows.max()) + 1
        close, row_to_local = _contract_close_from_row_range(sequences, first, min(last, sequences.shape[0] - 1), config.price_index)
        global_positions = np.flatnonzero(np.isin(target_rows, contract_rows))
        if global_positions.size == 0:
            continue
        wanted_rows = target_rows[global_positions]
        local_rows = np.asarray([row_to_local[int(row)] for row in wanted_rows], dtype=np.int64)
        fit_limit = fit_row_limit_by_contract.get(int(cid), local_rows.max() + 1) if fit_row_limit_by_contract else local_rows.max() + 1
        result = forecaster.fit_predict(close, local_rows, seq_len=seq_len, fit_row_limit=int(fit_limit))
        out_raw[global_positions] = result.prediction_raw
        out_guarded[global_positions] = result.prediction_guarded
        out_fallback[global_positions] = result.fallback_flag
        diag = result.diagnostics.to_dict()
        diag["contract_id"] = int(cid)
        diag["row_count"] = int(global_positions.size)
        diag["fallback_forecast_count"] = int(np.sum(result.fallback_flag))
        diag["capped_forecast_count"] = int(np.sum(result.capped_flag))
        diagnostics.append(diag)
    return out_raw, out_guarded, out_fallback, diagnostics


def _garch_oof(
    train_sequences: np.ndarray,
    bundle: dict[str, np.ndarray],
    plan: CrossfitPlan,
    config: StackConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    rows = plan.prediction_row_indices()
    out_raw = np.empty(rows.size, dtype=np.float32)
    out_guarded = np.empty(rows.size, dtype=np.float32)
    out_fallback = np.empty(rows.size, dtype=bool)
    diagnostics: list[dict[str, object]] = []
    train_rows = bundle["train_row_indices"]
    train_contracts = bundle["train_contract_ids"]
    forecaster = GuardedGarchForecaster()
    for cid in np.unique(plan.contract_ids()):
        c_all_rows = train_rows[train_contracts == cid]
        first = int(c_all_rows.min())
        last = int(c_all_rows.max()) + 1
        close, row_to_local = _contract_close_from_row_range(train_sequences, first, min(last, train_sequences.shape[0] - 1), config.price_index)
        for fit_start in sorted({int(r.max_training_start) for r in plan.records if r.contract_id == int(cid)}):
            positions = [
                i
                for i, record in enumerate(plan.records)
                if record.contract_id == int(cid) and int(record.max_training_start) == fit_start
            ]
            wanted_rows = rows[positions]
            local_rows = np.asarray([row_to_local[int(row)] for row in wanted_rows], dtype=np.int64)
            eligible_global_rows = train_rows[(train_contracts == cid) & (bundle["train_window_starts"] <= fit_start)]
            if eligible_global_rows.size == 0:
                raise ValueError(f"contract {cid} has no GARCH fitting rows before window start {fit_start}")
            fit_limit = row_to_local[int(np.max(eligible_global_rows))] + 1
            result = forecaster.fit_predict(close, local_rows, seq_len=int(train_sequences.shape[1]), fit_row_limit=int(fit_limit))
            out_raw[positions] = result.prediction_raw
            out_guarded[positions] = result.prediction_guarded
            out_fallback[positions] = result.fallback_flag
            diag = result.diagnostics.to_dict()
            diag["contract_id"] = int(cid)
            diag["max_training_start"] = int(fit_start)
            diag["row_count"] = int(len(positions))
            diag["fallback_forecast_count"] = int(np.sum(result.fallback_flag))
            diagnostics.append(diag)
    return out_raw, out_guarded, out_fallback, diagnostics


def _garch_test(
    train_sequences: np.ndarray,
    test_sequences: np.ndarray,
    bundle: dict[str, np.ndarray],
    config: StackConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    out_raw = np.empty(bundle["test_row_indices"].shape[0], dtype=np.float32)
    out_guarded = np.empty(bundle["test_row_indices"].shape[0], dtype=np.float32)
    out_fallback = np.empty(bundle["test_row_indices"].shape[0], dtype=bool)
    diagnostics: list[dict[str, object]] = []
    forecaster = GuardedGarchForecaster()
    for cid in np.unique(bundle["test_contract_ids"]):
        train_rows = bundle["train_row_indices"][bundle["train_contract_ids"] == cid]
        test_rows = bundle["test_row_indices"][bundle["test_contract_ids"] == cid]
        if train_rows.size == 0 or test_rows.size == 0:
            continue
        train_first = int(train_rows.min())
        train_last = int(train_rows.max()) + 1
        test_first = int(test_rows.min())
        test_last = int(test_rows.max()) + 1
        train_close, _ = _contract_close_from_row_range(
            train_sequences,
            train_first,
            min(train_last, train_sequences.shape[0] - 1),
            config.price_index,
        )
        test_close, test_local = _contract_close_from_row_range(
            test_sequences,
            test_first,
            min(test_last, test_sequences.shape[0] - 1),
            config.price_index,
        )
        offset = int(train_close.size)
        combined_close = np.concatenate([train_close, test_close])
        positions = np.flatnonzero(bundle["test_contract_ids"] == cid)
        local_rows = np.asarray([offset + test_local[int(row)] for row in test_rows], dtype=np.int64)
        result = forecaster.fit_predict(
            combined_close,
            local_rows,
            seq_len=int(test_sequences.shape[1]),
            fit_row_limit=max(0, int(train_close.size) - int(test_sequences.shape[1]) + 1),
            fit_price_count=int(train_close.size),
        )
        out_raw[positions] = result.prediction_raw
        out_guarded[positions] = result.prediction_guarded
        out_fallback[positions] = result.fallback_flag
        diag = result.diagnostics.to_dict()
        diag["contract_id"] = int(cid)
        diag["row_count"] = int(positions.size)
        diag["fallback_forecast_count"] = int(np.sum(result.fallback_flag))
        diagnostics.append(diag)
    return out_raw, out_guarded, out_fallback, diagnostics


def _save_oof(
    run_root: Path,
    plan: CrossfitPlan,
    targets: np.ndarray,
    garch_raw: np.ndarray,
    garch_guarded: np.ndarray,
    garch_fallback: np.ndarray,
    lstm_oof: dict[int, np.ndarray],
) -> None:
    oof = run_root / "oof"
    oof.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        oof / "fold_assignments.npz",
        contract_ids=plan.contract_ids(),
        processed_row_indices=plan.prediction_row_indices(),
        window_starts=plan.window_starts(),
        fold_ids=plan.fold_ids(),
        max_training_starts=plan.max_training_starts(),
        targets=targets,
        meta_excluded_burn_in=plan.burn_in_row_indices,
    )
    np.savez_compressed(
        oof / "garch_predictions.npz",
        prediction_raw=garch_raw,
        prediction_guarded=garch_guarded,
        fallback_flag=garch_fallback,
        targets=targets,
        contract_ids=plan.contract_ids(),
        processed_row_indices=plan.prediction_row_indices(),
        window_starts=plan.window_starts(),
    )
    for epoch, preds in lstm_oof.items():
        np.savez_compressed(
            oof / f"lstm_predictions_e{epoch}.npz",
            predictions=preds,
            targets=targets,
            contract_ids=plan.contract_ids(),
            processed_row_indices=plan.prediction_row_indices(),
            window_starts=plan.window_starts(),
        )


def _write_summary(run_root: Path, rows: list[dict[str, object]], comparisons: list[dict[str, object]]) -> None:
    lines = [
        "# GARCH--LSTM Stacking Volatility Benchmark",
        "",
        "This is a Peter-et-al.-inspired adapted stack, not an exact paper reproduction. All epoch budgets are reported; no test metric is used for model selection.",
        "",
        "| LSTM epoch | Model | MAE | RMSE | MSE | Pearson corr. | Macro-contract MSE | Raw negative fraction |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comp in comparisons:
        epoch = int(comp["epoch"])
        for model_name, prefix in (("Raw LSTM", "raw_lstm"), ("GARCH--LSTM stack", "stack")):
            corr = comp[f"{prefix}_corr"]
            corr_text = "null" if corr is None else f"{float(corr):.10f}"
            lines.append(
                f"| {epoch} | {model_name} | {comp[f'{prefix}_mae']:.10f} | {comp[f'{prefix}_rmse']:.10f} | "
                f"{comp[f'{prefix}_mse']:.10f} | {corr_text} | {comp[f'{prefix}_macro_contract_mse']:.10f} | "
                f"{comp[f'{prefix}_raw_negative_fraction']:.10f} |"
            )
    lines += [
        "",
        "| LSTM epoch | Intercept | GARCH coefficient | LSTM coefficient | Interaction coefficient | GARCH fallback rate | GARCH cap rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        c = row["coefficients"]
        lines.append(
            f"| {row['epoch']} | {row['intercept']:.10f} | {c['garch']:.10f} | {c['lstm']:.10f} | "
            f"{c['interaction']:.10f} | {row['garch_fallback_rate']:.10f} | {row['garch_cap_rate']:.10f} |"
        )
    (run_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    processed_npz: str | Path,
    labels_npz: str | Path,
    raw_lstm_run: str | Path,
    run_root: Path,
    config: StackConfig,
) -> list[dict[str, object]]:
    processed_path = Path(processed_npz)
    labels_path = Path(labels_npz)
    raw_run = Path(raw_lstm_run)
    train_sequences, test_sequences = load_processed_npz(processed_path)
    bundle, label_manifest = load_volatility_label_bundle(labels_path)
    raw_predictions = _load_raw_lstm_run(raw_run, processed_path, labels_path, config)
    expected_test = {
        "targets": bundle["test_labels"],
        "contract_ids": bundle["test_contract_ids"],
        "processed_row_indices": bundle["test_row_indices"],
        "window_starts": bundle["test_window_starts"],
    }
    assert_same_row_identity(expected_test, raw_predictions[config.epoch_budgets[0]])

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "oof").mkdir(parents=True, exist_ok=True)
    dump_json(run_root / "config.json", config.to_dict())
    dump_json(
        run_root / "dataset_manifest.json",
        {
            "processed_npz": str(processed_path),
            "labels_npz": str(labels_path),
            "processed_sha256": sha256_file(processed_path),
            "label_manifest": label_manifest,
            "train_sequence_shape": list(train_sequences.shape),
            "test_sequence_shape": list(test_sequences.shape),
        },
    )
    dump_json(run_root / "raw_lstm_run_manifest.json", {"raw_lstm_run": str(raw_run), "reuse_policy": "final test LSTM predictions are reused exactly"})

    t0 = time.perf_counter()
    plan = make_expanding_folds(bundle["train_contract_ids"], bundle["train_row_indices"], bundle["train_window_starts"], n_folds=config.crossfit_folds)
    assert_oof_integrity(plan)
    dump_json(run_root / "crossfit_manifest.json", plan.manifest())
    oof_rows = plan.prediction_row_indices()
    row_to_label = {int(row): float(label) for row, label in zip(bundle["train_row_indices"], bundle["train_labels"])}
    oof_targets = np.asarray([row_to_label[int(row)] for row in oof_rows], dtype=np.float32)

    garch_raw_oof, garch_guarded_oof, garch_fallback_oof, garch_oof_diag = _garch_oof(train_sequences, bundle, plan, config)
    lstm_oof = _train_lstm_oof(train_sequences, bundle, plan, run_root, config)
    crossfit_seconds = time.perf_counter() - t0
    _save_oof(run_root, plan, oof_targets, garch_raw_oof, garch_guarded_oof, garch_fallback_oof, lstm_oof)

    garch_raw_test, garch_guarded_test, garch_fallback_test, garch_test_diag = _garch_test(train_sequences, test_sequences, bundle, config)
    dump_json(run_root / "garch_diagnostics.json", {"oof": garch_oof_diag, "test": garch_test_diag})

    sweep: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for epoch in config.epoch_budgets:
        budget_dir = run_root / f"e{epoch}"
        budget_dir.mkdir(parents=True, exist_ok=True)
        t_meta = time.perf_counter()
        meta = StackingMetaModel.fit(
            garch_guarded_oof,
            lstm_oof[epoch],
            oof_targets,
            alpha=config.elasticnet_alpha,
            l1_ratio=config.elasticnet_l1_ratio,
        )
        raw_lstm = raw_predictions[epoch]["predictions"].astype(np.float32)
        interaction = garch_guarded_test * raw_lstm
        stack_raw = meta.predict_raw(garch_guarded_test, raw_lstm)
        stack_nonnegative = clipped_predictions(stack_raw)
        meta_seconds = time.perf_counter() - t_meta
        metrics = _metric_block(stack_nonnegative, bundle["test_labels"])
        clip = clipping_diagnostics(stack_raw, stack_nonnegative, bundle["test_labels"])
        per_contract = per_contract_metrics(stack_nonnegative.astype(np.float32), bundle["test_labels"], bundle["test_contract_ids"])
        raw_per_contract = per_contract_metrics(raw_lstm, bundle["test_labels"], bundle["test_contract_ids"])
        macro_mse = float(np.mean([row["mse"] for row in per_contract]))
        raw_macro_mse = float(np.mean([row["mse"] for row in raw_per_contract]))
        row = {
            "epoch": int(epoch),
            **metrics,
            **clip,
            "sample_count": int(bundle["test_labels"].shape[0]),
            "inference_seconds": float(meta_seconds),
            "crossfit_seconds": float(crossfit_seconds),
            "garch_fallback_rate": float(np.mean(garch_fallback_test)),
            "garch_cap_rate": 0.0,
            "intercept": meta.intercept,
            "coefficients": meta.model_dict()["coefficients"],
            "macro_contract_mse": macro_mse,
        }
        dump_json(budget_dir / "meta_scaler.json", meta.scaler_dict())
        dump_json(budget_dir / "meta_model.json", meta.model_dict())
        np.savez_compressed(
            budget_dir / "predictions.npz",
            garch_prediction_raw=garch_raw_test,
            garch_prediction_guarded=garch_guarded_test,
            garch_fallback_flag=garch_fallback_test,
            lstm_prediction=raw_lstm,
            interaction=interaction,
            stack_prediction_raw=stack_raw.astype(np.float32),
            stack_prediction_nonnegative=stack_nonnegative.astype(np.float32),
            targets=bundle["test_labels"],
            contract_ids=bundle["test_contract_ids"],
            processed_row_indices=bundle["test_row_indices"],
            window_starts=bundle["test_window_starts"],
        )
        dump_json(budget_dir / "metrics.json", row)
        dump_json(budget_dir / "per_contract_metrics.json", per_contract)
        (budget_dir / "summary.md").write_text(f"# Stack epoch {epoch}\n\nMSE: `{row['mse']:.10f}`\n", encoding="utf-8")
        raw_metrics = _metric_block(raw_lstm, bundle["test_labels"])
        comparison = {
            "epoch": int(epoch),
            "raw_lstm_mae": raw_metrics["mae"],
            "raw_lstm_rmse": raw_metrics["rmse"],
            "raw_lstm_mse": raw_metrics["mse"],
            "raw_lstm_corr": raw_metrics["corr"],
            "raw_lstm_macro_contract_mse": raw_macro_mse,
            "raw_lstm_raw_negative_fraction": float(np.mean(raw_lstm < 0.0)),
            "stack_mae": row["mae"],
            "stack_rmse": row["rmse"],
            "stack_mse": row["mse"],
            "stack_corr": row["corr"],
            "stack_macro_contract_mse": macro_mse,
            "stack_raw_negative_fraction": row["raw_negative_fraction"],
            "delta_mse_stack_minus_raw": float(row["mse"] - raw_metrics["mse"]),
        }
        comparisons.append(comparison)
        sweep.append(row)
    dump_json(run_root / "sweep_metrics.json", sweep)
    dump_json(run_root / "comparison_with_raw_lstm.json", comparisons)
    _write_summary(run_root, sweep, comparisons)
    return sweep


def verify_run(run_root: str | Path) -> bool:
    root = Path(run_root)
    ok = True
    with np.load(root / "oof" / "garch_predictions.npz") as garch_oof:
        g_oof = garch_oof["prediction_guarded"].copy()
        y_oof = garch_oof["targets"].copy()
    for budget_dir in sorted(root.glob("e[0-9]*")):
        epoch = int(budget_dir.name[1:])
        with np.load(root / "oof" / f"lstm_predictions_e{epoch}.npz") as lstm_oof:
            l_oof = lstm_oof["predictions"].copy()
        with np.load(budget_dir / "predictions.npz") as pred:
            g_test = pred["garch_prediction_guarded"].copy()
            l_test = pred["lstm_prediction"].copy()
            saved_raw = pred["stack_prediction_raw"].copy()
            saved_nonnegative = pred["stack_prediction_nonnegative"].copy()
        scaler_payload = json.loads((budget_dir / "meta_scaler.json").read_text(encoding="utf-8"))
        model_payload = json.loads((budget_dir / "meta_model.json").read_text(encoding="utf-8"))
        meta = StackingMetaModel.from_dict({**model_payload, **scaler_payload})
        replay_raw = meta.predict_raw(g_test, l_test)
        replay_nonnegative = clipped_predictions(replay_raw)
        ok = ok and np.allclose(saved_raw, replay_raw, rtol=1e-6, atol=1e-7)
        ok = ok and np.allclose(saved_nonnegative, replay_nonnegative, rtol=1e-6, atol=1e-7)
        # Also verify the serialized scaler/model can consume original OOF training features.
        _ = meta.transform(build_meta_features(g_oof, l_oof))
        if y_oof.shape[0] != l_oof.shape[0]:
            ok = False
    print(f"verify-run {'succeeded' if ok else 'failed'}: {root}", flush=True)
    return bool(ok)


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
    parser = argparse.ArgumentParser(description="Run the GARCH--LSTM stacking volatility benchmark.")
    parser.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    parser.add_argument("--labels-npz", default="data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz")
    parser.add_argument("--raw-lstm-run", default="src/baselines/raw_lstm_volatility/experiments/4h-seq64-top50-seed0")
    parser.add_argument("--run-name", default="4h-seq64-top50-seed0")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--epoch-budgets", default="15,50,100")
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--elasticnet-alpha", type=float, default=1e-4)
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-run", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verify_run:
            return 0 if verify_run(args.verify_run) else 1
        config = StackConfig(
            epoch_budgets=_parse_int_tuple(args.epoch_budgets),
            crossfit_folds=args.crossfit_folds,
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            elasticnet_alpha=args.elasticnet_alpha,
            elasticnet_l1_ratio=args.elasticnet_l1_ratio,
            device=args.device,
        )
        run_root = Path(args.run_root) if args.run_root else _default_run_root(args.run_name)
        _prepare_run_root(run_root, args.overwrite)
        run_experiment(args.processed_npz, args.labels_npz, args.raw_lstm_run, run_root, config)
        print(f"GARCH--LSTM stacking experiment completed: run_dir={run_root}", flush=True)
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"GARCH--LSTM stacking experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
