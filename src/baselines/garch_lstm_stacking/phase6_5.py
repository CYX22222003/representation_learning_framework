"""Strict Phase 6.5B GARCH--LSTM stacking for future H=8 variance.

The legacy runner in this package is retained for its historical four-hour
proxy.  This module is deliberately separate: it models raw hourly
probability changes, creates global-calendar expanding OOF predictions, and
uses the completed Phase 6 Raw-LSTM evaluation predictions only after exact
identity and provenance checks.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baselines.garch_lstm_stacking.garch import RawChangeGarchHorizonForecaster
from baselines.garch_lstm_stacking.meta import StackingMetaModel, clipping_diagnostics
from data_processing.phase5_walks import sha256_arrays, sha256_file
from data_processing.phase6_volatility_labels import validate_volatility_label_bundle_files
from training.phase5_downstream import environment_manifest, resolve_device, set_seed
from training.phase5_encoder import write_json
from training.phase6_volatility import (
    SNAPSHOT_EPOCHS,
    TARGET_MULTIPLIER,
    Phase6VolatilityConfig,
    build_model,
    validate_volatility_run,
    volatility_metrics,
)


HOUR_NS = 3_600_000_000_000
OOF_FOLDS = 5
OOF_SCHEMA = "phase6-5b-calendar-expanding-oof-v2"


@dataclass(frozen=True)
class Phase65GarchLSTMConfig:
    walk: int
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    folds: int = OOF_FOLDS
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_clip_norm: float = 5.0
    target_multiplier: float = TARGET_MULTIPLIER
    smooth_l1_beta: float = 1.0
    elasticnet_alpha: float = 1e-4
    elasticnet_l1_ratio: float = 0.5
    elasticnet_max_iter: int = 10_000
    horizon_hours: int = 8
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("Phase 6.5B requires one 50-epoch trajectory with 5/15/50 snapshots")
        if self.seed != 0 or self.folds != 5:
            raise ValueError("Phase 6.5B is frozen to seed 0 and five OOF folds")
        if self.batch_size != 512 or self.learning_rate != 1e-4 or self.weight_decay != 0.0:
            raise ValueError("Phase 6.5B Raw-LSTM optimization recipe changed")
        if self.gradient_clip_norm != 5.0 or self.target_multiplier != 10_000.0:
            raise ValueError("Phase 6.5B target/loss recipe changed")
        if self.smooth_l1_beta != 1.0 or self.horizon_hours != 8:
            raise ValueError("Phase 6.5B is frozen to SmoothL1 beta=1 and H=8")
        if (
            self.elasticnet_alpha != 1e-4
            or self.elasticnet_l1_ratio != 0.5
            or self.elasticnet_max_iter != 10_000
        ):
            raise ValueError("Phase 6.5B ElasticNet recipe changed")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["snapshot_epochs"] = list(self.snapshot_epochs)
        return result


@dataclass(frozen=True)
class CalendarOOFFold:
    fold_id: int
    prediction_positions: np.ndarray
    training_positions: np.ndarray
    prediction_start_ns: int
    prediction_end_ns: int
    training_target_cutoff_ns: int

    def manifest(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "prediction_count": int(len(self.prediction_positions)),
            "training_count": int(len(self.training_positions)),
            "prediction_start_ns": self.prediction_start_ns,
            "prediction_end_ns": self.prediction_end_ns,
            "training_target_cutoff_ns": self.training_target_cutoff_ns,
            "prediction_positions_sha256": sha256_arrays(self.prediction_positions),
            "training_positions_sha256": sha256_arrays(self.training_positions),
        }


@dataclass(frozen=True)
class CalendarOOFPlan:
    folds: tuple[CalendarOOFFold, ...]
    burn_in_positions: np.ndarray
    prediction_positions: np.ndarray
    fold_ids: np.ndarray

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": OOF_SCHEMA,
            "fold_count": len(self.folds),
            "folds": [fold.manifest() for fold in self.folds],
            "burn_in_count": int(len(self.burn_in_positions)),
            "prediction_count": int(len(self.prediction_positions)),
            "burn_in_positions_sha256": sha256_arrays(self.burn_in_positions),
            "prediction_positions_sha256": sha256_arrays(self.prediction_positions),
            "fold_ids_sha256": sha256_arrays(self.fold_ids),
            "protocol_note": (
                "Expanding OOF folds create stacking features only; they are not "
                "validation folds and never select a model or checkpoint."
            ),
        }


def make_calendar_expanding_oof_plan(
    decision_availability_ns: np.ndarray,
    target_availability_ns: np.ndarray,
    condition_ids: np.ndarray,
    *,
    decision_date_ns: np.ndarray | None = None,
    folds: int = OOF_FOLDS,
) -> CalendarOOFPlan:
    """Split unique decision times into burn-in plus five ordered blocks.

    A fold may train only on targets available strictly before its first
    prediction decision.  Grouping by timestamp prevents rows at one calendar
    instant from being divided between earlier and later folds.  When decision
    dates are supplied, rows whose contract lacks two causal fold-start prices
    are recorded as discarded warm-up rather than entering the meta learner.
    """

    decisions = np.asarray(decision_availability_ns, dtype=np.int64).reshape(-1)
    targets = np.asarray(target_availability_ns, dtype=np.int64).reshape(-1)
    contracts = np.asarray(condition_ids).astype(str).reshape(-1)
    if not (len(decisions) == len(targets) == len(contracts)) or not len(decisions):
        raise ValueError("OOF identity arrays must be non-empty and equally sized")
    decision_dates = None
    earliest_history_start_by_contract: dict[str, int] = {}
    if decision_date_ns is not None:
        decision_dates = np.asarray(decision_date_ns, dtype=np.int64).reshape(-1)
        if len(decision_dates) != len(decisions):
            raise ValueError("OOF decision dates must align with identity arrays")
        for condition_id in np.unique(contracts):
            local = decision_dates[contracts == condition_id]
            earliest_history_start_by_contract[str(condition_id)] = int(local.min()) - 63 * HOUR_NS
    if folds != 5:
        raise ValueError("Phase 6.5B requires exactly five OOF folds")
    unique_times = np.unique(decisions)
    time_blocks = np.array_split(unique_times, folds + 1)
    if any(not len(block) for block in time_blocks):
        raise ValueError("insufficient unique decision timestamps for OOF plan")
    discarded_warmup = [np.flatnonzero(np.isin(decisions, time_blocks[0])).astype(np.int64)]
    fold_rows: list[CalendarOOFFold] = []
    all_predictions: list[np.ndarray] = []
    all_fold_ids: list[np.ndarray] = []
    for fold_id, time_block in enumerate(time_blocks[1:], start=1):
        prediction = np.flatnonzero(np.isin(decisions, time_block)).astype(np.int64)
        cutoff = int(time_block.min())
        if decision_dates is not None:
            eligible = np.asarray(
                [
                    earliest_history_start_by_contract[str(contracts[position])] + HOUR_NS
                    <= cutoff
                    for position in prediction
                ],
                dtype=bool,
            )
            discarded_warmup.append(prediction[~eligible])
            prediction = prediction[eligible]
        training = np.flatnonzero(targets < cutoff).astype(np.int64)
        if not len(prediction) or not len(training):
            raise ValueError(f"OOF fold {fold_id} has an empty train or prediction population")
        if int(targets[training].max()) >= int(decisions[prediction].min()):
            raise ValueError(f"OOF fold {fold_id} violates target maturity")
        order = np.lexsort((contracts[prediction], decisions[prediction]))
        prediction = prediction[order]
        train_order = np.lexsort((contracts[training], decisions[training]))
        training = training[train_order]
        fold_rows.append(
            CalendarOOFFold(
                fold_id=fold_id,
                prediction_positions=prediction,
                training_positions=training,
                prediction_start_ns=int(decisions[prediction].min()),
                prediction_end_ns=int(decisions[prediction].max()),
                training_target_cutoff_ns=cutoff,
            )
        )
        all_predictions.append(prediction)
        all_fold_ids.append(np.full(len(prediction), fold_id, dtype=np.int8))
    prediction_positions = np.concatenate(all_predictions)
    fold_ids = np.concatenate(all_fold_ids)
    if len(np.unique(prediction_positions)) != len(prediction_positions):
        raise ValueError("an OOF row was assigned to more than one fold")
    burn_in = np.unique(np.concatenate(discarded_warmup)).astype(np.int64)
    return CalendarOOFPlan(
        folds=tuple(fold_rows),
        burn_in_positions=burn_in,
        prediction_positions=prediction_positions,
        fold_ids=fold_ids,
    )


def _load_labels(path: Path) -> dict[str, np.ndarray]:
    validate_volatility_label_bundle_files(path)
    with np.load(path, allow_pickle=False) as stored:
        return {key: np.asarray(stored[key]) for key in stored.files}


def _contract_histories(
    raw_sequences: np.ndarray,
    condition_ids: np.ndarray,
    decision_date_ns: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Replay unique hourly close histories from overlapping saved contexts."""

    sequences = np.asarray(raw_sequences, dtype=np.float64)
    contracts = np.asarray(condition_ids).astype(str)
    decisions = np.asarray(decision_date_ns, dtype=np.int64)
    if sequences.ndim != 3 or sequences.shape[1:] != (64, 5):
        raise ValueError("raw contexts must be [N,64,5]")
    if not (len(sequences) == len(contracts) == len(decisions)):
        raise ValueError("raw contexts and identities are not aligned")
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for condition_id in np.unique(contracts):
        observations: dict[int, float] = {}
        for position in np.flatnonzero(contracts == condition_id):
            timestamps = decisions[position] - np.arange(63, -1, -1, dtype=np.int64) * HOUR_NS
            closes = sequences[position, :, 3]
            for timestamp, close in zip(timestamps.tolist(), closes.tolist()):
                previous = observations.get(int(timestamp))
                if previous is not None and not np.isclose(previous, close, atol=1e-7, rtol=0.0):
                    raise ValueError(f"inconsistent replayed close for {condition_id} at {timestamp}")
                observations[int(timestamp)] = float(close)
        ordered = np.asarray(sorted(observations), dtype=np.int64)
        result[str(condition_id)] = (
            ordered,
            np.asarray([observations[int(timestamp)] for timestamp in ordered], dtype=np.float64),
        )
    return result


def _garch_predictions_for_plan(
    labels: Mapping[str, np.ndarray], plan: CalendarOOFPlan
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    histories = _contract_histories(
        labels["train_raw_sequences"],
        labels["train_condition_ids"],
        labels["train_decision_date_ns"],
    )
    ordered_predictions = np.empty(len(plan.prediction_positions), dtype=np.float64)
    output_position_by_source_position = {
        int(source_position): output_position
        for output_position, source_position in enumerate(plan.prediction_positions)
    }
    diagnostics: list[dict[str, Any]] = []
    cursor = 0
    forecaster = RawChangeGarchHorizonForecaster()
    contracts = np.asarray(labels["train_condition_ids"]).astype(str)
    for fold in plan.folds:
        for condition_id in np.unique(contracts[fold.prediction_positions]):
            local = fold.prediction_positions[contracts[fold.prediction_positions] == condition_id]
            timestamps, closes = histories[str(condition_id)]
            fit_mask = timestamps + HOUR_NS <= fold.prediction_start_ns
            if np.count_nonzero(fit_mask) < 2:
                raise ValueError(f"no causal GARCH history for fold {fold.fold_id}/{condition_id}")
            fitted_timestamps = timestamps[fit_mask]
            fitted_closes = closes[fit_mask]
            contiguous = np.diff(fitted_timestamps) == HOUR_NS
            changes = np.diff(fitted_closes)[contiguous]
            state = forecaster.fit_changes(changes, price_count=len(fitted_closes))
            forecast = forecaster.forecast(state, horizon=8)
            for position in local:
                try:
                    output_position = output_position_by_source_position[int(position)]
                except KeyError as exc:
                    raise ValueError("OOF prediction row is absent from the frozen plan") from exc
                ordered_predictions[output_position] = forecast.prediction_guarded
            diagnostics.append(
                {
                    "fold_id": fold.fold_id,
                    "condition_id": str(condition_id),
                    "prediction_rows": int(len(local)),
                    "prediction_raw": forecast.prediction_raw,
                    "prediction_guarded": forecast.prediction_guarded,
                    "capped_steps": forecast.capped_steps,
                    "state": forecast.state.to_dict(),
                }
            )
            cursor += len(local)
    if cursor != len(ordered_predictions) or not np.all(np.isfinite(ordered_predictions)):
        raise ValueError("GARCH OOF predictions are incomplete")
    return ordered_predictions, diagnostics


def _garch_evaluation_predictions(
    labels: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    histories = _contract_histories(
        labels["train_raw_sequences"],
        labels["train_condition_ids"],
        labels["train_decision_date_ns"],
    )
    test_contracts = np.asarray(labels["test_condition_ids"]).astype(str)
    output = np.empty(len(test_contracts), dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    forecaster = RawChangeGarchHorizonForecaster()
    training_forecasts: list[float] = []
    for timestamps, closes in histories.values():
        contiguous = np.diff(timestamps) == HOUR_NS
        changes = np.diff(closes)[contiguous]
        if not len(changes):
            continue
        state = forecaster.fit_changes(changes, price_count=len(closes))
        training_forecasts.append(forecaster.forecast(state, horizon=8).prediction_guarded)
    if not training_forecasts:
        raise ValueError("no training-only GARCH forecasts are available for evaluation fallback")
    pooled_fallback = float(np.median(np.asarray(training_forecasts, dtype=np.float64)))
    for condition_id in np.unique(test_contracts):
        positions = np.flatnonzero(test_contracts == condition_id)
        if condition_id not in histories:
            output[positions] = pooled_fallback
            diagnostics.append(
                {
                    "condition_id": condition_id,
                    "prediction_rows": int(len(positions)),
                    "prediction_raw": pooled_fallback,
                    "prediction_guarded": pooled_fallback,
                    "capped_steps": 0,
                    "state": None,
                    "fallback_reason": "unseen_evaluation_contract",
                    "pooled_training_contract_count": int(len(training_forecasts)),
                    "evaluation_updates_state": False,
                }
            )
            continue
        timestamps, closes = histories[condition_id]
        contiguous = np.diff(timestamps) == HOUR_NS
        changes = np.diff(closes)[contiguous]
        state = forecaster.fit_changes(changes, price_count=len(closes))
        forecast = forecaster.forecast(state, horizon=8)
        output[positions] = forecast.prediction_guarded
        diagnostics.append(
            {
                "condition_id": condition_id,
                "prediction_rows": int(len(positions)),
                "prediction_raw": forecast.prediction_raw,
                "prediction_guarded": forecast.prediction_guarded,
                "capped_steps": forecast.capped_steps,
                "state": forecast.state.to_dict(),
                "evaluation_updates_state": False,
            }
        )
    return output, diagnostics


@torch.no_grad()
def _predict(model: nn.Module, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    parts = [model(batch.to(device)).cpu().numpy() for batch in tensor.split(batch_size)]
    return np.concatenate(parts).reshape(-1).astype(np.float64) / TARGET_MULTIPLIER


def _train_lstm_oof(
    labels: Mapping[str, np.ndarray],
    plan: CalendarOOFPlan,
    run_root: Path,
    config: Phase65GarchLSTMConfig,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    sequences = np.asarray(labels["train_sequences"], dtype=np.float32)
    target = np.asarray(labels["train_realised_variance"], dtype=np.float32)
    by_epoch = {
        epoch: np.empty(len(plan.prediction_positions), dtype=np.float64)
        for epoch in config.snapshot_epochs
    }
    resources: list[dict[str, Any]] = []
    for fold in plan.folds:
        set_seed(config.seed)
        model_config = Phase6VolatilityConfig(model="raw_lstm", walk=config.walk, device=str(device))
        model = build_model(model_config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        criterion = nn.SmoothL1Loss(beta=config.smooth_l1_beta)
        train_positions = fold.training_positions
        generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(sequences[train_positions]),
                torch.from_numpy((target[train_positions] * TARGET_MULTIPLIER).reshape(-1, 1)),
            ),
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        losses: list[float] = []
        seconds: list[float] = []
        started = time.perf_counter()
        fold_root = run_root / "oof" / f"fold{fold.fold_id}"
        fold_root.mkdir(parents=True, exist_ok=False)
        for epoch in range(1, config.epochs + 1):
            tick = time.perf_counter()
            model.train()
            total = 0.0
            count = 0
            for values, expected in loader:
                values = values.to(device)
                expected = expected.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(values), expected)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite Phase 6.5B Raw-LSTM OOF loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                total += float(loss.detach()) * len(values)
                count += len(values)
            losses.append(total / max(count, 1))
            seconds.append(time.perf_counter() - tick)
            if epoch in config.snapshot_epochs:
                local_prediction = _predict(
                    model, sequences[fold.prediction_positions], config.batch_size, device
                )
                output_positions = np.flatnonzero(plan.fold_ids == fold.fold_id)
                if len(output_positions) != len(local_prediction):
                    raise ValueError("OOF LSTM output position mismatch")
                by_epoch[epoch][output_positions] = local_prediction
                checkpoint = fold_root / f"e{epoch}.pth"
                torch.save(
                    {
                        "schema_version": "phase6-5b-raw-lstm-oof-checkpoint-v1",
                        "fold_id": fold.fold_id,
                        "completed_epoch": epoch,
                        "walk": config.walk,
                        "model_state_dict": {
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        },
                        "training_positions_sha256": sha256_arrays(train_positions),
                        "prediction_positions_sha256": sha256_arrays(fold.prediction_positions),
                    },
                    checkpoint,
                )
        np.savez_compressed(
            fold_root / "history.npz",
            epochs=np.arange(1, config.epochs + 1, dtype=np.int16),
            train_loss=np.asarray(losses, dtype=np.float64),
            epoch_seconds=np.asarray(seconds, dtype=np.float64),
        )
        resources.append(
            {
                "fold_id": fold.fold_id,
                "elapsed_seconds": time.perf_counter() - started,
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            }
        )
    for epoch, values in by_epoch.items():
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise FloatingPointError(f"invalid Raw-LSTM OOF predictions at epoch {epoch}")
    return by_epoch, resources


def _load_phase6_lstm_evaluation(
    label_path: Path,
    raw_lstm_root: Path,
    labels: Mapping[str, np.ndarray],
    config: Phase65GarchLSTMConfig,
) -> dict[int, np.ndarray]:
    source_validation = validate_volatility_run(label_path, None, raw_lstm_root)
    if (
        source_validation.get("valid") is not True
        or int(source_validation.get("walk", -1)) != config.walk
        or source_validation.get("model") != "raw_lstm"
        or tuple(source_validation.get("snapshots", ())) != config.snapshot_epochs
    ):
        raise ValueError("Phase 6 Raw-LSTM source replay validation failed")
    dataset_manifest = json.loads((raw_lstm_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if dataset_manifest.get("label_sha256") != sha256_file(label_path):
        raise ValueError("Phase 6 Raw-LSTM source label hash mismatch")
    if int(dataset_manifest.get("walk", -1)) != config.walk:
        raise ValueError("Phase 6 Raw-LSTM walk mismatch")
    config_payload = json.loads((raw_lstm_root / "config.json").read_text(encoding="utf-8"))
    if config_payload.get("model") != "raw_lstm" or int(config_payload.get("seed", -1)) != 0:
        raise ValueError("Phase 6 Raw-LSTM source configuration mismatch")
    result: dict[int, np.ndarray] = {}
    for epoch in config.snapshot_epochs:
        path = raw_lstm_root / f"e{epoch}" / "predictions.npz"
        with np.load(path, allow_pickle=False) as stored:
            identity_pairs = (
                ("condition_ids", "test_condition_ids"),
                ("window_start_ns", "test_window_start_ns"),
                ("decision_date_ns", "test_decision_date_ns"),
                ("decision_availability_ns", "test_decision_availability_ns"),
                ("target_realised_variance", "test_realised_variance"),
            )
            for saved, expected in identity_pairs:
                if not np.array_equal(stored[saved], labels[expected]):
                    raise ValueError(f"Phase 6 Raw-LSTM evaluation identity mismatch: {saved}")
            result[epoch] = np.asarray(stored["prediction_realised_variance"], dtype=np.float64)
    return result


def smoke_test_phase65_garch_lstm() -> dict[str, Any]:
    prices = np.asarray([0.40, 0.41, 0.405, 0.42, 0.418, 0.43, 0.44, 0.435, 0.45])
    forecaster = RawChangeGarchHorizonForecaster()
    state = forecaster.fit(prices)
    forecast = forecaster.forecast(state, horizon=8)
    decisions = np.arange(36, dtype=np.int64) * HOUR_NS + 10 * HOUR_NS
    targets = decisions + 2 * HOUR_NS
    contracts = np.asarray([f"c{i % 3}" for i in range(36)])
    plan = make_calendar_expanding_oof_plan(decisions, targets, contracts)
    meta = StackingMetaModel.fit(
        np.linspace(0.01, 0.02, 20),
        np.linspace(0.02, 0.01, 20),
        np.linspace(0.015, 0.018, 20),
    )
    prediction = meta.predict_nonnegative(np.asarray([0.01]), np.asarray([0.02]))
    if forecast.prediction_guarded < 0.0 or prediction.shape != (1,):
        raise RuntimeError("Phase 6.5B smoke test failed")
    return {
        "valid": True,
        "garch_status": state.status,
        "horizon": len(forecast.hourly_variance_guarded),
        "oof_fold_count": len(plan.folds),
        "meta_feature_count": 3,
    }


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def run_phase65_garch_lstm(
    label_path: Path,
    raw_lstm_root: Path,
    run_root: Path,
    config: Phase65GarchLSTMConfig,
) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite Phase 6.5B run: {run_root}")
    # Resolve the requested runtime before creating any persistent run files.
    # A missing CUDA device must leave no directory that a later bootstrap can
    # mistake for a completed or replay-valid experiment.
    device = resolve_device(config.device)
    labels = _load_labels(label_path)
    plan = make_calendar_expanding_oof_plan(
        labels["train_decision_availability_ns"],
        labels["train_target_availability_ns"],
        labels["train_condition_ids"],
        decision_date_ns=labels["train_decision_date_ns"],
        folds=config.folds,
    )
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "oof").mkdir()
    write_json(run_root / "config.json", config.to_dict())
    write_json(run_root / "environment.json", environment_manifest(device))
    label_manifest = json.loads(Path(f"{label_path}.manifest.json").read_text(encoding="utf-8"))
    write_json(
        run_root / "dataset_manifest.json",
        {
            "phase": "6.5B",
            "walk": config.walk,
            "task": "future_realised_variance_h8",
            "label_path": str(label_path.resolve()),
            "label_sha256": sha256_file(label_path),
            "label_manifest_sha256": sha256_file(Path(f"{label_path}.manifest.json")),
            "raw_lstm_root": str(raw_lstm_root.resolve()),
            "raw_lstm_training_complete_sha256": sha256_file(raw_lstm_root / "training_complete.json"),
            "identity_hashes": label_manifest["identity_hashes"],
            "target_hashes": label_manifest["target_hashes"],
            "target_definition": "sum of 8 strictly future squared raw probability changes",
            "evaluation_updates_garch_state": False,
            "unseen_evaluation_contract_fallback": (
                "median guarded H=8 forecast across contract-local training-only GARCH states"
            ),
        },
    )
    write_json(run_root / "crossfit_manifest.json", plan.manifest())
    _atomic_savez(
        run_root / "oof" / "fold_assignments.npz",
        {
            "burn_in_positions": plan.burn_in_positions,
            "prediction_positions": plan.prediction_positions,
            "fold_ids": plan.fold_ids,
            "condition_ids": np.asarray(labels["train_condition_ids"])[plan.prediction_positions],
            "decision_availability_ns": np.asarray(labels["train_decision_availability_ns"])[plan.prediction_positions],
            "target_availability_ns": np.asarray(labels["train_target_availability_ns"])[plan.prediction_positions],
        },
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    garch_oof, garch_oof_diagnostics = _garch_predictions_for_plan(labels, plan)
    lstm_oof, lstm_resources = _train_lstm_oof(labels, plan, run_root, config, device)
    garch_test, garch_test_diagnostics = _garch_evaluation_predictions(labels)
    lstm_test = _load_phase6_lstm_evaluation(label_path, raw_lstm_root, labels, config)
    oof_target = np.asarray(labels["train_realised_variance"], dtype=np.float64)[plan.prediction_positions]
    _atomic_savez(
        run_root / "oof" / "base_predictions.npz",
        {
            "garch_prediction": garch_oof,
            "target_realised_variance": oof_target,
            "prediction_positions": plan.prediction_positions,
            "fold_ids": plan.fold_ids,
            **{f"raw_lstm_prediction_e{epoch}": values for epoch, values in lstm_oof.items()},
        },
    )
    write_json(
        run_root / "garch_diagnostics.json",
        {"oof": garch_oof_diagnostics, "evaluation": garch_test_diagnostics},
    )
    write_json(run_root / "oof" / "raw_lstm_resources.json", {"folds": lstm_resources})

    snapshots: list[dict[str, Any]] = []
    test_target = np.asarray(labels["test_realised_variance"], dtype=np.float64)
    identity_fields = (
        "condition_ids",
        "window_start_ns",
        "decision_date_ns",
        "decision_availability_ns",
        "target_start_ns",
        "target_end_ns",
        "target_availability_ns",
        "context_imputed_rows",
        "lifecycle_fraction",
        "lifecycle_stage",
        "categories",
        "event_families",
        "selection_ranks",
        "activity_change_count_24h",
        "future_update_count",
        "shared_row_indices",
    )
    for epoch in config.snapshot_epochs:
        meta = StackingMetaModel.fit(
            garch_oof,
            lstm_oof[epoch],
            oof_target,
            alpha=config.elasticnet_alpha,
            l1_ratio=config.elasticnet_l1_ratio,
            max_iter=config.elasticnet_max_iter,
        )
        prediction_raw = meta.predict_raw(garch_test, lstm_test[epoch])
        prediction = np.maximum(prediction_raw, 0.0)
        snapshot = run_root / f"e{epoch}"
        snapshot.mkdir()
        write_json(snapshot / "meta_scaler.json", meta.scaler_dict())
        write_json(snapshot / "meta_model.json", meta.model_dict())
        arrays = {
            "prediction_realised_variance": prediction,
            "prediction_raw": prediction_raw,
            "target_realised_variance": test_target,
            "garch_prediction": garch_test,
            "raw_lstm_prediction": lstm_test[epoch],
        }
        arrays.update({field: np.asarray(labels[f"test_{field}"]) for field in identity_fields})
        _atomic_savez(snapshot / "predictions.npz", arrays)
        metrics = {
            "phase": "6.5B",
            "walk": config.walk,
            "epoch": epoch,
            "stack": volatility_metrics(prediction, test_target),
            "garch": volatility_metrics(garch_test, test_target),
            "raw_lstm": volatility_metrics(lstm_test[epoch], test_target),
            "clipping": clipping_diagnostics(prediction_raw, prediction, test_target),
            "meta_model": meta.model_dict(),
            "predictions_sha256": sha256_file(snapshot / "predictions.npz"),
        }
        write_json(snapshot / "metrics.json", metrics)
        snapshots.append(metrics)
    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    complete = {
        "complete": True,
        "phase": "6.5B",
        "walk": config.walk,
        "snapshots": list(config.snapshot_epochs),
        "principal_epoch": 50,
        "evaluation_selected_checkpoint": False,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    write_json(run_root / "training_complete.json", complete)
    return complete


def validate_phase65_garch_lstm_run(
    label_path: Path, raw_lstm_root: Path, run_root: Path
) -> dict[str, Any]:
    config_payload = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    config_payload["snapshot_epochs"] = tuple(config_payload["snapshot_epochs"])
    config = Phase65GarchLSTMConfig(**config_payload)
    labels = _load_labels(label_path)
    plan = make_calendar_expanding_oof_plan(
        labels["train_decision_availability_ns"],
        labels["train_target_availability_ns"],
        labels["train_condition_ids"],
        decision_date_ns=labels["train_decision_date_ns"],
    )
    if json.loads((run_root / "crossfit_manifest.json").read_text(encoding="utf-8")) != plan.manifest():
        raise ValueError("Phase 6.5B crossfit manifest replay mismatch")
    lstm_test = _load_phase6_lstm_evaluation(label_path, raw_lstm_root, labels, config)
    with np.load(run_root / "oof" / "base_predictions.npz", allow_pickle=False) as stored:
        garch_oof = np.asarray(stored["garch_prediction"], dtype=np.float64)
        target = np.asarray(stored["target_realised_variance"], dtype=np.float64)
        lstm_oof = {
            epoch: np.asarray(stored[f"raw_lstm_prediction_e{epoch}"], dtype=np.float64)
            for epoch in config.snapshot_epochs
        }
    replay_garch_oof, _ = _garch_predictions_for_plan(labels, plan)
    if not np.allclose(replay_garch_oof, garch_oof, atol=1e-12, rtol=0.0):
        raise ValueError("Phase 6.5B GARCH OOF replay mismatch")
    sequences = np.asarray(labels["train_sequences"], dtype=np.float32)
    for fold in plan.folds:
        output_positions = np.flatnonzero(plan.fold_ids == fold.fold_id)
        for epoch in config.snapshot_epochs:
            checkpoint = torch.load(
                run_root / "oof" / f"fold{fold.fold_id}" / f"e{epoch}.pth",
                map_location="cpu",
                weights_only=True,
            )
            if (
                checkpoint.get("fold_id") != fold.fold_id
                or checkpoint.get("completed_epoch") != epoch
                or checkpoint.get("training_positions_sha256")
                != sha256_arrays(fold.training_positions)
                or checkpoint.get("prediction_positions_sha256")
                != sha256_arrays(fold.prediction_positions)
            ):
                raise ValueError(f"Phase 6.5B fold {fold.fold_id}/e{epoch} checkpoint metadata mismatch")
            model_config = Phase6VolatilityConfig(
                model="raw_lstm", walk=config.walk, device="cpu"
            )
            model = build_model(model_config)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            replay = _predict(
                model,
                sequences[fold.prediction_positions],
                config.batch_size,
                torch.device("cpu"),
            )
            # cuDNN and the CPU LSTM backend accumulate small recurrent
            # reduction differences.  OOF folds can amplify that drift more
            # than the full-data Phase 6 trajectory; 5e-5 is in raw realised-
            # variance units (0.5 in the 10,000x optimization unit).  Saved
            # CUDA predictions remain immutable and drive all reported metrics.
            if not np.allclose(
                replay,
                lstm_oof[epoch][output_positions],
                atol=5e-5,
                rtol=1e-5,
            ):
                raise ValueError(f"Phase 6.5B fold {fold.fold_id}/e{epoch} Raw-LSTM replay mismatch")
    garch_test, _ = _garch_evaluation_predictions(labels)
    for epoch in config.snapshot_epochs:
        meta = StackingMetaModel.fit(
            garch_oof,
            lstm_oof[epoch],
            target,
            alpha=config.elasticnet_alpha,
            l1_ratio=config.elasticnet_l1_ratio,
            max_iter=config.elasticnet_max_iter,
        )
        replay = meta.predict_nonnegative(garch_test, lstm_test[epoch])
        with np.load(run_root / f"e{epoch}" / "predictions.npz", allow_pickle=False) as stored:
            if not np.allclose(replay, stored["prediction_realised_variance"], atol=1e-12, rtol=0.0):
                raise ValueError(f"Phase 6.5B e{epoch} meta prediction replay mismatch")
            if not np.array_equal(stored["target_realised_variance"], labels["test_realised_variance"]):
                raise ValueError(f"Phase 6.5B e{epoch} target mismatch")
    return {
        "valid": True,
        "walk": config.walk,
        "snapshots": list(config.snapshot_epochs),
        "oof_rows": int(len(plan.prediction_positions)),
        "principal_epoch": 50,
    }
