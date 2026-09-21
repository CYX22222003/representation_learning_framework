"""Leakage-safe Phase 5 global-calendar walk preparation.

The builder in this module consumes the approved, bounded-fill native hourly
FinData artifact.  It does not acquire, quarantine, repair, or fill candles.
It constructs the common row population used by the Phase 5 framework and all
strict baselines.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PHASE5_SCHEMA_VERSION = 1
FEATURE_COLUMNS = ("open", "high", "low", "close", "volume")
CLASS_NAMES = ("DOWN", "STABLE", "UP")
NATIVE_STEP = pd.Timedelta(hours=1)
NATIVE_STEP_NS = int(NATIVE_STEP.value)


@dataclass(frozen=True)
class Phase5WalkSpec:
    """One fixed-duration global-calendar Phase 5 walk."""

    walk: int
    train_start: pd.Timestamp
    cutoff: pd.Timestamp
    evaluation_end: pd.Timestamp

    @classmethod
    def from_values(
        cls,
        walk: int,
        train_start: str | pd.Timestamp,
        cutoff: str | pd.Timestamp,
        evaluation_end: str | pd.Timestamp,
    ) -> "Phase5WalkSpec":
        result = cls(
            walk=int(walk),
            train_start=_utc(train_start),
            cutoff=_utc(cutoff),
            evaluation_end=_utc(evaluation_end),
        )
        if not result.train_start < result.cutoff < result.evaluation_end:
            raise ValueError("walk boundaries must satisfy train_start < cutoff < evaluation_end")
        return result


@dataclass(frozen=True)
class Phase5PreparedBundle:
    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def normalize_phase5_candles(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the approved hourly bounded-fill schema and return sorted rows."""

    required = {
        "condition_id",
        "interval_minutes",
        "date",
        *FEATURE_COLUMNS,
        "native_resolution",
        "is_observed",
        "is_imputed",
        "original_gap_length_bars",
        "time_since_last_observation",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Phase 5 candle artifact is missing columns: {missing}")

    result = frame.copy()
    result["condition_id"] = result["condition_id"].astype(str)
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
    result = result.sort_values(["condition_id", "date"], kind="stable").reset_index(drop=True)
    if result.duplicated(["condition_id", "date"]).any():
        raise ValueError("duplicate condition_id/date identities")
    if not result["interval_minutes"].eq(60).all() or not result["native_resolution"].eq("1h").all():
        raise ValueError("Phase 5 primary input must be native one-hour candles")

    numeric = result.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("OHLCV contains non-finite values")
    if (result["volume"] < 0).any():
        raise ValueError("volume must be non-negative")
    invalid_ohlc = (
        result["high"].lt(result[["open", "close", "low"]].max(axis=1))
        | result["low"].gt(result[["open", "close", "high"]].min(axis=1))
    )
    if invalid_ohlc.any():
        raise ValueError("OHLC rows violate high/low consistency")

    observed = result["is_observed"].astype(bool)
    imputed = result["is_imputed"].astype(bool)
    if not (observed ^ imputed).all():
        raise ValueError("every candle must be exactly one of observed or imputed")
    if not result.loc[imputed, "volume"].eq(0.0).all():
        raise ValueError("is_imputed -> volume == 0 invariant failed")
    if not result.loc[observed, "volume"].gt(0.0).all():
        raise ValueError("is_observed -> volume > 0 invariant failed")
    if not result.loc[imputed, ["open", "high", "low"]].eq(
        result.loc[imputed, "close"], axis=0
    ).all().all():
        raise ValueError("imputed candles must have flat OHLC")
    if not result.loc[imputed, "original_gap_length_bars"].eq(1).all():
        raise ValueError("imputed candles must represent exactly one missing bar")
    if not result.loc[imputed, "time_since_last_observation"].eq(60).all():
        raise ValueError("imputed hourly candles must be 60 minutes after the prior observation")
    if not result.loc[observed, "original_gap_length_bars"].eq(0).all():
        raise ValueError("observed candles must have original_gap_length_bars == 0")
    if not result.loc[observed, "time_since_last_observation"].eq(0).all():
        raise ValueError("observed candles must have time_since_last_observation == 0")

    for condition_id, contract in result.groupby("condition_id", sort=False):
        contract = contract.reset_index(drop=True)
        differences = contract["date"].diff().dropna()
        if (differences <= pd.Timedelta(0)).any() or (differences % NATIVE_STEP != pd.Timedelta(0)).any():
            raise ValueError(f"{condition_id}: timestamps are off the hourly grid")
        # A two-hour difference is one complete isolated gap and must already
        # have been materialized by the approved bounded-fill artifact.
        if differences.eq(2 * NATIVE_STEP).any():
            raise ValueError(f"{condition_id}: unfilled isolated one-hour gap")
        imputed_positions = np.flatnonzero(contract["is_imputed"].to_numpy(bool))
        for position in imputed_positions:
            if position == 0 or position == len(contract) - 1:
                raise ValueError(f"{condition_id}: imputed candle is at a contract boundary")
            previous = contract.iloc[position - 1]
            current = contract.iloc[position]
            following = contract.iloc[position + 1]
            if not bool(previous["is_observed"]) or not bool(following["is_observed"]):
                raise ValueError(
                    f"{condition_id}: imputed candle is not isolated between observed neighbors"
                )
            if (
                current["date"] - previous["date"] != NATIVE_STEP
                or following["date"] - current["date"] != NATIVE_STEP
            ):
                raise ValueError(
                    f"{condition_id}: imputed candle does not complete one isolated hourly gap"
                )
            preceding_close = float(previous["close"])
            if not np.all(
                current.loc[["open", "high", "low", "close"]].to_numpy(np.float64)
                == preceding_close
            ):
                raise ValueError(
                    f"{condition_id}: imputed OHLC does not equal the preceding observed close"
                )
    return result


def normalize_phase5_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"condition_id", "category", "event_family", "selection_rank"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"market metadata is missing columns: {missing}")
    result = metadata.copy()
    result["condition_id"] = result["condition_id"].astype(str)
    if result["condition_id"].duplicated().any():
        raise ValueError("market metadata has duplicate condition_id rows")
    for column in ("start_date_search", "end_date_search"):
        if column in result:
            result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    return result.set_index("condition_id", drop=False)


def _with_contract_state(frame: pd.DataFrame, activity_hours: int) -> pd.DataFrame:
    if activity_hours <= 0:
        raise ValueError("activity_hours must be positive")
    result = frame.copy()
    differences = result["date"].diff()
    result["segment"] = differences.ne(NATIVE_STEP).cumsum().astype(np.int32) - 1
    result["availability"] = result["date"] + NATIVE_STEP
    result["price_changed"] = (
        result.groupby("segment", sort=False)["close"].diff().abs().fillna(0.0).gt(0.0)
    )
    activity_count = (
        result.groupby("segment", sort=False)["price_changed"]
        .rolling(activity_hours, min_periods=activity_hours)
        .sum()
        .reset_index(level=0, drop=True)
    )
    result["activity_change_count"] = activity_count.fillna(0.0).astype(np.int16)
    result["active"] = activity_count.gt(0.0).fillna(False)
    return result


def _lifecycle_values(
    decision_dates: np.ndarray,
    metadata_row: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(decision_dates)
    unknown_fraction = np.full(count, np.nan, dtype=np.float32)
    unknown_stage = np.full(count, -1, dtype=np.int8)
    start = metadata_row.get("start_date_search")
    end = metadata_row.get("end_date_search")
    if pd.isna(start) or pd.isna(end) or not pd.Timestamp(start) < pd.Timestamp(end):
        return unknown_fraction, unknown_stage
    start_ns = int(pd.Timestamp(start).value)
    duration = int(pd.Timestamp(end).value) - start_ns
    fractions = np.clip((decision_dates.astype(np.int64) - start_ns) / duration, 0.0, 1.0)
    stages = np.minimum((fractions * 3.0).astype(np.int8), 2)
    return fractions.astype(np.float32), stages


def _candidate_records(
    candles: pd.DataFrame,
    metadata: pd.DataFrame,
    spec: Phase5WalkSpec,
    *,
    seq_len: int,
    horizon: int,
    activity_hours: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build encoder, supervised-train, and evaluation rows independently.

    Encoder eligibility is deliberately target-free: only context and decision
    information available before the walk cutoff may affect that population.
    Target existence, segment continuity, observation status, and maturity are
    applied only to supervised train/evaluation populations.
    """

    encoder_records: list[dict[str, Any]] = []
    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    for condition_id, raw_contract in candles.groupby("condition_id", sort=True):
        if condition_id not in metadata.index:
            raise ValueError(f"missing market metadata for {condition_id}")
        contract = _with_contract_state(raw_contract.reset_index(drop=True), activity_hours)
        if len(contract) < seq_len:
            continue
        decision = np.arange(seq_len - 1, len(contract), dtype=np.int64)
        start = decision - seq_len + 1
        segments = contract["segment"].to_numpy(np.int32)
        observed = contract["is_observed"].to_numpy(bool)
        context_valid = (
            (segments[start] == segments[decision])
            & observed[decision]
            & contract["active"].to_numpy(bool)[decision]
        )
        for start_idx, decision_idx in zip(
            start[context_valid], decision[context_valid], strict=True
        ):
            start_idx = int(start_idx)
            decision_idx = int(decision_idx)
            window_start = contract.at[int(start_idx), "date"]
            decision_availability = contract.at[int(decision_idx), "availability"]
            base_record = {
                "condition_id": condition_id,
                "contract": contract,
                "start": start_idx,
                "decision": decision_idx,
                "metadata": metadata.loc[condition_id],
            }

            # Encoder rows are decided without constructing or inspecting a
            # future target. This is the strict walk-cutoff boundary.
            if (
                window_start >= spec.train_start
                and decision_availability < spec.cutoff
            ):
                encoder_records.append(base_record)

            # Split downstream candidates by decision-time availability before
            # constructing or inspecting their targets.
            decision_partition: str | None = None
            if window_start >= spec.train_start and decision_availability < spec.cutoff:
                decision_partition = "train"
            elif (
                window_start >= spec.train_start
                and spec.cutoff <= decision_availability < spec.evaluation_end
            ):
                decision_partition = "test"
            if decision_partition is None:
                continue

            # Target rules belong only to the already-partitioned downstream
            # populations.
            target_idx = decision_idx + horizon
            if target_idx >= len(contract):
                continue
            if (
                segments[decision_idx] != segments[target_idx]
                or not observed[target_idx]
            ):
                continue
            target_availability = contract.at[target_idx, "availability"]
            supervised_record = {**base_record, "target": int(target_idx)}
            if decision_partition == "train" and target_availability < spec.cutoff:
                train_records.append(supervised_record)
            elif (
                decision_partition == "test"
                and target_availability < spec.evaluation_end
            ):
                test_records.append(supervised_record)

    supported_contracts = {record["condition_id"] for record in train_records}
    test_records = [
        record for record in test_records if record["condition_id"] in supported_contracts
    ]
    return encoder_records, train_records, test_records


def _fit_volume_scaler(candles: pd.DataFrame, spec: Phase5WalkSpec) -> dict[str, Any]:
    availability = candles["date"] + NATIVE_STEP
    fit_mask = candles["date"].ge(spec.train_start) & availability.lt(spec.cutoff)
    values = candles.loc[fit_mask, "volume"].to_numpy(dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("walk has no finite training-interval volume rows")
    mean = float(values.mean())
    std = float(values.std())
    denominator = std if std > 1e-8 else 1.0
    return {
        "method": "global_walk_training_interval_volume_zscore",
        "fit_population": "unique bounded-fill candles with date >= train_start and availability < cutoff",
        "fit_row_count": int(len(values)),
        "mean": mean,
        "std": std,
        "denominator": denominator,
        "imputed_raw_value": 0.0,
        "imputed_scaled_value": float((0.0 - mean) / denominator),
        "uses_evaluation_rows": False,
    }


def _empty_split(seq_len: int, *, labels: bool) -> dict[str, np.ndarray]:
    result = {
        "raw_sequences": np.empty((0, seq_len, len(FEATURE_COLUMNS)), dtype=np.float32),
        "sequences": np.empty((0, seq_len, len(FEATURE_COLUMNS)), dtype=np.float32),
        "condition_ids": np.empty(0, dtype="<U1"),
        "window_start_ns": np.empty(0, dtype=np.int64),
        "decision_date_ns": np.empty(0, dtype=np.int64),
        "decision_availability_ns": np.empty(0, dtype=np.int64),
        "context_is_imputed": np.empty((0, seq_len), dtype=bool),
        "context_is_observed": np.empty((0, seq_len), dtype=bool),
        "context_original_gap_length_bars": np.empty((0, seq_len), dtype=np.int16),
        "context_time_since_observation": np.empty((0, seq_len), dtype=np.int16),
        "context_imputed_rows": np.empty(0, dtype=np.int16),
        "activity_change_count_24h": np.empty(0, dtype=np.int16),
        "lifecycle_fraction": np.empty(0, dtype=np.float32),
        "lifecycle_stage": np.empty(0, dtype=np.int8),
        "categories": np.empty(0, dtype="<U1"),
        "event_families": np.empty(0, dtype="<U1"),
        "selection_ranks": np.empty(0, dtype=np.int16),
    }
    if labels:
        result.update(
            {
                "target_date_ns": np.empty(0, dtype=np.int64),
                "target_availability_ns": np.empty(0, dtype=np.int64),
                "current_close": np.empty(0, dtype=np.float64),
                "target_close": np.empty(0, dtype=np.float64),
                "regression_labels": np.empty(0, dtype=np.float64),
                "classification_labels": np.empty(0, dtype=np.int64),
            }
        )
    return result


def _materialize(
    records: list[dict[str, Any]],
    *,
    seq_len: int,
    tau: float,
    volume_scaler: Mapping[str, Any],
    labels: bool,
) -> dict[str, np.ndarray]:
    if not records:
        return _empty_split(seq_len, labels=labels)
    sequences: list[np.ndarray] = []
    raw_sequences: list[np.ndarray] = []
    condition_ids: list[str] = []
    window_start_ns: list[int] = []
    decision_date_ns: list[int] = []
    decision_availability_ns: list[int] = []
    masks: list[np.ndarray] = []
    time_since: list[np.ndarray] = []
    original_gap_lengths: list[np.ndarray] = []
    categories: list[str] = []
    families: list[str] = []
    ranks: list[int] = []
    lifecycle_fractions: list[float] = []
    lifecycle_stages: list[int] = []
    activity_counts: list[int] = []
    target_date_ns: list[int] = []
    target_availability_ns: list[int] = []
    current_close: list[float] = []
    target_close: list[float] = []

    mean = float(volume_scaler["mean"])
    denominator = float(volume_scaler["denominator"])
    for record in records:
        contract = record["contract"]
        start = record["start"]
        decision = record["decision"]
        target = record.get("target")
        window = contract.loc[start:decision, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        raw_sequences.append(window.astype(np.float32))
        window[:, 4] = (window[:, 4] - mean) / denominator
        sequences.append(window.astype(np.float32))
        condition_ids.append(record["condition_id"])
        window_start_ns.append(int(contract.at[start, "date"].value))
        decision_date_ns.append(int(contract.at[decision, "date"].value))
        decision_availability_ns.append(int(contract.at[decision, "availability"].value))
        mask = contract.loc[start:decision, "is_imputed"].to_numpy(bool)
        masks.append(mask)
        original_gap_lengths.append(
            contract.loc[start:decision, "original_gap_length_bars"].to_numpy(np.int16)
        )
        time_since.append(
            contract.loc[start:decision, "time_since_last_observation"].to_numpy(np.int16)
        )
        metadata_row = record["metadata"]
        categories.append(str(metadata_row.get("category", "unknown")))
        families.append(str(metadata_row.get("event_family", "unknown")))
        ranks.append(int(metadata_row.get("selection_rank", -1)))
        fraction, stage = _lifecycle_values(
            np.asarray([decision_date_ns[-1]], dtype=np.int64), metadata_row
        )
        lifecycle_fractions.append(float(fraction[0]))
        lifecycle_stages.append(int(stage[0]))
        activity_counts.append(int(contract.at[decision, "activity_change_count"]))
        if labels:
            if target is None:
                raise ValueError("labelled Phase 5 row is missing its downstream target")
            target = int(target)
            target_date_ns.append(int(contract.at[target, "date"].value))
            target_availability_ns.append(int(contract.at[target, "availability"].value))
            current_close.append(float(contract.at[decision, "close"]))
            target_close.append(float(contract.at[target, "close"]))

    result: dict[str, np.ndarray] = {
        "raw_sequences": np.stack(raw_sequences).astype(np.float32),
        "sequences": np.stack(sequences).astype(np.float32),
        "condition_ids": np.asarray(condition_ids, dtype=np.str_),
        "window_start_ns": np.asarray(window_start_ns, dtype=np.int64),
        "decision_date_ns": np.asarray(decision_date_ns, dtype=np.int64),
        "decision_availability_ns": np.asarray(decision_availability_ns, dtype=np.int64),
        "context_is_imputed": np.stack(masks).astype(bool),
        "context_is_observed": ~np.stack(masks).astype(bool),
        "context_original_gap_length_bars": np.stack(original_gap_lengths).astype(np.int16),
        "context_time_since_observation": np.stack(time_since).astype(np.int16),
        "context_imputed_rows": np.asarray([mask.sum() for mask in masks], dtype=np.int16),
        "activity_change_count_24h": np.asarray(activity_counts, dtype=np.int16),
        "lifecycle_fraction": np.asarray(lifecycle_fractions, dtype=np.float32),
        "lifecycle_stage": np.asarray(lifecycle_stages, dtype=np.int8),
        "categories": np.asarray(categories, dtype=np.str_),
        "event_families": np.asarray(families, dtype=np.str_),
        "selection_ranks": np.asarray(ranks, dtype=np.int16),
    }
    if labels:
        current = np.asarray(current_close, dtype=np.float64)
        future = np.asarray(target_close, dtype=np.float64)
        delta = future - current
        classes = np.where(delta < -tau, 0, np.where(np.abs(delta) <= tau, 1, 2))
        result.update(
            {
                "target_date_ns": np.asarray(target_date_ns, dtype=np.int64),
                "target_availability_ns": np.asarray(target_availability_ns, dtype=np.int64),
                "current_close": current,
                "target_close": future,
                "regression_labels": delta,
                "classification_labels": classes.astype(np.int64),
            }
        )
    return result


def _prefix_arrays(prefix: str, values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def build_phase5_walk_bundle(
    candles: pd.DataFrame,
    metadata: pd.DataFrame,
    spec: Phase5WalkSpec,
    *,
    seq_len: int = 64,
    horizon: int = 2,
    tau: float = 0.001,
    activity_hours: int = 24,
    task_role: str = "primary_h2",
    source_provenance: Mapping[str, Any] | None = None,
) -> Phase5PreparedBundle:
    """Build encoder-training, supervised-training, and evaluation arrays."""

    expected_horizon = {"primary_h2": 2, "exploratory_raw_delta_h8": 8}.get(task_role)
    if expected_horizon is None:
        raise ValueError("unknown Phase 5 task role")
    if (
        seq_len != 64
        or horizon != expected_horizon
        or tau != 0.001
        or activity_hours != 24
    ):
        raise ValueError(
            f"Phase 5 {task_role} contract is fixed to "
            f"seq64/h{expected_horizon}/tau0.001/activity24h"
        )
    clean = normalize_phase5_candles(candles)
    market_metadata = normalize_phase5_metadata(metadata)
    source_contracts = set(clean["condition_id"].unique())
    if not source_contracts.issubset(set(market_metadata.index)):
        missing = sorted(source_contracts - set(market_metadata.index))
        raise ValueError(f"metadata does not cover candle contracts: {missing}")
    volume_scaler = _fit_volume_scaler(clean, spec)
    encoder_records, train_records, test_records = _candidate_records(
        clean,
        market_metadata,
        spec,
        seq_len=seq_len,
        horizon=horizon,
        activity_hours=activity_hours,
    )
    if not encoder_records or not train_records or not test_records:
        raise ValueError("walk must produce non-empty encoder_train, train, and test populations")

    split_values = {
        "encoder_train": _materialize(
            encoder_records, seq_len=seq_len, tau=tau, volume_scaler=volume_scaler, labels=False
        ),
        "train": _materialize(
            train_records, seq_len=seq_len, tau=tau, volume_scaler=volume_scaler, labels=True
        ),
        "test": _materialize(
            test_records, seq_len=seq_len, tau=tau, volume_scaler=volume_scaler, labels=True
        ),
    }
    arrays: dict[str, np.ndarray] = {}
    for split, values in split_values.items():
        arrays.update(_prefix_arrays(split, values))

    identity_fields = (
        "condition_ids",
        "window_start_ns",
        "decision_date_ns",
        "decision_availability_ns",
    )
    identity_hashes = {
        split: sha256_arrays(*(split_values[split][field] for field in identity_fields))
        for split in split_values
    }
    sequence_hashes = {
        split: sha256_arrays(split_values[split]["sequences"])
        for split in split_values
    }
    raw_sequence_hashes = {
        split: sha256_arrays(split_values[split]["raw_sequences"])
        for split in split_values
    }
    label_hashes = {
        split: sha256_arrays(
            split_values[split]["target_date_ns"],
            split_values[split]["regression_labels"],
            split_values[split]["classification_labels"],
        )
        for split in ("train", "test")
    }

    contracts: list[dict[str, Any]] = []
    for condition_id in sorted(source_contracts):
        row = market_metadata.loc[condition_id]
        contracts.append(
            {
                "condition_id": condition_id,
                "selection_rank": int(row["selection_rank"]),
                "category": str(row["category"]),
                "event_family": str(row["event_family"]),
                "filled_rows": int(clean["condition_id"].eq(condition_id).sum()),
                "encoder_train_rows": int(
                    np.count_nonzero(split_values["encoder_train"]["condition_ids"] == condition_id)
                ),
                "train_rows": int(
                    np.count_nonzero(split_values["train"]["condition_ids"] == condition_id)
                ),
                "test_rows": int(
                    np.count_nonzero(split_values["test"]["condition_ids"] == condition_id)
                ),
            }
        )

    manifest = {
        "schema_version": PHASE5_SCHEMA_VERSION,
        "phase": 5,
        "purpose": "walk_specific_train_test_data",
        "task_role": task_role,
        "pipeline": "global_calendar_walk_first",
        "walk": spec.walk,
        "intervals": {
            "training": {
                "start_inclusive": spec.train_start.isoformat(),
                "cutoff_exclusive": spec.cutoff.isoformat(),
            },
            "evaluation": {
                "start_inclusive": spec.cutoff.isoformat(),
                "end_exclusive": spec.evaluation_end.isoformat(),
            },
        },
        "native_resolution": "1h",
        "candle_timestamp_semantics": "bar_start",
        "availability_rule": "date + 1h",
        "seq_len": seq_len,
        "horizon_bars": horizon,
        "horizon_hours": horizon,
        "activity_hours": activity_hours,
        "activity_rule": "at least one non-zero close change in the 24 intervals ending at decision",
        "feature_columns": list(FEATURE_COLUMNS),
        "classification": {"tau": tau, "class_names": list(CLASS_NAMES)},
        "gap_policy": {
            "input": "approved candles_1h_clean_ffill1.parquet",
            "isolated_one_bar_gaps": "already filled with flat OHLC and zero volume",
            "longer_gaps": "sequence_break",
            "decision_endpoint": "observed_only",
            "target_endpoint": "observed_only",
        },
        "evaluation_context_policy": "historical_pre_cutoff_context_allowed",
        "evaluation_contract_support_rule": (
            "condition must contribute at least one active supervised training row with a "
            "target mature before the walk cutoff"
        ),
        "row_population_rules": {
            "encoder_train": {
                "uses": [
                    "contiguous context",
                    "observed decision endpoint",
                    "causal prior-24h activity",
                    "rolling training start",
                    "decision availability strictly before cutoff",
                ],
                "target_dependency": "none",
            },
            "train": {
                "base": "encoder context/decision eligibility",
                "adds": [
                    f"{horizon}-hour target exists",
                    "target remains in the same segment",
                    "target endpoint is observed",
                    "target availability strictly before cutoff",
                ],
            },
            "test": {
                "base": "context/decision eligibility",
                "adds": [
                    "decision availability at or after cutoff",
                    f"{horizon}-hour target exists",
                    "target remains in the same segment",
                    "target endpoint is observed",
                    "target availability strictly before evaluation end",
                    "contract has active supervised training support",
                ],
            },
        },
        "preprocessing": {"volume": volume_scaler, "ohlc": "unscaled_probability"},
        "row_counts": {
            split: int(len(split_values[split]["sequences"])) for split in split_values
        },
        "contract_counts": {
            split: int(len(np.unique(split_values[split]["condition_ids"])))
            for split in split_values
        },
        "class_counts": {
            split: {
                name: int(np.count_nonzero(split_values[split]["classification_labels"] == index))
                for index, name in enumerate(CLASS_NAMES)
            }
            for split in ("train", "test")
        },
        "zero_delta_counts": {
            split: int(np.count_nonzero(split_values[split]["regression_labels"] == 0.0))
            for split in ("train", "test")
        },
        "imputation_exposure_counts": {
            split: int(np.count_nonzero(split_values[split]["context_imputed_rows"] > 0))
            for split in split_values
        },
        "identity_hashes": identity_hashes,
        "sequence_hashes": sequence_hashes,
        "raw_sequence_hashes": raw_sequence_hashes,
        "label_hashes": label_hashes,
        "source_provenance": dict(source_provenance or {}),
        "accepted_assumptions": {
            "retrospective_cohort_selection": True,
            "approved_final_pruning_is_offline_cleaning": True,
            "quarantine_availability_replay_required": False,
        },
        "lifecycle_metadata": {
            "role": "reporting_only",
            "fraction": "clipped elapsed fraction between retrospective market start/end metadata",
            "stage_codes": {"0": "early", "1": "middle", "2": "late", "-1": "unknown"},
            "used_for_eligibility_or_fitting": False,
        },
        "contracts": contracts,
    }
    validate_phase5_arrays(arrays, manifest)
    return Phase5PreparedBundle(arrays=arrays, manifest=manifest)


def validate_phase5_arrays(arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Replay bundle-internal Phase 5 identity, boundary, and label invariants."""

    if manifest.get("schema_version") != PHASE5_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 5 bundle schema")
    if manifest.get("phase") != 5 or manifest.get("purpose") != "walk_specific_train_test_data":
        raise ValueError("manifest is not a Phase 5 walk data artifact")
    if manifest.get("row_population_rules", {}).get("encoder_train", {}).get(
        "target_dependency"
    ) != "none":
        raise ValueError("encoder_train manifest must explicitly declare no target dependency")
    seq_len = int(manifest["seq_len"])
    horizon = int(manifest["horizon_bars"])
    tau = float(manifest["classification"]["tau"])
    cutoff = int(_utc(manifest["intervals"]["training"]["cutoff_exclusive"]).value)
    train_start = int(_utc(manifest["intervals"]["training"]["start_inclusive"]).value)
    evaluation_end = int(_utc(manifest["intervals"]["evaluation"]["end_exclusive"]).value)
    sentinel = float(manifest["preprocessing"]["volume"]["imputed_scaled_value"])

    identity_fields = (
        "condition_ids",
        "window_start_ns",
        "decision_date_ns",
        "decision_availability_ns",
    )
    for split in ("encoder_train", "train", "test"):
        required = {
            f"{split}_sequences",
            f"{split}_raw_sequences",
            f"{split}_context_is_imputed",
            f"{split}_context_is_observed",
            f"{split}_context_original_gap_length_bars",
            f"{split}_context_time_since_observation",
            f"{split}_context_imputed_rows",
            f"{split}_activity_change_count_24h",
            *(f"{split}_{field}" for field in identity_fields),
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"{split} arrays missing: {missing}")
        sequences = np.asarray(arrays[f"{split}_sequences"])
        raw_sequences = np.asarray(arrays[f"{split}_raw_sequences"])
        count = len(sequences)
        if count != int(manifest["row_counts"][split]):
            raise ValueError(f"{split} row count disagrees with manifest")
        if sequences.dtype != np.float32 or sequences.shape[1:] != (seq_len, len(FEATURE_COLUMNS)):
            raise ValueError(f"{split}_sequences must be float32 [N,{seq_len},5]")
        if not np.isfinite(sequences).all():
            raise ValueError(f"{split}_sequences contains non-finite values")
        if raw_sequences.dtype != np.float32 or raw_sequences.shape != sequences.shape:
            raise ValueError(f"{split}_raw_sequences shape or dtype mismatch")
        if not np.isfinite(raw_sequences).all():
            raise ValueError(f"{split}_raw_sequences contains non-finite values")
        for field in identity_fields:
            if len(arrays[f"{split}_{field}"]) != count:
                raise ValueError(f"{split}_{field} length mismatch")
        mask = np.asarray(arrays[f"{split}_context_is_imputed"])
        observed = np.asarray(arrays[f"{split}_context_is_observed"])
        gap_lengths = np.asarray(arrays[f"{split}_context_original_gap_length_bars"])
        time_since = np.asarray(arrays[f"{split}_context_time_since_observation"])
        if (
            mask.shape != (count, seq_len)
            or observed.shape != (count, seq_len)
            or gap_lengths.shape != (count, seq_len)
            or time_since.shape != (count, seq_len)
        ):
            raise ValueError(f"{split} context metadata shape mismatch")
        if not np.array_equal(observed, ~mask):
            raise ValueError(f"{split} observed/imputed context masks disagree")
        if not np.array_equal(gap_lengths, mask.astype(np.int16)):
            raise ValueError(f"{split} original gap lengths do not replay")
        if not np.array_equal(mask.sum(axis=1), arrays[f"{split}_context_imputed_rows"]):
            raise ValueError(f"{split} imputation counts do not replay")
        if int(np.count_nonzero(mask.any(axis=1))) != int(
            manifest["imputation_exposure_counts"][split]
        ):
            raise ValueError(f"{split} imputation exposure disagrees with manifest")
        if not np.all(time_since[mask] == 60) or not np.all(time_since[~mask] == 0):
            raise ValueError(f"{split} time-since-observation metadata is invalid")
        if not np.allclose(sequences[:, :, 4][mask], sentinel, rtol=0.0, atol=1e-6):
            raise ValueError(f"{split} imputed volume sentinel does not replay")
        if not np.all(raw_sequences[:, :, 4][mask] == 0.0):
            raise ValueError(f"{split} raw imputed volume is not zero")
        if not np.all(raw_sequences[:, :, 4][~mask] > 0.0):
            raise ValueError(f"{split} raw observed volume is not positive")
        if not np.array_equal(sequences[:, :, :4], raw_sequences[:, :, :4]):
            raise ValueError(f"{split} OHLC changed during preprocessing")
        expected_scaled_volume = (
            raw_sequences[:, :, 4].astype(np.float64)
            - float(manifest["preprocessing"]["volume"]["mean"])
        ) / float(manifest["preprocessing"]["volume"]["denominator"])
        if not np.allclose(sequences[:, :, 4], expected_scaled_volume, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{split} volume scaling does not replay")
        if mask[:, -1].any():
            raise ValueError(f"{split} contains an imputed decision endpoint")
        decision = np.asarray(arrays[f"{split}_decision_date_ns"], dtype=np.int64)
        decision_availability = np.asarray(
            arrays[f"{split}_decision_availability_ns"], dtype=np.int64
        )
        if not np.array_equal(decision_availability, decision + NATIVE_STEP_NS):
            raise ValueError(f"{split} decision availability mismatch")
        if np.any(np.asarray(arrays[f"{split}_window_start_ns"]) < train_start):
            raise ValueError(f"{split} window begins before the rolling training start")
        # Activity is evaluated on the authoritative float64 source candles.
        # The count is stored because extremely small source movements can
        # legitimately collapse when model inputs are encoded as float32.
        if np.any(np.asarray(arrays[f"{split}_activity_change_count_24h"]) <= 0):
            raise ValueError(f"{split} contains a row that fails active_24h")
        identity = sha256_arrays(*(arrays[f"{split}_{field}"] for field in identity_fields))
        if identity != manifest["identity_hashes"][split]:
            raise ValueError(f"{split} identity hash mismatch")
        if sha256_arrays(sequences) != manifest["sequence_hashes"][split]:
            raise ValueError(f"{split} sequence hash mismatch")
        if sha256_arrays(raw_sequences) != manifest["raw_sequence_hashes"][split]:
            raise ValueError(f"{split} raw sequence hash mismatch")
        if split == "encoder_train" and np.any(decision_availability >= cutoff):
            raise ValueError("encoder_train reaches or crosses the walk cutoff")
        if len(np.unique(arrays[f"{split}_condition_ids"])) != int(
            manifest["contract_counts"][split]
        ):
            raise ValueError(f"{split} contract count disagrees with manifest")

    for split in ("train", "test"):
        required = {
            f"{split}_target_date_ns",
            f"{split}_target_availability_ns",
            f"{split}_current_close",
            f"{split}_target_close",
            f"{split}_regression_labels",
            f"{split}_classification_labels",
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"{split} label arrays missing: {missing}")
        decision = np.asarray(arrays[f"{split}_decision_date_ns"], dtype=np.int64)
        target = np.asarray(arrays[f"{split}_target_date_ns"], dtype=np.int64)
        target_availability = np.asarray(arrays[f"{split}_target_availability_ns"], dtype=np.int64)
        if not np.array_equal(target, decision + horizon * NATIVE_STEP_NS):
            raise ValueError(
                f"{split} target is not the exact {horizon}-hour contract-local endpoint"
            )
        if not np.array_equal(target_availability, target + NATIVE_STEP_NS):
            raise ValueError(f"{split} target availability mismatch")
        replay_delta = (
            np.asarray(arrays[f"{split}_target_close"], dtype=np.float64)
            - np.asarray(arrays[f"{split}_current_close"], dtype=np.float64)
        )
        if not np.array_equal(replay_delta, arrays[f"{split}_regression_labels"]):
            raise ValueError(f"{split} regression labels do not replay")
        replay_class = np.where(
            replay_delta < -tau, 0, np.where(np.abs(replay_delta) <= tau, 1, 2)
        ).astype(np.int64)
        if not np.array_equal(replay_class, arrays[f"{split}_classification_labels"]):
            raise ValueError(f"{split} classification labels do not replay")
        label_hash = sha256_arrays(
            target,
            arrays[f"{split}_regression_labels"],
            arrays[f"{split}_classification_labels"],
        )
        if label_hash != manifest["label_hashes"][split]:
            raise ValueError(f"{split} label hash mismatch")
        if split == "train" and np.any(target_availability >= cutoff):
            raise ValueError("training target matures at or after the cutoff")
        if split == "test":
            decision_availability = np.asarray(
                arrays[f"{split}_decision_availability_ns"], dtype=np.int64
            )
            if np.any(decision_availability < cutoff):
                raise ValueError("evaluation decision occurs before the cutoff")
            if np.any(target_availability >= evaluation_end):
                raise ValueError("evaluation target matures at or after the evaluation end")
        actual_class_counts = {
            name: int(np.count_nonzero(arrays[f"{split}_classification_labels"] == index))
            for index, name in enumerate(CLASS_NAMES)
        }
        if actual_class_counts != manifest["class_counts"][split]:
            raise ValueError(f"{split} class counts disagree with manifest")
        if int(np.count_nonzero(replay_delta == 0.0)) != int(
            manifest["zero_delta_counts"][split]
        ):
            raise ValueError(f"{split} zero-delta count disagrees with manifest")

    train_identity = set(
        zip(
            arrays["train_condition_ids"].tolist(),
            arrays["train_decision_date_ns"].tolist(),
            strict=True,
        )
    )
    test_identity = set(
        zip(
            arrays["test_condition_ids"].tolist(),
            arrays["test_decision_date_ns"].tolist(),
            strict=True,
        )
    )
    if train_identity & test_identity:
        raise ValueError("train and evaluation identities overlap")
    encoder_identity = set(
        zip(
            arrays["encoder_train_condition_ids"].tolist(),
            arrays["encoder_train_decision_date_ns"].tolist(),
            strict=True,
        )
    )
    if not train_identity.issubset(encoder_identity):
        raise ValueError("supervised training rows are not a subset of encoder training rows")
    if not set(arrays["test_condition_ids"].tolist()).issubset(
        set(arrays["train_condition_ids"].tolist())
    ):
        raise ValueError("evaluation contains a contract without active supervised training support")
    return {
        "valid": True,
        "walk": int(manifest["walk"]),
        "row_counts": dict(manifest["row_counts"]),
        "identity_hashes": dict(manifest["identity_hashes"]),
        "label_hashes": dict(manifest["label_hashes"]),
    }


def validate_phase5_bundle_files(npz_path: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    """Load a saved bundle and verify its hashes, source lineage, and arrays."""

    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path or f"{npz_path}.manifest.json")
    if not npz_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Phase 5 NPZ and companion manifest are both required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Phase 5 manifest must be a JSON object")
    expected_npz = manifest.get("artifact", {}).get("sha256")
    actual_npz = sha256_file(npz_path)
    if expected_npz != actual_npz:
        raise ValueError("Phase 5 NPZ file hash mismatch")
    for name, record in manifest.get("source_provenance", {}).items():
        if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
            continue
        source_path = Path(record["path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"missing Phase 5 source {name}: {source_path}")
        if sha256_file(source_path) != record["sha256"]:
            raise ValueError(f"Phase 5 source hash mismatch: {name}")
    with np.load(npz_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    result = validate_phase5_arrays(arrays, manifest)
    result.update(
        {
            "npz_path": str(npz_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "npz_sha256": actual_npz,
        }
    )
    return result
