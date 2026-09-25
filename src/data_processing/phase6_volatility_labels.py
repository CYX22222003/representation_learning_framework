"""Replayable Phase 6 H=8 future-realised-variance label bundles.

The builder consumes the accepted Phase 5 bounded-fill hourly candles but
applies the stricter Phase 6 target rule: every close in ``(t, t + 8h]`` must
be present, observed, and in the forecast origin's uninterrupted gap segment.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from data_processing.phase5_walks import (
    FEATURE_COLUMNS,
    NATIVE_STEP_NS,
    Phase5WalkSpec,
    _fit_volume_scaler,
    _materialize,
    _with_contract_state,
    normalize_phase5_candles,
    normalize_phase5_metadata,
    sha256_arrays,
    sha256_file,
)
from data_processing.phase6_volatility import (
    FORMULA_VERSION,
    FROZEN_PRIMARY_HORIZON_HOURS,
    _context_records,
    _eligibility,
    future_realised_variance,
)


PHASE6_VOLATILITY_LABEL_SCHEMA_VERSION = 1
COMPARATOR_NAMES = (
    "zero_reference",
    "training_median_reference",
    "historical_persistence_reference",
    "canonical_framework",
    "raw_ohlcv_mlp",
    "raw_lstm",
)
SPLITS = ("train", "test")


@dataclass(frozen=True)
class VolatilityLabelBundle:
    """In-memory arrays and manifest for one Phase 6 walk."""

    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]


def _configuration_hash(configuration: Mapping[str, Any]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prefix_arrays(prefix: str, values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def _target_arrays(
    records_and_positions: list[tuple[dict[str, Any], np.ndarray]],
    *,
    membership: str,
) -> dict[str, np.ndarray]:
    path_rows: list[np.ndarray] = []
    path_dates: list[np.ndarray] = []
    close_paths: list[np.ndarray] = []
    squared_changes: list[np.ndarray] = []
    realised_variance: list[float] = []
    path_observed: list[np.ndarray] = []
    path_imputed: list[np.ndarray] = []
    segment_ids: list[int] = []
    source_starts: list[int] = []
    source_decisions: list[int] = []

    for record, future_positions in records_and_positions:
        contract = record["contract"]
        start = int(record["start"])
        decision = int(record["decision"])
        positions = np.concatenate(
            (np.asarray([decision], dtype=np.int64), np.asarray(future_positions, dtype=np.int64))
        )
        closes = contract["close"].to_numpy(np.float64)[positions]
        target, components = future_realised_variance(closes)
        path_rows.append(positions)
        path_dates.append(
            contract["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)[positions]
        )
        close_paths.append(closes)
        squared_changes.append(components)
        realised_variance.append(target)
        path_observed.append(contract["is_observed"].to_numpy(bool)[positions])
        path_imputed.append(contract["is_imputed"].to_numpy(bool)[positions])
        segment_ids.append(int(contract.at[decision, "segment"]))
        source_starts.append(start)
        source_decisions.append(decision)

    close_path = np.stack(close_paths).astype(np.float64)
    components = np.stack(squared_changes).astype(np.float64)
    target = np.asarray(realised_variance, dtype=np.float64)
    dates = np.stack(path_dates).astype(np.int64)
    count = len(target)
    return {
        "membership": np.asarray([membership] * count, dtype=np.str_),
        "shared_row_indices": np.arange(count, dtype=np.int64),
        "source_contract_row_start": np.asarray(source_starts, dtype=np.int64),
        "source_contract_row_decision": np.asarray(source_decisions, dtype=np.int64),
        "gap_segment_ids": np.asarray(segment_ids, dtype=np.int32),
        "target_path_source_rows": np.stack(path_rows).astype(np.int64),
        "target_path_date_ns": dates,
        "target_start_ns": dates[:, 0],
        "target_first_future_ns": dates[:, 1],
        "target_end_ns": dates[:, -1],
        "target_availability_ns": dates[:, -1] + NATIVE_STEP_NS,
        "target_close_path": close_path,
        "target_squared_changes": components,
        "realised_variance": target,
        "target_path_is_observed": np.stack(path_observed).astype(bool),
        "target_path_is_imputed": np.stack(path_imputed).astype(bool),
        "future_update_count": np.count_nonzero(components > 0.0, axis=1).astype(np.int16),
    }


def build_volatility_label_bundle(
    candles: pd.DataFrame,
    metadata: pd.DataFrame,
    spec: Phase5WalkSpec,
    *,
    supported_contracts: Iterable[str],
    source_provenance: Mapping[str, Any] | None = None,
    expected_stage_a_counts: Mapping[str, int] | None = None,
    seq_len: int = 64,
    horizon: int = FROZEN_PRIMARY_HORIZON_HOURS,
    activity_hours: int = 24,
) -> VolatilityLabelBundle:
    """Build one walk's strictly future H=8 realised-variance bundle."""

    if seq_len != 64 or horizon != 8 or activity_hours != 24:
        raise ValueError("Phase 6 volatility labels are frozen to seq64/h8/activity24h")
    clean = normalize_phase5_candles(candles)
    market_metadata = normalize_phase5_metadata(metadata)
    source_contracts = set(clean["condition_id"].unique())
    if not source_contracts.issubset(set(market_metadata.index)):
        raise ValueError("metadata does not cover every candle contract")
    support = {str(value) for value in supported_contracts}
    if not support or not support.issubset(source_contracts):
        raise ValueError("supported contracts must be a non-empty subset of source contracts")

    records = _context_records(
        clean, market_metadata, spec, seq_len=seq_len, activity_hours=activity_hours
    )
    eligible: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = {
        "train": [],
        "test": [],
    }
    for record in records:
        accepted, _, future_positions = _eligibility(record, horizon, spec, support)
        if not accepted:
            continue
        assert future_positions is not None
        split = "train" if record["split"] == "train" else "test"
        eligible[split].append((record, future_positions))
    if not eligible["train"] or not eligible["test"]:
        raise ValueError("walk must produce non-empty training and evaluation label populations")

    counts = {split: len(eligible[split]) for split in SPLITS}
    if expected_stage_a_counts is not None:
        expected = {split: int(expected_stage_a_counts[split]) for split in SPLITS}
        if counts != expected:
            raise ValueError(f"Stage B rows disagree with frozen Stage A capacity: {counts} != {expected}")

    volume_scaler = _fit_volume_scaler(clean, spec)
    arrays: dict[str, np.ndarray] = {}
    split_values: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        split_records = [record for record, _ in eligible[split]]
        context = _materialize(
            split_records,
            seq_len=seq_len,
            tau=0.001,
            volume_scaler=volume_scaler,
            labels=False,
        )
        targets = _target_arrays(
            eligible[split], membership="training" if split == "train" else "evaluation"
        )
        values = {**context, **targets}
        split_values[split] = values
        arrays.update(_prefix_arrays(split, values))

    identity_fields = (
        "condition_ids",
        "window_start_ns",
        "decision_date_ns",
        "decision_availability_ns",
    )
    identity_hashes = {
        split: sha256_arrays(*(split_values[split][name] for name in identity_fields))
        for split in SPLITS
    }
    target_hashes = {
        split: sha256_arrays(
            split_values[split]["target_path_date_ns"],
            split_values[split]["target_close_path"],
            split_values[split]["target_squared_changes"],
            split_values[split]["realised_variance"],
        )
        for split in SPLITS
    }
    configuration = {
        "phase": 6,
        "stage": "B_supervised_label_bundle",
        "walk": spec.walk,
        "native_resolution": "1h",
        "sequence_length": seq_len,
        "horizon_hours": horizon,
        "activity_hours": activity_hours,
        "formula_version": FORMULA_VERSION,
        "price_change_convention": "raw bounded probability changes",
        "target_interval": "(t,t+8h]",
        "future_candles_required_observed": True,
        "zero_targets_retained": True,
    }
    manifest = {
        "schema_version": PHASE6_VOLATILITY_LABEL_SCHEMA_VERSION,
        **configuration,
        "purpose": "walk_specific_future_realised_variance_labels",
        "formula": "RV(t,8h)=sum_{j=1..8}(p[t+j]-p[t+j-1])^2",
        "configuration": configuration,
        "configuration_sha256": _configuration_hash(configuration),
        "intervals": {
            "training": {
                "start_inclusive": spec.train_start.isoformat(),
                "cutoff_exclusive_for_target_availability": spec.cutoff.isoformat(),
            },
            "evaluation": {
                "start_inclusive_for_decision_availability": spec.cutoff.isoformat(),
                "end_inclusive_for_target_availability": spec.evaluation_end.isoformat(),
            },
        },
        "timestamp_semantics": {
            "candle": "bar_start",
            "availability": "date + 1h",
            "target_start": "forecast-origin candle timestamp t",
            "target_first_future": "t + 1h",
            "target_end": "t + 8h",
            "target_availability": "target_end + 1h",
        },
        "row_population_rules": {
            "observed_decision_endpoint": True,
            "active_24h_at_decision": True,
            "exact_future_hourly_timestamps": True,
            "same_contract_and_gap_segment": True,
            "all_future_target_candles_observed": True,
            "context_isolated_one_hour_imputation_permitted": True,
            "future_activity_used_for_filtering": False,
            "retrospective_end_metadata_used_for_filtering": False,
            "evaluation_contract_support_source": "Phase 5 canonical supervised training contracts",
        },
        "source_row_semantics": "zero-based row positions after canonical condition/date normalization",
        "feature_columns": list(FEATURE_COLUMNS),
        "preprocessing": {"volume": volume_scaler, "ohlc": "unchanged"},
        "row_counts": counts,
        "contract_counts": {
            split: int(np.unique(split_values[split]["condition_ids"]).size) for split in SPLITS
        },
        "zero_target_counts": {
            split: int(np.count_nonzero(split_values[split]["realised_variance"] == 0.0))
            for split in SPLITS
        },
        "context_imputation_exposure_counts": {
            split: int(np.count_nonzero(split_values[split]["context_imputed_rows"] > 0))
            for split in SPLITS
        },
        "identity_hashes": identity_hashes,
        "context_hashes": {
            split: sha256_arrays(
                split_values[split]["raw_sequences"], split_values[split]["sequences"]
            )
            for split in SPLITS
        },
        "target_hashes": target_hashes,
        "shared_row_index_hashes": {
            split: sha256_arrays(split_values[split]["shared_row_indices"])
            for split in SPLITS
        },
        "comparator_alignment": {
            "row_index_field": "shared_row_indices",
            "comparators": list(COMPARATOR_NAMES),
            "all_consume_identical_saved_rows": True,
        },
        "stage_a_eligible_rows": (
            {split: int(expected_stage_a_counts[split]) for split in SPLITS}
            if expected_stage_a_counts is not None
            else None
        ),
        "source_provenance": dict(source_provenance or {}),
    }
    validate_volatility_label_arrays(arrays, manifest)
    return VolatilityLabelBundle(arrays=arrays, manifest=manifest)


def validate_volatility_label_arrays(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay all bundle-internal timing, identity, and target invariants."""

    if int(manifest.get("schema_version", -1)) != PHASE6_VOLATILITY_LABEL_SCHEMA_VERSION:
        raise ValueError("unexpected Phase 6 volatility-label schema version")
    if manifest.get("formula_version") != FORMULA_VERSION:
        raise ValueError("volatility formula version mismatch")
    if int(manifest.get("horizon_hours", -1)) != 8:
        raise ValueError("Phase 6 volatility label horizon must be eight hours")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict) or _configuration_hash(configuration) != manifest.get(
        "configuration_sha256"
    ):
        raise ValueError("configuration hash mismatch")

    seq_len = int(manifest["sequence_length"])
    cutoff = int(pd.Timestamp(manifest["intervals"]["training"]["cutoff_exclusive_for_target_availability"]).value)
    evaluation_end = int(
        pd.Timestamp(manifest["intervals"]["evaluation"]["end_inclusive_for_target_availability"]).value
    )
    identity_fields = (
        "condition_ids",
        "window_start_ns",
        "decision_date_ns",
        "decision_availability_ns",
    )
    for split in SPLITS:
        prefix = f"{split}_"
        count = int(manifest["row_counts"][split])
        required = {
            prefix + name
            for name in (
                "raw_sequences",
                "sequences",
                *identity_fields,
                "context_is_imputed",
                "context_is_observed",
                "context_original_gap_length_bars",
                "context_time_since_observation",
                "context_imputed_rows",
                "activity_change_count_24h",
                "lifecycle_fraction",
                "lifecycle_stage",
                "categories",
                "event_families",
                "selection_ranks",
                "membership",
                "shared_row_indices",
                "source_contract_row_start",
                "source_contract_row_decision",
                "gap_segment_ids",
                "target_path_source_rows",
                "target_path_date_ns",
                "target_start_ns",
                "target_first_future_ns",
                "target_end_ns",
                "target_availability_ns",
                "target_close_path",
                "target_squared_changes",
                "realised_variance",
                "target_path_is_observed",
                "target_path_is_imputed",
                "future_update_count",
            )
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"{split} arrays missing: {missing}")

        raw = np.asarray(arrays[prefix + "raw_sequences"])
        model = np.asarray(arrays[prefix + "sequences"])
        if raw.dtype != np.float32 or raw.shape != (count, seq_len, len(FEATURE_COLUMNS)):
            raise ValueError(f"{split} raw context shape or dtype mismatch")
        if model.dtype != np.float32 or model.shape != raw.shape:
            raise ValueError(f"{split} model-ready context shape or dtype mismatch")
        if not np.isfinite(raw).all() or not np.isfinite(model).all():
            raise ValueError(f"{split} context contains non-finite values")
        if not np.array_equal(raw[:, :, :4], model[:, :, :4]):
            raise ValueError(f"{split} OHLC changed during preprocessing")
        expected_volume = (
            raw[:, :, 4].astype(np.float64)
            - float(manifest["preprocessing"]["volume"]["mean"])
        ) / float(manifest["preprocessing"]["volume"]["denominator"])
        if not np.allclose(model[:, :, 4], expected_volume, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{split} volume scaling does not replay")

        mask = np.asarray(arrays[prefix + "context_is_imputed"], dtype=bool)
        observed = np.asarray(arrays[prefix + "context_is_observed"], dtype=bool)
        if mask.shape != (count, seq_len) or not np.array_equal(observed, ~mask):
            raise ValueError(f"{split} context observation masks are invalid")
        gap_lengths = np.asarray(
            arrays[prefix + "context_original_gap_length_bars"], dtype=np.int16
        )
        time_since = np.asarray(
            arrays[prefix + "context_time_since_observation"], dtype=np.int16
        )
        if gap_lengths.shape != mask.shape or not np.array_equal(
            gap_lengths, mask.astype(np.int16)
        ):
            raise ValueError(f"{split} context gap metadata is invalid")
        if time_since.shape != mask.shape or not np.all(time_since[mask] == 60) or not np.all(
            time_since[~mask] == 0
        ):
            raise ValueError(f"{split} context time-since-observation metadata is invalid")
        if mask[:, -1].any():
            raise ValueError(f"{split} contains an imputed forecast origin")
        if not np.array_equal(mask.sum(axis=1), arrays[prefix + "context_imputed_rows"]):
            raise ValueError(f"{split} context imputation counts do not replay")
        if int(np.count_nonzero(mask.any(axis=1))) != int(
            manifest["context_imputation_exposure_counts"][split]
        ):
            raise ValueError(f"{split} context imputation exposure count mismatch")

        decision = np.asarray(arrays[prefix + "decision_date_ns"], dtype=np.int64)
        decision_availability = np.asarray(
            arrays[prefix + "decision_availability_ns"], dtype=np.int64
        )
        path_dates = np.asarray(arrays[prefix + "target_path_date_ns"], dtype=np.int64)
        expected_dates = decision[:, None] + np.arange(9, dtype=np.int64)[None, :] * NATIVE_STEP_NS
        if path_dates.shape != (count, 9) or not np.array_equal(path_dates, expected_dates):
            raise ValueError(f"{split} target path is not the exact t..t+8h grid")
        if not np.array_equal(decision_availability, decision + NATIVE_STEP_NS):
            raise ValueError(f"{split} decision availability mismatch")
        if not np.array_equal(arrays[prefix + "target_start_ns"], decision):
            raise ValueError(f"{split} target start is not the forecast origin")
        if not np.array_equal(arrays[prefix + "target_first_future_ns"], path_dates[:, 1]):
            raise ValueError(f"{split} first future timestamp mismatch")
        if not np.array_equal(arrays[prefix + "target_end_ns"], path_dates[:, -1]):
            raise ValueError(f"{split} target end mismatch")
        target_availability = np.asarray(
            arrays[prefix + "target_availability_ns"], dtype=np.int64
        )
        if not np.array_equal(target_availability, path_dates[:, -1] + NATIVE_STEP_NS):
            raise ValueError(f"{split} target availability mismatch")
        if np.any(path_dates[:, 1:] <= decision[:, None]):
            raise ValueError(f"{split} target return overlaps the historical input context")

        closes = np.asarray(arrays[prefix + "target_close_path"], dtype=np.float64)
        components = np.asarray(arrays[prefix + "target_squared_changes"], dtype=np.float64)
        target = np.asarray(arrays[prefix + "realised_variance"], dtype=np.float64)
        if closes.shape != (count, 9) or components.shape != (count, 8) or target.shape != (count,):
            raise ValueError(f"{split} target array shape mismatch")
        replay_components = np.square(np.diff(closes, axis=1))
        if not np.allclose(components, replay_components, rtol=1e-12, atol=1e-15):
            raise ValueError(f"{split} component squared changes do not replay")
        if not np.allclose(target, components.sum(axis=1), rtol=1e-12, atol=1e-15):
            raise ValueError(f"{split} realised variance is not the component sum")
        if not np.array_equal(
            arrays[prefix + "future_update_count"],
            np.count_nonzero(components > 0.0, axis=1).astype(np.int16),
        ):
            raise ValueError(f"{split} future update counts do not replay")
        target_observed = np.asarray(arrays[prefix + "target_path_is_observed"], dtype=bool)
        target_imputed = np.asarray(arrays[prefix + "target_path_is_imputed"], dtype=bool)
        if target_observed.shape != (count, 9) or not target_observed.all():
            raise ValueError(f"{split} target path contains a non-observed candle")
        if target_imputed.shape != (count, 9) or target_imputed.any():
            raise ValueError(f"{split} target path contains an imputed candle")

        expected_membership = "training" if split == "train" else "evaluation"
        if not np.all(np.asarray(arrays[prefix + "membership"]).astype(str) == expected_membership):
            raise ValueError(f"{split} membership markers are invalid")
        shared_rows = np.asarray(arrays[prefix + "shared_row_indices"], dtype=np.int64)
        if not np.array_equal(shared_rows, np.arange(count, dtype=np.int64)):
            raise ValueError(f"{split} shared comparator row map is invalid")
        if sha256_arrays(shared_rows) != manifest["shared_row_index_hashes"][split]:
            raise ValueError(f"{split} shared comparator row hash mismatch")
        identity_hash = sha256_arrays(
            *(np.asarray(arrays[prefix + name]) for name in identity_fields)
        )
        if identity_hash != manifest["identity_hashes"][split]:
            raise ValueError(f"{split} identity hash mismatch")
        if sha256_arrays(raw, model) != manifest["context_hashes"][split]:
            raise ValueError(f"{split} context hash mismatch")
        if sha256_arrays(path_dates, closes, components, target) != manifest["target_hashes"][split]:
            raise ValueError(f"{split} target hash mismatch")
        if int(np.count_nonzero(target == 0.0)) != int(manifest["zero_target_counts"][split]):
            raise ValueError(f"{split} zero-target count mismatch")
        if len(np.unique(arrays[prefix + "condition_ids"])) != int(
            manifest["contract_counts"][split]
        ):
            raise ValueError(f"{split} contract count mismatch")
        if np.any(np.asarray(arrays[prefix + "activity_change_count_24h"]) <= 0):
            raise ValueError(f"{split} includes an inactive forecast origin")
        if split == "train" and np.any(target_availability >= cutoff):
            raise ValueError("training target matures at or after the cutoff")
        if split == "test":
            if np.any(decision_availability < cutoff):
                raise ValueError("evaluation decision occurs before the cutoff")
            if np.any(target_availability > evaluation_end):
                raise ValueError("evaluation target matures after the evaluation end")

    train_identity = set(
        zip(
            arrays["train_condition_ids"].astype(str).tolist(),
            arrays["train_decision_date_ns"].tolist(),
            strict=True,
        )
    )
    test_identity = set(
        zip(
            arrays["test_condition_ids"].astype(str).tolist(),
            arrays["test_decision_date_ns"].tolist(),
            strict=True,
        )
    )
    if train_identity & test_identity:
        raise ValueError("training and evaluation identities overlap")
    stage_a = manifest.get("stage_a_eligible_rows")
    if stage_a is not None and {split: int(stage_a[split]) for split in SPLITS} != {
        split: int(manifest["row_counts"][split]) for split in SPLITS
    }:
        raise ValueError("saved row counts do not replay frozen Stage A capacity")
    if not manifest.get("comparator_alignment", {}).get("all_consume_identical_saved_rows"):
        raise ValueError("comparator row alignment was not declared")
    return {
        "valid": True,
        "walk": int(manifest["walk"]),
        "row_counts": dict(manifest["row_counts"]),
        "identity_hashes": dict(manifest["identity_hashes"]),
        "target_hashes": dict(manifest["target_hashes"]),
        "comparators": list(manifest["comparator_alignment"]["comparators"]),
    }


def _validate_against_source(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any], candles_path: Path
) -> None:
    clean = normalize_phase5_candles(pd.read_parquet(candles_path))
    contracts = {
        str(condition_id): _with_contract_state(contract.reset_index(drop=True), 24)
        for condition_id, contract in clean.groupby("condition_id", sort=True)
    }
    offsets = np.arange(int(manifest["sequence_length"]), dtype=np.int64)
    for split in SPLITS:
        prefix = f"{split}_"
        condition_ids = np.asarray(arrays[prefix + "condition_ids"]).astype(str)
        for condition_id in np.unique(condition_ids):
            if condition_id not in contracts:
                raise ValueError(f"source replay is missing contract {condition_id}")
            row_indices = np.flatnonzero(condition_ids == condition_id)
            contract = contracts[condition_id]
            starts = np.asarray(arrays[prefix + "source_contract_row_start"], dtype=np.int64)[
                row_indices
            ]
            decisions = np.asarray(
                arrays[prefix + "source_contract_row_decision"], dtype=np.int64
            )[row_indices]
            path_rows = np.asarray(
                arrays[prefix + "target_path_source_rows"], dtype=np.int64
            )[row_indices]
            if np.any(starts < 0) or np.any(path_rows < 0) or np.any(path_rows >= len(contract)):
                raise ValueError(f"{split} source-row map is outside contract {condition_id}")
            if not np.array_equal(decisions, starts + int(manifest["sequence_length"]) - 1):
                raise ValueError(f"{split} context source-row map is not contiguous")
            if not np.array_equal(path_rows[:, 0], decisions):
                raise ValueError(f"{split} target source-row map does not begin at the decision")
            context_rows = starts[:, None] + offsets[None, :]
            expected_raw = contract.loc[:, FEATURE_COLUMNS].to_numpy(np.float64)[context_rows].astype(
                np.float32
            )
            if not np.array_equal(expected_raw, arrays[prefix + "raw_sequences"][row_indices]):
                raise ValueError(f"{split} raw contexts do not replay from source {condition_id}")
            source_dates = contract["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            source_closes = contract["close"].to_numpy(np.float64)
            if not np.array_equal(source_dates[path_rows], arrays[prefix + "target_path_date_ns"][row_indices]):
                raise ValueError(f"{split} target dates do not replay from source {condition_id}")
            if not np.array_equal(source_closes[path_rows], arrays[prefix + "target_close_path"][row_indices]):
                raise ValueError(f"{split} target closes do not replay from source {condition_id}")
            observed = contract["is_observed"].to_numpy(bool)[path_rows]
            imputed = contract["is_imputed"].to_numpy(bool)[path_rows]
            segments = contract["segment"].to_numpy(np.int32)[path_rows]
            stored_segments = np.asarray(arrays[prefix + "gap_segment_ids"], dtype=np.int32)[
                row_indices
            ]
            if not observed.all() or imputed.any():
                raise ValueError(f"{split} source target path is not fully observed")
            if not np.all(segments == stored_segments[:, None]):
                raise ValueError(f"{split} target crosses a source gap segment")


def write_volatility_label_bundle(
    npz_path: Path,
    bundle: VolatilityLabelBundle,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Atomically write one NPZ bundle and its hash-bearing JSON manifest."""

    npz_path = Path(npz_path)
    manifest_path = Path(f"{npz_path}.manifest.json")
    occupied = [path for path in (npz_path, manifest_path) if path.exists()]
    if occupied and not overwrite:
        raise FileExistsError(f"refusing to overwrite Phase 6 label artifacts: {occupied}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(handle, **bundle.arrays)
    os.replace(temporary_npz, npz_path)
    manifest = {
        **bundle.manifest,
        "artifact": {
            "path": str(npz_path.resolve()),
            "sha256": sha256_file(npz_path),
            "bytes": npz_path.stat().st_size,
        },
    }
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return npz_path, manifest_path


def validate_volatility_label_bundle_files(
    npz_path: Path, manifest_path: Path | None = None, *, replay_source: bool = True
) -> dict[str, Any]:
    """Validate artifact/source hashes, arrays, and exact source-row replay."""

    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path or f"{npz_path}.manifest.json")
    if not npz_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Phase 6 volatility NPZ and companion manifest are both required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(npz_path) != manifest.get("artifact", {}).get("sha256"):
        raise ValueError("Phase 6 volatility NPZ hash mismatch")
    for name, record in manifest.get("source_provenance", {}).items():
        if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
            continue
        source_path = Path(record["path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"missing Phase 6 source {name}: {source_path}")
        if sha256_file(source_path) != record["sha256"]:
            raise ValueError(f"Phase 6 source hash mismatch: {name}")
    with np.load(npz_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    result = validate_volatility_label_arrays(arrays, manifest)
    if replay_source:
        candles_record = manifest.get("source_provenance", {}).get(
            "candles_1h_clean_ffill1", {}
        )
        if "path" not in candles_record:
            raise ValueError("source replay requires candles_1h_clean_ffill1 provenance")
        _validate_against_source(arrays, manifest, Path(candles_record["path"]))
        result["all_rows_replayed_from_source"] = True
    result.update(
        {
            "npz_path": str(npz_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "npz_sha256": sha256_file(npz_path),
        }
    )
    return result
