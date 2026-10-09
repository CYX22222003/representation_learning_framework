"""Observed-path intersection of immutable Phase 5 h8 price identities.

Never fits a scaler or repairs candles. Original row indices are retained for
the mandatory H0-D0/Raw-LSTM intersection reruns. Validation independently
reconstructs every saved context and future path from the accepted source.
"""
from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data_processing.phase5_walks import (
    FEATURE_COLUMNS, NATIVE_STEP_NS, _with_contract_state,
    normalize_phase5_candles, sha256_arrays, sha256_file,
    validate_phase5_bundle_files,
)
from training.phase5_encoder import write_json
from training.phase6_9_xlstm_mixer import (
    DATA_SCHEMA_VERSION, IDENTITY_FIELDS, METHOD, phase_data_path,
    validate_xlstm_mixer_arrays,
)


def build_future_paths(
    candles: pd.DataFrame,
    original: Mapping[str, np.ndarray],
    original_manifest: Mapping[str, Any],
    *,
    enforce_frozen_counts: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Join all eight observed bars, preserving original split and row order."""
    clean = normalize_phase5_candles(candles)
    contracts = {
        str(key): _with_contract_state(value.reset_index(drop=True), 24)
        for key, value in clean.groupby("condition_id", sort=False)
    }
    volume = dict(original_manifest["preprocessing"]["volume"])
    if volume.get("uses_evaluation_rows") is not False or float(volume["denominator"]) <= 0:
        raise ValueError("the upstream volume transform must be train-only and valid")
    volume["sha256"] = sha256(json.dumps(volume, sort_keys=True).encode()).hexdigest()
    arrays: dict[str, np.ndarray] = {}
    identities, counts, original_counts = {}, {}, {}
    for split, prefix in (("train", "train"), ("evaluation", "test")):
        conditions = np.asarray(original[f"{prefix}_condition_ids"])
        decision = np.asarray(original[f"{prefix}_decision_date_ns"])
        contexts = np.asarray(original[f"{prefix}_sequences"])
        n = len(conditions)
        keep = np.zeros(n, bool)
        targets = np.zeros((n, 8, 5), np.float32)
        segments = np.zeros(n, np.int32)
        for condition in np.unique(conditions):
            rows = np.flatnonzero(conditions == condition)
            contract = contracts[str(condition)]
            dates = contract["date"].astype("int64").to_numpy()
            pos = np.searchsorted(dates, decision[rows])
            if np.any(pos >= len(dates)) or not np.array_equal(dates[pos], decision[rows]):
                raise ValueError("original decision identity is absent from the source")
            context_pos = pos[:, None] + np.arange(-63, 1)[None, :]
            if np.any(context_pos < 0):
                raise ValueError("original context crosses a contract start")
            raw = contract.loc[:, FEATURE_COLUMNS].to_numpy(np.float64)
            scaled = raw.copy()
            scaled[:, 4] = (scaled[:, 4] - volume["mean"]) / volume["denominator"]
            segment = contract["segment"].to_numpy(np.int32)
            if not np.all(segment[context_pos] == segment[pos, None]):
                raise ValueError("original context crosses a sequence break")
            if not np.array_equal(scaled[context_pos].astype(np.float32), contexts[rows]):
                raise ValueError("original scaled contexts disagree with source replay")
            if not np.array_equal(raw[context_pos].astype(np.float32), original[f"{prefix}_raw_sequences"][rows]):
                raise ValueError("original raw contexts disagree with source replay")
            observed = contract["is_observed"].to_numpy(bool)
            if not np.all(observed[pos]) or not np.all(contract["active"].to_numpy(bool)[pos]):
                raise ValueError("original decision endpoint is unobserved or inactive")
            if not np.array_equal(~observed[context_pos], original[f"{prefix}_context_is_imputed"][rows]):
                raise ValueError("original context imputation metadata differs from source")
            future_pos = pos[:, None] + np.arange(1, 9)[None, :]
            safe = np.minimum(future_pos, len(dates) - 1)
            eligible = (
                (future_pos[:, -1] < len(dates))
                & np.all(dates[safe] == decision[rows, None] + NATIVE_STEP_NS * np.arange(1, 9), axis=1)
                & np.all(segment[safe] == segment[pos, None], axis=1)
                & np.all(observed[safe], axis=1)
            )
            keep[rows] = eligible
            targets[rows] = scaled[safe].astype(np.float32)
            segments[rows] = segment[pos]
        selected = np.flatnonzero(keep)
        if not len(selected):
            raise ValueError(f"{split} has no fully observed future paths")
        time = decision[selected, None] + NATIVE_STEP_NS * np.arange(1, 9)
        # This retains the broader h8 endpoint identity for joining comparators.
        row_ids = np.asarray([
            f"{conditions[i]}:{int(original[f'{prefix}_window_start_ns'][i])}:"
            f"{int(decision[i])}:{int(original[f'{prefix}_decision_availability_ns'][i])}"
            for i in selected
        ])
        metadata = {
            "row_ids": row_ids,
            "condition_ids": conditions[selected],
            "segment_ids": segments[selected],
            "decision_time_ns": decision[selected],
            "decision_availability_ns": np.asarray(original[f"{prefix}_decision_availability_ns"])[selected],
            "target_time_ns": time,
            "target_availability_ns": time + NATIVE_STEP_NS,
            "target_observed": np.ones((len(selected), 8), bool),
            "target_imputed": np.zeros((len(selected), 8), bool),
            # Float32 tensors are optimization inputs, not a replacement for
            # the original float64 endpoint labels used by comparator metrics.
            "current_close": np.asarray(original[f"{prefix}_current_close"])[selected],
            "target_close": np.asarray(original[f"{prefix}_target_close"])[selected],
            "original_row_indices": selected.astype(np.int64),
        }
        if not np.array_equal(time[:, -1], original[f"{prefix}_target_date_ns"][selected]):
            raise ValueError("h8 target timestamps differ from original price identities")
        if not np.array_equal(targets[selected, -1, 3], metadata["target_close"].astype(np.float32)):
            raise ValueError("h8 endpoint labels differ from source")
        arrays[f"{split}_contexts"] = contexts[selected]
        arrays[f"{split}_targets"] = targets[selected]
        for name, values in metadata.items():
            arrays[f"{split}_{name}"] = values
        for name in ("context_is_imputed", "context_is_observed", "context_original_gap_length_bars",
                     "context_time_since_observation", "activity_change_count_24h", "lifecycle_stage",
                     "categories", "event_families"):
            arrays[f"{split}_{name}"] = np.asarray(original[f"{prefix}_{name}"])[selected]
        identities[split] = sha256_arrays(*(metadata[name] for name in IDENTITY_FIELDS))
        counts[split], original_counts[split] = len(selected), n
    intervals = original_manifest["intervals"]
    cutoff = int(pd.Timestamp(intervals["training"]["cutoff_exclusive"]).value)
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION, "method": METHOD, "method_id": "XM-MV8",
        "walk": int(original_manifest["walk"]), "validation_passed": True,
        "channel_order": list(FEATURE_COLUMNS), "additional_xlstm_channel_scaler": False,
        "volume_transform": volume, "identity_hashes": identities, "row_counts": counts,
        "original_row_counts": original_counts, "training_cutoff_ns": cutoff,
        "evaluation_start_ns": cutoff,
        "evaluation_end_ns": int(pd.Timestamp(intervals["evaluation"]["end_exclusive"]).value),
        "full_path_policy": {"horizon_hours": list(range(1, 9)), "same_condition": True,
            "same_segment": True, "all_target_bars_observed": True,
            "target_imputation_allowed": False, "availability_only_intersection": True},
        "matched_comparator_reruns_required": ["H0-D0", "Raw LSTM"],
    }
    validate_xlstm_mixer_arrays(arrays, manifest, enforce_frozen_counts=enforce_frozen_counts)
    return arrays, manifest


def _load_sources(root: Path, walk: int) -> tuple[pd.DataFrame, dict, dict, dict]:
    original_path = root / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8" / "data" / f"walk{walk}" / "market_1h_seq64_h8.npz"
    original_manifest_path = Path(f"{original_path}.manifest.json")
    validate_phase5_bundle_files(original_path)
    original_manifest = json.loads(original_manifest_path.read_text())
    if original_manifest["walk"] != walk or original_manifest["horizon_hours"] != 8:
        raise ValueError("original price bundle walk/horizon mismatch")
    provenance = {
        "original_price_bundle": {"path": str(original_path.resolve()), "sha256": sha256_file(original_path)},
        "original_price_manifest": {"path": str(original_manifest_path.resolve()), "sha256": sha256_file(original_manifest_path)},
        **original_manifest["source_provenance"],
    }
    for source in provenance.values():
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("accepted source hash changed")
    with np.load(original_path, allow_pickle=False) as stored:
        original = {name: stored[name] for name in stored.files if name.startswith(("train_", "test_"))}
    candles = pd.read_parquet(provenance["candles_1h_clean_ffill1"]["path"])
    return candles, original, original_manifest, provenance


def prepare_future_path_data(root: Path, walk: int) -> dict[str, Any]:
    path = phase_data_path(root, walk)
    if path.exists() or Path(f"{path}.manifest.json").exists():
        return validate_future_path_data(root, walk)
    candles, original, original_manifest, provenance = _load_sources(root, walk)
    arrays, manifest = build_future_paths(candles, original, original_manifest)
    manifest["source_provenance"] = provenance
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    manifest["dataset_sha256"] = sha256_file(path)
    write_json(Path(f"{path}.manifest.json"), manifest)
    return validate_future_path_data(root, walk)


def validate_future_path_data(root: Path, walk: int) -> dict[str, Any]:
    path = phase_data_path(root, walk)
    manifest = json.loads(Path(f"{path}.manifest.json").read_text())
    if sha256_file(path) != manifest.get("dataset_sha256"):
        raise ValueError("future-path artifact hash mismatch")
    candles, original, original_manifest, provenance = _load_sources(root, walk)
    expected, expected_manifest = build_future_paths(candles, original, original_manifest)
    # WSL's Windows mount accepts either casing; bash resolves this workspace
    # using its on-disk spelling whereas Python CLI invocations may not.
    def stable_sources(records):
        return {name: {**record, "path": str(record["path"]).casefold()}
                for name, record in records.items()}
    if stable_sources(manifest.get("source_provenance", {})) != stable_sources(provenance):
        raise ValueError("future-path source provenance mismatch")
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError("future-path manifest replay mismatch")
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(expected):
            raise ValueError("future-path array inventory mismatch")
        for name, values in expected.items():
            if not np.array_equal(stored[name], values):
                raise ValueError(f"future-path source replay mismatch: {name}")
    return {"valid": True, "walk": walk, "path": str(path),
            "row_counts": manifest["row_counts"], "identity_hashes": manifest["identity_hashes"],
            "source_replay": True, "matched_controls_trained": False}
