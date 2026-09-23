"""Training-only Phase 6 future-realised-variance horizon audit.

This module deliberately stops before label-bundle construction or model
training.  It reuses the accepted Phase 5 context/activity contract, measures
evaluation-period *capacity* only, and never emits evaluation target values.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr

from data_processing.phase5_walks import (
    NATIVE_STEP,
    Phase5WalkSpec,
    _lifecycle_values,
    _with_contract_state,
    normalize_phase5_candles,
    normalize_phase5_metadata,
    sha256_file,
)


PHASE6_VOLATILITY_AUDIT_SCHEMA_VERSION = 1
PHASE6_VOLATILITY_FREEZE_SCHEMA_VERSION = 1
FORMULA_VERSION = "phase6-future-realised-variance-raw-probability-v1"
DEFAULT_CANDIDATE_HORIZONS = (2, 4, 8, 24)
FROZEN_PRIMARY_HORIZON_HOURS = 8
CAPACITY_FILENAME = "horizon_capacity.csv"
TARGET_DISTRIBUTION_FILENAME = "target_distribution.csv"
TARGET_DEPENDENCE_FILENAME = "target_dependence.csv"
TARGET_STRATA_FILENAME = "target_strata.csv"
MANIFEST_FILENAME = "manifest.json"
REPORT_FILENAME = "report.md"


@dataclass(frozen=True)
class VolatilityWalkAudit:
    """In-memory outputs for one walk's horizon audit."""

    capacity: pd.DataFrame
    distribution: pd.DataFrame
    dependence: pd.DataFrame
    strata: pd.DataFrame
    manifest: dict[str, Any]


def parse_candidate_horizons(value: str | Iterable[int]) -> tuple[int, ...]:
    """Parse, sort, and validate the predeclared hourly horizon set."""

    if isinstance(value, str):
        candidates = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        candidates = [int(item) for item in value]
    horizons = tuple(sorted(set(candidates)))
    if not horizons or any(horizon <= 1 for horizon in horizons):
        raise ValueError("every Phase 6 volatility candidate horizon must exceed one hour")
    if max(horizons) >= 64:
        raise ValueError("candidate horizons must be shorter than the fixed 64-hour context")
    return horizons


def future_realised_variance(closes: np.ndarray) -> tuple[float, np.ndarray]:
    """Return sum of squared raw probability changes and its increments."""

    values = np.asarray(closes, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("realised-variance closes must be a finite one-dimensional array")
    squared = np.square(np.diff(values))
    return float(squared.sum(dtype=np.float64)), squared


def _safe_corr(x: Sequence[float], y: Sequence[float], *, rank: bool = False) -> float:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if (
        len(left) < 2
        or np.allclose(left, left[0], rtol=1e-12, atol=1e-15)
        or np.allclose(right, right[0], rtol=1e-12, atol=1e-15)
    ):
        return float("nan")
    if rank:
        return float(spearmanr(left, right).statistic)
    return float(np.corrcoef(left, right)[0, 1])


def _price_band(value: float) -> str:
    if value < 0.05:
        return "[0,0.05)"
    if value < 0.25:
        return "[0.05,0.25)"
    if value < 0.75:
        return "[0.25,0.75)"
    if value < 0.95:
        return "[0.75,0.95)"
    return "[0.95,1]"


def _activity_band(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 8:
        return "4-8"
    return "9+"


def _update_band(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    return "4+"


def _time_to_resolution_band(hours: float) -> str:
    if not np.isfinite(hours):
        return "unknown"
    if hours < 0:
        return "past_metadata_end"
    if hours <= 24:
        return "0-24h"
    if hours <= 24 * 7:
        return "1-7d"
    if hours <= 24 * 30:
        return "7-30d"
    return "30d+"


def _lifecycle_stage_label(value: int) -> str:
    return {0: "early", 1: "middle", 2: "late"}.get(int(value), "unknown")


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {
            "rows": 0,
            "zero_rate": float("nan"),
            "positive_rate": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "q01": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "q50": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "maximum": float("nan"),
            "skew": float("nan"),
        }
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "rows": int(len(values)),
        "zero_rate": float(np.mean(values == 0.0)),
        "positive_rate": float(np.mean(values > 0.0)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q50": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "maximum": float(values.max()),
        "skew": (
            float(skew(values, bias=False))
            if len(values) >= 3
            and not np.allclose(values, values[0], rtol=1e-12, atol=1e-15)
            else float("nan")
        ),
    }


def _context_records(
    clean: pd.DataFrame,
    metadata: pd.DataFrame,
    spec: Phase5WalkSpec,
    *,
    seq_len: int,
    activity_hours: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for condition_id, raw_contract in clean.groupby("condition_id", sort=True):
        contract = _with_contract_state(raw_contract.reset_index(drop=True), activity_hours)
        if len(contract) < seq_len:
            continue
        segments = contract["segment"].to_numpy(np.int32)
        observed = contract["is_observed"].to_numpy(bool)
        active = contract["active"].to_numpy(bool)
        for decision in range(seq_len - 1, len(contract)):
            start = decision - seq_len + 1
            if segments[start] != segments[decision] or not observed[decision] or not active[decision]:
                continue
            window_start = contract.at[start, "date"]
            if window_start < spec.train_start:
                continue
            availability = contract.at[decision, "availability"]
            if availability < spec.cutoff:
                split = "train"
            elif availability < spec.evaluation_end:
                split = "evaluation"
            else:
                continue
            records.append(
                {
                    "condition_id": str(condition_id),
                    "contract": contract,
                    "metadata": metadata.loc[condition_id],
                    "start": start,
                    "decision": decision,
                    "split": split,
                }
            )
    return records


def _eligibility(
    record: Mapping[str, Any],
    horizon: int,
    spec: Phase5WalkSpec,
    supported_contracts: set[str],
) -> tuple[bool, dict[str, bool], np.ndarray | None]:
    """Check target eligibility without calculating an evaluation target value."""

    contract = record["contract"]
    decision = int(record["decision"])
    split = str(record["split"])
    decision_date = contract.at[decision, "date"]
    target_date = decision_date + horizon * NATIVE_STEP
    target_availability = target_date + NATIVE_STEP
    reasons = {
        "unsupported_contract": split == "evaluation"
        and str(record["condition_id"]) not in supported_contracts,
        "cutoff_maturity": split == "train" and target_availability >= spec.cutoff,
        "evaluation_end_maturity": split == "evaluation"
        and target_availability > spec.evaluation_end,
        "source_boundary": target_date > contract["date"].iloc[-1],
        "contract_boundary": target_date > contract["date"].iloc[-1],
        "missing_intermediate_candle": False,
        "gap_segment_break": False,
        "imputed_target_candle": False,
    }
    # Do not inspect future rows for an evaluation contract that is already
    # outside the training-supported universe.
    if reasons["unsupported_contract"]:
        return False, reasons, None
    if reasons["cutoff_maturity"] or reasons["evaluation_end_maturity"]:
        return False, reasons, None

    dates = contract["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    expected = np.asarray(
        [int((decision_date + step * NATIVE_STEP).value) for step in range(1, horizon + 1)],
        dtype=np.int64,
    )
    positions = np.searchsorted(dates, expected)
    present = (positions < len(dates)) & (dates[np.minimum(positions, len(dates) - 1)] == expected)
    if not present.all():
        reasons["missing_intermediate_candle"] = True
        reasons["gap_segment_break"] = True
        return False, reasons, None
    future_positions = positions.astype(np.int64)
    segments = contract["segment"].to_numpy(np.int32)
    if np.any(segments[future_positions] != segments[decision]):
        reasons["gap_segment_break"] = True
    if contract["is_imputed"].to_numpy(bool)[future_positions].any():
        reasons["imputed_target_candle"] = True
    eligible = not any(reasons.values())
    return eligible, reasons, future_positions if eligible else None


def _capacity_row(
    frame: pd.DataFrame,
    *,
    walk: int,
    horizon: int,
    split: str,
    scope: str,
    condition_id: str,
) -> dict[str, Any]:
    reason_columns = [column for column in frame if column.startswith("drop_")]
    eligible = frame["eligible"].to_numpy(bool) if len(frame) else np.empty(0, dtype=bool)
    return {
        "walk": walk,
        "horizon_hours": horizon,
        "split": split,
        "scope": scope,
        "condition_id": condition_id,
        "candidate_rows": int(len(frame)),
        "eligible_rows": int(eligible.sum()),
        "eligible_contracts": int(frame.loc[eligible, "condition_id"].nunique()) if len(frame) else 0,
        **{column: int(frame[column].sum()) for column in reason_columns},
    }


def _raw_change_dependence(
    clean: pd.DataFrame, spec: Phase5WalkSpec, activity_hours: int
) -> tuple[float, float, int]:
    raw_pairs: list[tuple[float, float]] = []
    squared_pairs: list[tuple[float, float]] = []
    for _, raw_contract in clean.groupby("condition_id", sort=False):
        contract = _with_contract_state(raw_contract.reset_index(drop=True), activity_hours)
        allowed = contract["date"].ge(spec.train_start) & contract["availability"].lt(spec.cutoff)
        for _, segment in contract.loc[allowed].groupby("segment", sort=False):
            close = segment["close"].to_numpy(np.float64)
            observed = segment["is_observed"].to_numpy(bool)
            if len(close) < 3:
                continue
            changes = np.diff(close)
            valid_change = observed[:-1] & observed[1:]
            valid_pair = valid_change[:-1] & valid_change[1:]
            raw_pairs.extend(zip(changes[:-1][valid_pair], changes[1:][valid_pair], strict=True))
            squared = np.square(changes)
            squared_pairs.extend(zip(squared[:-1][valid_pair], squared[1:][valid_pair], strict=True))
    raw_left = [pair[0] for pair in raw_pairs]
    raw_right = [pair[1] for pair in raw_pairs]
    sq_left = [pair[0] for pair in squared_pairs]
    sq_right = [pair[1] for pair in squared_pairs]
    return _safe_corr(raw_left, raw_right), _safe_corr(sq_left, sq_right), len(raw_pairs)


def audit_volatility_walk(
    candles: pd.DataFrame,
    metadata: pd.DataFrame,
    spec: Phase5WalkSpec,
    *,
    supported_contracts: Iterable[str],
    expected_encoder_identities: set[tuple[str, int, int, int]] | None = None,
    candidate_horizons: Iterable[int] = DEFAULT_CANDIDATE_HORIZONS,
    seq_len: int = 64,
    activity_hours: int = 24,
) -> VolatilityWalkAudit:
    """Audit one walk using training targets and evaluation capacity only."""

    horizons = parse_candidate_horizons(candidate_horizons)
    if seq_len != 64 or activity_hours != 24:
        raise ValueError("Phase 6 inherits the fixed Phase 5 seq64/activity24h contract")
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
    if not records or not any(record["split"] == "train" for record in records):
        raise ValueError("walk produced no context-eligible training candidates")
    identity_replayed = False
    if expected_encoder_identities is not None:
        actual_identities = {
            (
                str(record["condition_id"]),
                int(record["contract"].at[int(record["start"]), "date"].value),
                int(record["contract"].at[int(record["decision"]), "date"].value),
                int(record["contract"].at[int(record["decision"]), "availability"].value),
            )
            for record in records
            if record["split"] == "train"
        }
        if actual_identities != expected_encoder_identities:
            missing = len(expected_encoder_identities - actual_identities)
            extra = len(actual_identities - expected_encoder_identities)
            raise ValueError(
                f"Phase 5 encoder identity replay failed: missing={missing}, extra={extra}"
            )
        identity_replayed = True
    raw_acf, squared_acf, raw_pair_count = _raw_change_dependence(clean, spec, activity_hours)

    capacity_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    dependence_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    walk_counts: dict[str, Any] = {}

    for horizon in horizons:
        eligibility_rows: list[dict[str, Any]] = []
        training_targets: list[dict[str, Any]] = []
        for record in records:
            eligible, reasons, future_positions = _eligibility(record, horizon, spec, support)
            row = {
                "condition_id": record["condition_id"],
                "split": record["split"],
                "eligible": eligible,
                **{f"drop_{name}": value for name, value in reasons.items()},
            }
            eligibility_rows.append(row)
            if record["split"] != "train" or not eligible:
                continue
            assert future_positions is not None
            contract = record["contract"]
            decision = int(record["decision"])
            start = int(record["start"])
            close_positions = np.concatenate(([decision], future_positions))
            closes = contract["close"].to_numpy(np.float64)[close_positions]
            target, squared_increments = future_realised_variance(closes)
            historical_closes = contract.loc[decision - horizon : decision, "close"].to_numpy(
                np.float64
            )
            historical, _ = future_realised_variance(historical_closes)
            decision_date = contract.at[decision, "date"]
            lifecycle_fraction, lifecycle_stage = _lifecycle_values(
                np.asarray([int(decision_date.value)], dtype=np.int64), record["metadata"]
            )
            end_date = record["metadata"].get("end_date_search")
            time_to_resolution = (
                (pd.Timestamp(end_date) - decision_date).total_seconds() / 3600.0
                if not pd.isna(end_date)
                else float("nan")
            )
            context_imputed_rows = int(
                contract.loc[start:decision, "is_imputed"].to_numpy(bool).sum()
            )
            training_targets.append(
                {
                    "condition_id": record["condition_id"],
                    "decision_date": decision_date,
                    "target": target,
                    "historical_rv": historical,
                    "largest_squared_increment": float(squared_increments.max()),
                    "future_update_count": int(np.count_nonzero(squared_increments > 0.0)),
                    "starting_price": float(contract.at[decision, "close"]),
                    "activity_change_count": int(contract.at[decision, "activity_change_count"]),
                    "lifecycle_fraction": float(lifecycle_fraction[0]),
                    "lifecycle_stage": int(lifecycle_stage[0]),
                    "time_to_resolution_hours": float(time_to_resolution),
                    "context_imputed_rows": context_imputed_rows,
                }
            )

        eligibility = pd.DataFrame(eligibility_rows)
        for split in ("train", "evaluation"):
            split_frame = eligibility.loc[eligibility["split"].eq(split)].copy()
            capacity_rows.append(
                _capacity_row(
                    split_frame,
                    walk=spec.walk,
                    horizon=horizon,
                    split=split,
                    scope="aggregate",
                    condition_id="__all__",
                )
            )
            for condition_id in sorted(source_contracts):
                contract_frame = split_frame.loc[split_frame["condition_id"].eq(condition_id)]
                capacity_rows.append(
                    _capacity_row(
                        contract_frame,
                        walk=spec.walk,
                        horizon=horizon,
                        split=split,
                        scope="contract",
                        condition_id=condition_id,
                    )
                )

        targets = pd.DataFrame(training_targets)
        if targets.empty:
            raise ValueError(f"walk {spec.walk} horizon {horizon} has no eligible training targets")
        values = targets["target"].to_numpy(np.float64)
        total = float(values.sum())
        largest_target_share = float(values.max() / total) if total > 0 else float("nan")
        largest_increment = float(targets["largest_squared_increment"].max())
        largest_increment_share = largest_increment / total if total > 0 else float("nan")
        distribution_rows.append(
            {
                "walk": spec.walk,
                "horizon_hours": horizon,
                "split": "train",
                **_summary(values),
                "contracts": int(targets["condition_id"].nunique()),
                "aggregate_rv": total,
                "largest_interval_share": largest_target_share,
                "largest_squared_increment_share": largest_increment_share,
                "mean_future_update_count": float(targets["future_update_count"].mean()),
                "context_imputation_exposure_rate": float(
                    np.mean(targets["context_imputed_rows"].to_numpy() > 0)
                ),
            }
        )

        lag_left: list[float] = []
        lag_right: list[float] = []
        for _, contract_targets in targets.groupby("condition_id", sort=False):
            contract_targets = contract_targets.sort_values("decision_date")
            dates = contract_targets["decision_date"].to_numpy(dtype="datetime64[ns]")
            adjacent = np.diff(dates).astype("timedelta64[h]").astype(np.int64) == 1
            rv = contract_targets["target"].to_numpy(np.float64)
            lag_left.extend(rv[:-1][adjacent])
            lag_right.extend(rv[1:][adjacent])
        dependence_rows.append(
            {
                "walk": spec.walk,
                "horizon_hours": horizon,
                "split": "train",
                "raw_change_lag1_pearson": raw_acf,
                "squared_change_lag1_pearson": squared_acf,
                "raw_change_pair_count": raw_pair_count,
                "rv_lag1_pearson": _safe_corr(lag_left, lag_right),
                "rv_lag1_pair_count": len(lag_left),
                "historical_future_pearson": _safe_corr(
                    targets["historical_rv"], targets["target"]
                ),
                "historical_future_spearman": _safe_corr(
                    targets["historical_rv"], targets["target"], rank=True
                ),
            }
        )

        stratum_values = {
            "starting_price_band": targets["starting_price"].map(_price_band),
            "causal_activity_band": targets["activity_change_count"].map(_activity_band),
            "lifecycle_stage": targets["lifecycle_stage"].map(_lifecycle_stage_label),
            "time_to_resolution_band": targets["time_to_resolution_hours"].map(
                _time_to_resolution_band
            ),
            "future_update_count_band_retrospective": targets["future_update_count"].map(
                _update_band
            ),
            "context_imputation_exposure": np.where(
                targets["context_imputed_rows"].to_numpy() > 0, "any", "none"
            ),
        }
        for stratum, labels in stratum_values.items():
            labelled = targets.assign(_stratum_value=np.asarray(labels, dtype=str))
            for label, group in labelled.groupby("_stratum_value", sort=True):
                strata_rows.append(
                    {
                        "walk": spec.walk,
                        "horizon_hours": horizon,
                        "split": "train",
                        "stratum": stratum,
                        "value": label,
                        "contracts": int(group["condition_id"].nunique()),
                        **_summary(group["target"].to_numpy(np.float64)),
                    }
                )
        aggregate_capacity = capacity_rows[-(2 * (len(source_contracts) + 1)) :]
        walk_counts[str(horizon)] = {
            row["split"]: {
                "candidate_rows": row["candidate_rows"],
                "eligible_rows": row["eligible_rows"],
                "eligible_contracts": row["eligible_contracts"],
            }
            for row in aggregate_capacity
            if row["scope"] == "aggregate"
        }

    manifest = {
        "schema_version": PHASE6_VOLATILITY_AUDIT_SCHEMA_VERSION,
        "phase": 6,
        "stage": "A_read_only_horizon_audit",
        "walk": spec.walk,
        "candidate_horizons_hours": list(horizons),
        "primary_horizon_frozen": False,
        "formula_version": FORMULA_VERSION,
        "formula": "RV(t,H)=sum_{j=1..H}(p[t+j]-p[t+j-1])^2",
        "price_change_convention": "raw bounded probability changes",
        "timestamp_semantics": {
            "candle": "bar_start",
            "availability": "date + 1h",
            "training_target": "availability strictly before cutoff",
            "evaluation_target": "availability no later than evaluation end",
        },
        "intervals": {
            "training": {
                "start_inclusive": spec.train_start.isoformat(),
                "cutoff_exclusive": spec.cutoff.isoformat(),
            },
            "evaluation": {
                "start_inclusive": spec.cutoff.isoformat(),
                "end_inclusive_for_target_maturity": spec.evaluation_end.isoformat(),
            },
        },
        "context": {"sequence_length": seq_len, "activity_hours": activity_hours},
        "phase5_encoder_identity_replayed": identity_replayed,
        "target_rules": {
            "future_interval": "(t,t+H]",
            "all_future_closes_exact_native_timestamps": True,
            "all_future_closes_observed": True,
            "same_contract_and_gap_segment": True,
            "zero_targets_retained": True,
            "future_update_count_used_for_eligibility": False,
            "retrospective_end_metadata_used_for_eligibility": False,
        },
        "evaluation_outcome_policy": {
            "capacity_only": True,
            "target_values_computed_or_emitted": False,
            "used_for_horizon_selection": False,
        },
        "supported_contract_count": len(support),
        "source_contract_count": len(source_contracts),
        "counts": walk_counts,
    }
    return VolatilityWalkAudit(
        capacity=pd.DataFrame(capacity_rows),
        distribution=pd.DataFrame(distribution_rows),
        dependence=pd.DataFrame(dependence_rows),
        strata=pd.DataFrame(strata_rows),
        manifest=manifest,
    )


def combine_walk_audits(audits: Sequence[VolatilityWalkAudit]) -> dict[str, pd.DataFrame]:
    if not audits:
        raise ValueError("at least one walk audit is required")
    return {
        "capacity": pd.concat([audit.capacity for audit in audits], ignore_index=True),
        "distribution": pd.concat([audit.distribution for audit in audits], ignore_index=True),
        "dependence": pd.concat([audit.dependence for audit in audits], ignore_index=True),
        "strata": pd.concat([audit.strata for audit in audits], ignore_index=True),
    }


def _frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_audit_tables(
    output_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    walk_manifests: Sequence[Mapping[str, Any]],
    source_provenance: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the four canonical tables and their replay manifest."""

    output_dir = Path(output_dir).resolve()
    names = {
        "capacity": CAPACITY_FILENAME,
        "distribution": TARGET_DISTRIBUTION_FILENAME,
        "dependence": TARGET_DEPENDENCE_FILENAME,
        "strata": TARGET_STRATA_FILENAME,
    }
    expected = [output_dir / value for value in names.values()] + [output_dir / MANIFEST_FILENAME]
    occupied = [path for path in expected if path.exists()]
    if occupied and not overwrite:
        raise FileExistsError(f"refusing to overwrite Phase 6 audit artifacts: {occupied}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for key, filename in names.items():
        frame = frames[key]
        path = output_dir / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
        artifacts[key] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
            "canonical_frame_sha256": _frame_hash(frame),
        }
    plot_paths = _write_audit_plots(output_dir, frames)
    report_path = _write_audit_report(output_dir, frames)
    for index, path in enumerate(plot_paths):
        artifacts[f"plot_{index + 1}"] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    artifacts["report"] = {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "bytes": report_path.stat().st_size,
    }
    manifest = {
        "schema_version": PHASE6_VOLATILITY_AUDIT_SCHEMA_VERSION,
        "phase": 6,
        "stage": "A_read_only_horizon_audit",
        "primary_horizon_frozen": False,
        "formula_version": FORMULA_VERSION,
        "candidate_horizons_hours": list(
            parse_candidate_horizons(walk_manifests[0]["candidate_horizons_hours"])
        ),
        "evaluation_outcomes_used": False,
        "walks": list(walk_manifests),
        "source_provenance": dict(source_provenance),
        "artifacts": artifacts,
        "git_commit": _git_commit(),
    }
    write_json(output_dir / MANIFEST_FILENAME, manifest)
    return manifest


def _write_audit_plots(
    output_dir: Path, frames: Mapping[str, pd.DataFrame]
) -> list[Path]:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phase6-volatility-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    distribution = frames["distribution"].copy()
    capacity = frames["capacity"].copy()
    aggregate = capacity.loc[capacity["scope"].eq("aggregate")]
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for walk, group in distribution.groupby("walk", sort=True):
        axes[0].plot(group["horizon_hours"], group["zero_rate"], marker="o", label=f"Walk {walk}")
        axes[1].plot(
            group["horizon_hours"], group["largest_interval_share"], marker="o", label=f"Walk {walk}"
        )
    axes[0].set(title="Training target zero rate", xlabel="Horizon (hours)", ylabel="Fraction")
    axes[1].set(
        title="Largest interval share of aggregate training RV",
        xlabel="Horizon (hours)",
        ylabel="Fraction",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    path = plot_dir / "training_target_degeneracy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for split, axis in zip(("train", "evaluation"), axes, strict=True):
        subset = aggregate.loc[aggregate["split"].eq(split)]
        for walk, group in subset.groupby("walk", sort=True):
            ratio = group["eligible_rows"] / group["candidate_rows"].replace(0, np.nan)
            axis.plot(group["horizon_hours"], ratio, marker="o", label=f"Walk {walk}")
        axis.set(
            title=f"{split.title()} target capacity",
            xlabel="Horizon (hours)",
            ylabel="Eligible / candidate rows",
            ylim=(0.0, 1.02),
        )
        axis.grid(alpha=0.25)
        axis.legend()
    path = plot_dir / "horizon_capacity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def _write_audit_report(output_dir: Path, frames: Mapping[str, pd.DataFrame]) -> Path:
    distribution = frames["distribution"].sort_values(["walk", "horizon_hours"])
    capacity = frames["capacity"]
    aggregate = capacity.loc[capacity["scope"].eq("aggregate")]
    lines = [
        "# Phase 6 volatility horizon audit",
        "",
        "This is the Stage A, training-period-only audit. It does not freeze a primary horizon, "
        "construct a model-ready label bundle, or authorize model training. Evaluation outcomes "
        "were not calculated or used; only evaluation row and contract capacity is reported.",
        "",
        f"Target: `{FORMULA_VERSION}` — `RV(t,H) = sum((p[t+j] - p[t+j-1])^2, j=1..H)`.",
        "",
        "## Candidate summary",
        "",
        "| Walk | H | Train rows | Train contracts | Eval rows | Eval contracts | Train zero rate | Largest interval share |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in distribution.itertuples(index=False):
        train = aggregate.loc[
            aggregate["walk"].eq(row.walk)
            & aggregate["horizon_hours"].eq(row.horizon_hours)
            & aggregate["split"].eq("train")
        ].iloc[0]
        evaluation = aggregate.loc[
            aggregate["walk"].eq(row.walk)
            & aggregate["horizon_hours"].eq(row.horizon_hours)
            & aggregate["split"].eq("evaluation")
        ].iloc[0]
        lines.append(
            f"| {int(row.walk)} | {int(row.horizon_hours)} | {int(train.eligible_rows):,} | "
            f"{int(train.eligible_contracts)} | {int(evaluation.eligible_rows):,} | "
            f"{int(evaluation.eligible_contracts)} | {row.zero_rate:.4f} | "
            f"{row.largest_interval_share:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation gate",
            "",
            "Choose the primary horizon only after reviewing capacity in both walks, training-target "
            "degeneracy and concentration, dependence, and the reporting strata. The choice must be "
            "recorded in a dated amendment before label-bundle construction. This report intentionally "
            "does not nominate or freeze that horizon.",
            "",
            "Drop counts in `horizon_capacity.csv` are diagnostic flags and can overlap (for example, "
            "a missing future timestamp is also a gap-segment break).",
            "",
        ]
    )
    path = output_dir / REPORT_FILENAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)
    return path


def validate_audit_artifacts(output_dir: Path) -> dict[str, Any]:
    """Validate artifact hashes, schemas, and the training-only output contract."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PHASE6_VOLATILITY_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 6 volatility audit schema")
    if manifest.get("primary_horizon_frozen") is not False:
        raise ValueError("Stage A audit must not freeze a primary horizon")
    if manifest.get("evaluation_outcomes_used") is not False:
        raise ValueError("Stage A audit must not use evaluation outcomes")
    for walk in manifest.get("walks", []):
        policy = walk.get("evaluation_outcome_policy", {})
        if policy.get("target_values_computed_or_emitted") is not False:
            raise ValueError("walk manifest does not exclude evaluation target values")
    frames: dict[str, pd.DataFrame] = {}
    table_keys = {"capacity", "distribution", "dependence", "strata"}
    for key, artifact in manifest["artifacts"].items():
        path = Path(artifact["path"])
        if not path.is_absolute():
            path = output_dir / path
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Phase 6 audit artifact hash mismatch: {path}")
        if key in table_keys:
            frame = pd.read_csv(path)
            if len(frame) != int(artifact["rows"]):
                raise ValueError(f"Phase 6 audit row count mismatch: {path}")
            frames[key] = frame
    if set(frames["distribution"]["split"]) != {"train"}:
        raise ValueError("target distributions must contain training rows only")
    if set(frames["dependence"]["split"]) != {"train"}:
        raise ValueError("target dependence must contain training rows only")
    if set(frames["strata"]["split"]) != {"train"}:
        raise ValueError("target strata must contain training rows only")
    if set(frames["capacity"]["split"]) != {"train", "evaluation"}:
        raise ValueError("capacity must report both train and evaluation partitions")
    forbidden = {"target", "realised_variance", "rv_value", "target_mean"}
    evaluation_capacity = frames["capacity"].loc[frames["capacity"]["split"].eq("evaluation")]
    if forbidden & set(evaluation_capacity.columns):
        raise ValueError("evaluation capacity leaks target-value columns")
    return {
        "walks": [int(walk["walk"]) for walk in manifest["walks"]],
        "candidate_horizons_hours": manifest["candidate_horizons_hours"],
        "artifact_rows": {key: int(len(frame)) for key, frame in frames.items()},
        "evaluation_outcomes_used": False,
    }


def validate_horizon_freeze_manifest(
    freeze_path: Path, repository_root: Path
) -> dict[str, Any]:
    """Replay the approved H=8 decision against the immutable Stage A audit."""

    repository_root = Path(repository_root)
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != PHASE6_VOLATILITY_FREEZE_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 6 volatility horizon-freeze schema")
    decision = freeze.get("decision", {})
    if decision.get("primary_horizon_hours") != FROZEN_PRIMARY_HORIZON_HOURS:
        raise ValueError("Phase 6 primary volatility horizon must be frozen to eight hours")
    if decision.get("formula_version") != FORMULA_VERSION:
        raise ValueError("horizon freeze formula version mismatch")
    gates = freeze.get("gate_status", {})
    if gates != {
        "horizon_freeze_complete": True,
        "label_bundle_authorized": True,
        "label_bundle_complete": False,
        "model_training_authorized": False,
    }:
        raise ValueError("horizon-freeze execution gates are inconsistent")
    if freeze.get("approval", {}).get("evaluation_outcomes_used") is not False:
        raise ValueError("horizon freeze must exclude evaluation outcomes")

    evidence = freeze["audit_evidence"]
    evidence_paths = {
        "audit_manifest": repository_root / evidence["audit_manifest_path"],
        "horizon_capacity": repository_root
        / "experiments/phase6/volatility_prediction/data_exploration/horizon_capacity.csv",
        "target_distribution": repository_root
        / "experiments/phase6/volatility_prediction/data_exploration/target_distribution.csv",
        "target_dependence": repository_root
        / "experiments/phase6/volatility_prediction/data_exploration/target_dependence.csv",
    }
    expected_hashes = {
        "audit_manifest": evidence["audit_manifest_sha256"],
        "horizon_capacity": evidence["horizon_capacity_sha256"],
        "target_distribution": evidence["target_distribution_sha256"],
        "target_dependence": evidence["target_dependence_sha256"],
    }
    for name, path in evidence_paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"horizon-freeze evidence hash mismatch: {path}")
    amendment = freeze["supporting_amendment"]
    amendment_path = repository_root / amendment["path"]
    if not amendment_path.is_file() or sha256_file(amendment_path) != amendment["sha256"]:
        raise ValueError("horizon-freeze amendment hash mismatch")

    audit_manifest = json.loads(evidence_paths["audit_manifest"].read_text(encoding="utf-8"))
    if audit_manifest.get("primary_horizon_frozen") is not False:
        raise ValueError("the pre-decision audit manifest must remain unfrozen")
    if audit_manifest.get("evaluation_outcomes_used") is not False:
        raise ValueError("the source audit used evaluation outcomes")
    if audit_manifest.get("candidate_horizons_hours") != list(DEFAULT_CANDIDATE_HORIZONS):
        raise ValueError("source audit candidate horizons do not replay")

    capacity = pd.read_csv(evidence_paths["horizon_capacity"])
    distribution = pd.read_csv(evidence_paths["target_distribution"])
    dependence = pd.read_csv(evidence_paths["target_dependence"])
    for walk in (1, 2):
        summary = freeze["evidence_summary"][f"walk{walk}"]
        train_capacity = capacity.loc[
            capacity["walk"].eq(walk)
            & capacity["horizon_hours"].eq(FROZEN_PRIMARY_HORIZON_HOURS)
            & capacity["split"].eq("train")
            & capacity["scope"].eq("aggregate")
        ].iloc[0]
        evaluation_capacity = capacity.loc[
            capacity["walk"].eq(walk)
            & capacity["horizon_hours"].eq(FROZEN_PRIMARY_HORIZON_HOURS)
            & capacity["split"].eq("evaluation")
            & capacity["scope"].eq("aggregate")
        ].iloc[0]
        target_distribution = distribution.loc[
            distribution["walk"].eq(walk)
            & distribution["horizon_hours"].eq(FROZEN_PRIMARY_HORIZON_HOURS)
        ].iloc[0]
        target_dependence = dependence.loc[
            dependence["walk"].eq(walk)
            & dependence["horizon_hours"].eq(FROZEN_PRIMARY_HORIZON_HOURS)
        ].iloc[0]
        exact = {
            "training_rows": int(train_capacity["eligible_rows"]),
            "training_contracts": int(train_capacity["eligible_contracts"]),
            "evaluation_capacity_rows": int(evaluation_capacity["eligible_rows"]),
            "evaluation_capacity_contracts": int(evaluation_capacity["eligible_contracts"]),
        }
        if any(int(summary[key]) != value for key, value in exact.items()):
            raise ValueError(f"walk {walk} frozen capacity evidence does not replay")
        approximate = {
            "training_zero_rate": float(target_distribution["zero_rate"]),
            "largest_interval_share": float(target_distribution["largest_interval_share"]),
            "rv_lag1_pearson": float(target_dependence["rv_lag1_pearson"]),
            "historical_future_spearman": float(
                target_dependence["historical_future_spearman"]
            ),
        }
        if any(
            not np.isclose(float(summary[key]), value, rtol=0.0, atol=5e-7)
            for key, value in approximate.items()
        ):
            raise ValueError(f"walk {walk} frozen distribution/dependence evidence does not replay")
    return {
        "primary_horizon_hours": FROZEN_PRIMARY_HORIZON_HOURS,
        "formula_version": FORMULA_VERSION,
        "horizon_freeze_complete": True,
        "label_bundle_authorized": True,
        "model_training_authorized": False,
    }
