"""Explore native 15-minute/one-hour dynamics and bounded forward filling.

This is descriptive EDA only. Forward-filled rows are simulated and measured;
they do not overwrite the raw or clean source artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RECENT_ROOT = (
    ROOT
    / "data_new/findata/polymarket/historical_diverse_top50_2025-12-01_2026-08-31"
)
DEFAULT_OUTPUT = RECENT_ROOT / "analysis/native_15m_1h_dynamics"
THRESHOLD = 0.005


@dataclass
class Stats:
    targets: int = 0
    zero: int = 0
    meaningful_005: int = 0
    move_gt_01: int = 0
    move_gt_05: int = 0
    absolute_sum: float = 0.0
    squared_sum: float = 0.0

    def update(self, delta: np.ndarray) -> None:
        if not len(delta):
            return
        absolute = np.abs(delta)
        self.targets += len(delta)
        self.zero += int((delta == 0).sum())
        self.meaningful_005 += int((absolute > THRESHOLD).sum())
        self.move_gt_01 += int((absolute > 0.01).sum())
        self.move_gt_05 += int((absolute > 0.05).sum())
        self.absolute_sum += float(absolute.sum())
        self.squared_sum += float(np.square(delta).sum())

    def record(self) -> dict[str, object]:
        denominator = self.targets or 1
        return {
            "targets": self.targets,
            "zero_fraction": self.zero / denominator if self.targets else None,
            "meaningful_005_fraction": self.meaningful_005 / denominator if self.targets else None,
            "move_gt_01_fraction": self.move_gt_01 / denominator if self.targets else None,
            "move_gt_05_fraction": self.move_gt_05 / denominator if self.targets else None,
            "mean_absolute_move": self.absolute_sum / denominator if self.targets else None,
            "root_mean_square_move": math.sqrt(self.squared_sum / denominator) if self.targets else None,
        }


@dataclass
class Cohort:
    name: str
    frequency_minutes: int
    maximum_fill_bars: int
    source: str
    files_considered: int = 0
    contracts_with_rows: int = 0
    rows: int = 0
    nonfinite_rows_dropped: int = 0
    total_missing_grid_rows: int = 0
    bounded_fill_rows: int = 0
    long_gap_rows_not_filled: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    stats: dict[tuple[int, str], Stats] = field(default_factory=dict)
    contract_records: list[dict[str, object]] = field(default_factory=list)


def prepare_output(path: Path, overwrite: bool) -> None:
    allowed = (ROOT / "data_new").resolve()
    resolved = path.resolve()
    if allowed != resolved and allowed not in resolved.parents:
        raise ValueError(f"output must be inside {allowed}: {resolved}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite occupied output: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize(frame: pd.DataFrame, *, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
    if cutoff is not None:
        result = result.loc[result["date"].lt(cutoff)].copy()
    identity = ["condition_id", "date"] if "condition_id" in result else ["date"]
    result = result.sort_values(identity).drop_duplicates(identity, keep="last")
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.reset_index(drop=True)


def remove_nonfinite(frame: pd.DataFrame, cohort: Cohort) -> pd.DataFrame:
    columns = ["open", "high", "low", "close", "volume"]
    finite = np.isfinite(frame[columns].to_numpy(np.float64)).all(axis=1)
    cohort.nonfinite_rows_dropped += int((~finite).sum())
    return frame.loc[finite].reset_index(drop=True)


def bounded_forward_fill(
    dates: np.ndarray,
    close: np.ndarray,
    *,
    step: np.timedelta64,
    maximum_fill_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    if not len(dates):
        return dates, close, np.empty(0, dtype=bool), 0, 0, 0
    output_dates: list[np.datetime64] = [dates[0]]
    output_close: list[float] = [float(close[0])]
    observed: list[bool] = [True]
    total_missing = 0
    filled = 0
    unfilled = 0
    step_ns = int(step.astype("timedelta64[ns]").astype(np.int64))
    for index in range(1, len(dates)):
        gap_ns = int((dates[index] - dates[index - 1]).astype("timedelta64[ns]").astype(np.int64))
        missing = max(gap_ns // step_ns - 1, 0) if gap_ns % step_ns == 0 else 0
        total_missing += missing
        if 0 < missing <= maximum_fill_bars:
            for offset in range(1, missing + 1):
                output_dates.append(dates[index - 1] + offset * step)
                output_close.append(float(close[index - 1]))
                observed.append(False)
            filled += missing
        else:
            unfilled += missing
        output_dates.append(dates[index])
        output_close.append(float(close[index]))
        observed.append(True)
    return (
        np.asarray(output_dates),
        np.asarray(output_close, dtype=np.float64),
        np.asarray(observed, dtype=bool),
        total_missing,
        filled,
        unfilled,
    )


def deltas_at_horizon(
    dates: np.ndarray,
    close: np.ndarray,
    observed: np.ndarray,
    *,
    horizon_bars: int,
    step: np.timedelta64,
) -> tuple[np.ndarray, np.ndarray]:
    if len(close) <= horizon_bars:
        return np.empty(0), np.empty(0, dtype=bool)
    valid = np.ones(len(close) - horizon_bars, dtype=bool)
    for offset in range(horizon_bars):
        valid &= dates[offset + 1 : len(close) - horizon_bars + offset + 1] - dates[
            offset : len(close) - horizon_bars + offset
        ] == step
    starts = np.flatnonzero(valid)
    delta = close[starts + horizon_bars] - close[starts]
    touches_fill = np.zeros(len(starts), dtype=bool)
    for offset in range(horizon_bars + 1):
        touches_fill |= ~observed[starts + offset]
    return delta, touches_fill


def add_contract(cohort: Cohort, frame: pd.DataFrame, contract_id: str, horizons: list[int]) -> None:
    cohort.files_considered += 1
    frame = remove_nonfinite(frame, cohort)
    if frame.empty:
        return
    cohort.contracts_with_rows += 1
    cohort.rows += len(frame)
    cohort.start = frame["date"].iloc[0] if cohort.start is None else min(cohort.start, frame["date"].iloc[0])
    cohort.end = frame["date"].iloc[-1] if cohort.end is None else max(cohort.end, frame["date"].iloc[-1])
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    close = frame["close"].to_numpy(np.float64)
    step = np.timedelta64(cohort.frequency_minutes, "m")
    observed = np.ones(len(frame), dtype=bool)
    filled_dates, filled_close, filled_observed, missing, filled, unfilled = bounded_forward_fill(
        dates,
        close,
        step=step,
        maximum_fill_bars=cohort.maximum_fill_bars,
    )
    cohort.total_missing_grid_rows += missing
    cohort.bounded_fill_rows += filled
    cohort.long_gap_rows_not_filled += unfilled
    record: dict[str, object] = {
        "cohort": cohort.name,
        "contract_id": contract_id,
        "rows": len(frame),
        "missing_grid_rows": missing,
        "bounded_fill_rows": filled,
        "unfilled_long_gap_rows": unfilled,
    }
    for horizon in horizons:
        raw_delta, _ = deltas_at_horizon(
            dates, close, observed, horizon_bars=horizon, step=step
        )
        filled_delta, touches_fill = deltas_at_horizon(
            filled_dates,
            filled_close,
            filled_observed,
            horizon_bars=horizon,
            step=step,
        )
        untouched_delta = filled_delta[~touches_fill]
        touched_delta = filled_delta[touches_fill]
        for mode, delta in (
            ("strict_observed", raw_delta),
            ("bounded_ffill_all", filled_delta),
            ("bounded_ffill_untouched", untouched_delta),
            ("bounded_ffill_touched", touched_delta),
        ):
            cohort.stats.setdefault((horizon, mode), Stats()).update(delta)
        absolute = np.abs(raw_delta)
        prefix = f"h{horizon}_strict"
        record[f"{prefix}_targets"] = len(raw_delta)
        record[f"{prefix}_zero_fraction"] = float((raw_delta == 0).mean()) if len(raw_delta) else np.nan
        record[f"{prefix}_meaningful_005_fraction"] = (
            float((absolute > THRESHOLD).mean()) if len(raw_delta) else np.nan
        )
        record[f"{prefix}_mean_absolute_move"] = float(absolute.mean()) if len(raw_delta) else np.nan
    cohort.contract_records.append(record)


def load_recent(path: Path, cohort: Cohort, horizons: list[int]) -> None:
    frame = normalize(pd.read_parquet(path))
    for condition_id, contract in frame.groupby("condition_id", sort=True):
        add_contract(cohort, contract.reset_index(drop=True), str(condition_id), horizons)


def summary_records(cohort: Cohort) -> list[dict[str, object]]:
    records = []
    for (horizon, mode), stats in sorted(cohort.stats.items()):
        record = {
            "cohort": cohort.name,
            "frequency_minutes": cohort.frequency_minutes,
            "horizon_bars": horizon,
            "horizon_minutes": horizon * cohort.frequency_minutes,
            "mode": mode,
            **stats.record(),
        }
        records.append(record)
    return records


def cohort_record(cohort: Cohort) -> dict[str, object]:
    return {
        "cohort": cohort.name,
        "source": cohort.source,
        "frequency_minutes": cohort.frequency_minutes,
        "maximum_fill_bars": cohort.maximum_fill_bars,
        "files_considered": cohort.files_considered,
        "contracts_with_rows": cohort.contracts_with_rows,
        "rows": cohort.rows,
        "nonfinite_rows_dropped": cohort.nonfinite_rows_dropped,
        "start": cohort.start,
        "end": cohort.end,
        "missing_grid_rows": cohort.total_missing_grid_rows,
        "bounded_fill_rows": cohort.bounded_fill_rows,
        "unfilled_long_gap_rows": cohort.long_gap_rows_not_filled,
        "bounded_fill_fraction_of_observed": (
            cohort.bounded_fill_rows / cohort.rows if cohort.rows else None
        ),
    }


def pct(value: object) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.2%}"


def render_report(
    cohorts: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    maximum_fill_bars: int,
) -> str:
    lines = [
        "# Native 15-Minute and One-Hour Dynamics",
        "",
        "Descriptive EDA only. Raw and clean artifacts are read without modification. "
        "Bounded forward filling is simulated in memory.",
        "",
        "## Coverage and gaps",
        "",
        "| Cohort | Frequency | Contracts | Observed rows | Missing grid rows | Bounded fills | Fill/observed | Long-gap rows left missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cohorts.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.frequency_minutes}m | {row.contracts_with_rows:,} | "
            f"{row.rows:,} | {row.missing_grid_rows:,} | {row.bounded_fill_rows:,} | "
            f"{pct(row.bounded_fill_fraction_of_observed)} | {row.unfilled_long_gap_rows:,} |"
        )
    lines.extend(
        [
            "",
            f"The bounded policy fills at most {maximum_fill_bars} missing native bar(s): "
            f"{15 * maximum_fill_bars} minutes for 15-minute data and "
            f"{maximum_fill_bars} hour(s) for one-hour data. Longer gaps remain gaps.",
            "",
            "## Probability movement",
            "",
            "| Cohort | Horizon | Mode | Targets | Exact zero | >0.005 | >0.01 | >0.05 | Mean abs move |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    shown = summary.loc[summary["mode"].isin(["strict_observed", "bounded_ffill_all", "bounded_ffill_touched"])]
    for row in shown.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.horizon_minutes}m | {row.mode} | {row.targets:,} | "
            f"{pct(row.zero_fraction)} | {pct(row.meaningful_005_fraction)} | "
            f"{pct(row.move_gt_01_fraction)} | {pct(row.move_gt_05_fraction)} | "
            f"{row.mean_absolute_move:.6f} |"
        )
    lines.extend(
        [
            "",
            "`bounded_ffill_touched` isolates targets whose interval contains at least one "
            "synthetic row. It is diagnostic and should not be interpreted as newly observed market movement.",
            "",
            "## Filling policy",
            "",
            "Bounded zero-volume forward-fill candles are defensible when accompanied by an "
            "imputation mask and time-since-last-observation feature. This analysis applies no "
            "random augmentation and does not write simulated movement into OHLCV or targets.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--maximum-fill-bars",
        type=int,
        default=1,
        help="maximum consecutive missing native bars to forward fill",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.maximum_fill_bars < 0:
        parser.error("maximum-fill-bars must be non-negative")
    prepare_output(args.output_dir, args.overwrite)

    cohorts = [
        Cohort("recent_raw_1h", 60, args.maximum_fill_bars, "FinData native one-hour raw"),
        Cohort("recent_clean_1h", 60, args.maximum_fill_bars, "FinData native one-hour after quarantine"),
        Cohort("recent_raw_15m", 15, args.maximum_fill_bars, "FinData native 15-minute raw"),
        Cohort("recent_clean_15m", 15, args.maximum_fill_bars, "FinData native 15-minute after quarantine"),
    ]
    print("Loading only the recent FinData one-hour and 15-minute cohorts...", flush=True)
    load_recent(RECENT_ROOT / "candles_1h.parquet", cohorts[0], [1, 2])
    load_recent(RECENT_ROOT / "candles_1h_clean.parquet", cohorts[1], [1, 2])
    load_recent(RECENT_ROOT / "candles_15min.parquet", cohorts[2], [1, 2, 4])
    load_recent(RECENT_ROOT / "candles_15min_clean.parquet", cohorts[3], [1, 2, 4])

    cohort_frame = pd.DataFrame([cohort_record(cohort) for cohort in cohorts])
    summary = pd.DataFrame(
        [record for cohort in cohorts for record in summary_records(cohort)]
    )
    verification_columns = [
        "targets",
        "zero_fraction",
        "meaningful_005_fraction",
        "move_gt_01_fraction",
        "move_gt_05_fraction",
        "mean_absolute_move",
        "root_mean_square_move",
    ]
    for (cohort_name, horizon), group in summary.groupby(["cohort", "horizon_bars"]):
        indexed = group.set_index("mode")
        observed = indexed.loc["strict_observed", verification_columns]
        untouched = indexed.loc["bounded_ffill_untouched", verification_columns]
        if not np.allclose(
            observed.to_numpy(np.float64),
            untouched.to_numpy(np.float64),
            equal_nan=True,
        ):
            raise AssertionError(
                f"untouched filled-grid targets differ from strict observed targets: "
                f"{cohort_name=} {horizon=}"
            )
    contracts = pd.concat(
        [pd.DataFrame(cohort.contract_records) for cohort in cohorts], ignore_index=True
    )
    cohort_frame.to_csv(args.output_dir / "cohort_gap_summary.csv", index=False)
    summary.to_csv(args.output_dir / "movement_summary.csv", index=False)
    contracts.to_parquet(args.output_dir / "contract_metrics.parquet", index=False)
    report = render_report(
        cohort_frame,
        summary,
        maximum_fill_bars=args.maximum_fill_bars,
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "descriptive native-frequency movement and bounded-forward-fill EDA",
        "source_scope": "newly collected FinData cohort only; old feather archive excluded",
        "stable_threshold": THRESHOLD,
        "bounded_fill_policy": {
            "maximum_missing_bars": args.maximum_fill_bars,
            "15m_maximum_duration_minutes": 15 * args.maximum_fill_bars,
            "1h_maximum_duration_minutes": 60 * args.maximum_fill_bars,
            "fill_close": "last observed close",
            "recommended_OHLC": "all equal to last observed close",
            "recommended_volume": 0,
            "raw_files_modified": False,
        },
        "cohorts": cohort_frame.to_dict(orient="records"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
