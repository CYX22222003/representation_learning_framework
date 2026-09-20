"""Audit internal timestamp gaps in the new FinData 15-minute/one-hour cohort."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data_new/findata/polymarket/historical_diverse_top50_2025-12-01_2026-08-31"
)
DEFAULT_OUTPUT = DEFAULT_INPUT / "analysis/native_gap_audit"


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


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"condition_id", "date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candle file is missing columns: {missing}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
    result = (
        result.sort_values(["condition_id", "date"])
        .drop_duplicates(["condition_id", "date"], keep="last")
        .reset_index(drop=True)
    )
    return result


def gap_tables(
    frame: pd.DataFrame,
    *,
    resolution: str,
    interval_minutes: int,
    title_by_condition: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step = pd.Timedelta(minutes=interval_minutes)
    gap_records: list[dict[str, object]] = []
    contract_records: list[dict[str, object]] = []
    missing_records: list[dict[str, object]] = []
    for condition_id, contract in frame.groupby("condition_id", sort=True):
        contract = contract.sort_values("date").reset_index(drop=True)
        dates = contract["date"]
        differences = dates.diff()
        ratios = differences / step
        if not ratios.dropna().apply(float.is_integer).all():
            raise ValueError(f"off-grid timestamp difference for {condition_id} at {resolution}")
        missing_bars = ratios.fillna(1).astype(int).sub(1)
        gap_indices = np.flatnonzero(missing_bars.to_numpy() > 0)
        for index in gap_indices:
            count = int(missing_bars.iloc[index])
            previous_date = dates.iloc[index - 1]
            next_date = dates.iloc[index]
            gap_records.append(
                {
                    "resolution": resolution,
                    "condition_id": condition_id,
                    "question": title_by_condition.get(condition_id),
                    "previous_date": previous_date,
                    "next_date": next_date,
                    "missing_bars": count,
                    "gap_duration_minutes": float((next_date - previous_date) / pd.Timedelta(minutes=1)),
                }
            )
            for offset in range(1, count + 1):
                missing_records.append(
                    {
                        "resolution": resolution,
                        "condition_id": condition_id,
                        "date": previous_date + offset * step,
                    }
                )
        span_rows = int((dates.iloc[-1] - dates.iloc[0]) / step) + 1
        total_missing = int(missing_bars.sum())
        contract_records.append(
            {
                "resolution": resolution,
                "condition_id": condition_id,
                "question": title_by_condition.get(condition_id),
                "observed_rows": len(contract),
                "start": dates.iloc[0],
                "end": dates.iloc[-1],
                "internal_grid_rows": span_rows,
                "internal_missing_rows": total_missing,
                "internal_coverage_fraction": len(contract) / span_rows,
                "gap_events": len(gap_indices),
                "one_bar_gap_events": int((missing_bars == 1).sum()),
                "two_to_four_bar_gap_events": int(missing_bars.between(2, 4).sum()),
                "over_four_bar_gap_events": int((missing_bars > 4).sum()),
                "maximum_missing_bars": int(missing_bars.max()),
                "maximum_gap_hours": float((missing_bars.max() + 1) * interval_minutes / 60),
                "cap1_fill_rows": int(missing_bars.loc[missing_bars <= 1].sum()),
                "cap4_fill_rows": int(missing_bars.loc[missing_bars <= 4].sum()),
            }
        )
    return (
        pd.DataFrame(gap_records),
        pd.DataFrame(contract_records),
        pd.DataFrame(missing_records),
    )


def distribution(gaps: pd.DataFrame) -> pd.DataFrame:
    labels = ["1", "2", "3", "4", "5-8", "9-16", "17-32", "33-96", ">96"]
    values = gaps["missing_bars"]
    band = pd.cut(
        values,
        bins=[0, 1, 2, 3, 4, 8, 16, 32, 96, np.inf],
        labels=labels,
        include_lowest=True,
        right=True,
    )
    result = (
        gaps.assign(gap_band=band)
        .groupby(["resolution", "gap_band"], observed=False)
        .agg(gap_events=("missing_bars", "size"), missing_rows=("missing_bars", "sum"))
        .reset_index()
    )
    totals = result.groupby("resolution")["missing_rows"].transform("sum")
    result["missing_row_fraction"] = result["missing_rows"] / totals
    return result


def cross_resolution(
    missing_1h: pd.DataFrame,
    missing_15m: pd.DataFrame,
    clean_1h: pd.DataFrame,
    clean_15m: pd.DataFrame,
) -> dict[str, object]:
    observed_15m = clean_15m[["condition_id", "date"]].copy()
    observed_15m["hour"] = observed_15m["date"].dt.floor("1h")
    bars_per_hour = (
        observed_15m.groupby(["condition_id", "hour"])["date"]
        .nunique()
        .rename("observed_15m_bars")
        .reset_index()
    )
    missing_hour = missing_1h.rename(columns={"date": "hour"}).merge(
        bars_per_hour,
        on=["condition_id", "hour"],
        how="left",
    )
    missing_hour["observed_15m_bars"] = missing_hour["observed_15m_bars"].fillna(0).astype(int)

    observed_hours = clean_1h[["condition_id", "date"]].rename(columns={"date": "hour"})
    missing_quarter = missing_15m.copy()
    missing_quarter["hour"] = missing_quarter["date"].dt.floor("1h")
    missing_quarter = missing_quarter.merge(
        observed_hours.assign(native_1h_present=True),
        on=["condition_id", "hour"],
        how="left",
    )
    native_hour_present = missing_quarter["native_1h_present"].eq(True)
    return {
        "missing_native_1h_slots": len(missing_hour),
        "missing_1h_with_any_15m_activity": int(missing_hour["observed_15m_bars"].gt(0).sum()),
        "missing_1h_with_any_15m_activity_fraction": float(
            missing_hour["observed_15m_bars"].gt(0).mean()
        ),
        "missing_1h_with_four_15m_bars": int(missing_hour["observed_15m_bars"].eq(4).sum()),
        "missing_1h_with_four_15m_bars_fraction": float(
            missing_hour["observed_15m_bars"].eq(4).mean()
        ),
        "missing_native_15m_slots": len(missing_quarter),
        "missing_15m_inside_present_native_1h": int(native_hour_present.sum()),
        "missing_15m_inside_present_native_1h_fraction": float(native_hour_present.mean()),
    }


def dataset_summary(
    frame: pd.DataFrame,
    contracts: pd.DataFrame,
    gaps: pd.DataFrame,
    *,
    resolution: str,
    interval_minutes: int,
) -> dict[str, object]:
    missing = int(contracts["internal_missing_rows"].sum())
    observed = len(frame)
    return {
        "resolution": resolution,
        "contracts": int(frame["condition_id"].nunique()),
        "observed_rows": observed,
        "internal_grid_rows": observed + missing,
        "internal_missing_rows": missing,
        "pooled_internal_coverage_fraction": observed / (observed + missing),
        "gap_events": len(gaps),
        "one_bar_gap_events": int(gaps["missing_bars"].eq(1).sum()),
        "over_four_bar_gap_events": int(gaps["missing_bars"].gt(4).sum()),
        "cap1_fill_rows": int(gaps.loc[gaps["missing_bars"].le(1), "missing_bars"].sum()),
        "cap1_fraction_of_missing": float(
            gaps.loc[gaps["missing_bars"].le(1), "missing_bars"].sum() / missing
        ),
        "cap4_fill_rows": int(gaps.loc[gaps["missing_bars"].le(4), "missing_bars"].sum()),
        "cap4_fraction_of_missing": float(
            gaps.loc[gaps["missing_bars"].le(4), "missing_bars"].sum() / missing
        ),
        "maximum_missing_bars": int(gaps["missing_bars"].max()),
        "maximum_missing_duration_hours": float(
            gaps["missing_bars"].max() * interval_minutes / 60
        ),
        "maximum_observation_gap_hours": float(
            (gaps["missing_bars"].max() + 1) * interval_minutes / 60
        ),
        "median_contract_coverage": float(contracts["internal_coverage_fraction"].median()),
        "p10_contract_coverage": float(contracts["internal_coverage_fraction"].quantile(0.10)),
        "minimum_contract_coverage": float(contracts["internal_coverage_fraction"].min()),
        "contracts_below_90pct_coverage": int(contracts["internal_coverage_fraction"].lt(0.90).sum()),
        "contracts_below_75pct_coverage": int(contracts["internal_coverage_fraction"].lt(0.75).sum()),
        "contracts_below_50pct_coverage": int(contracts["internal_coverage_fraction"].lt(0.50).sum()),
    }


def pct(value: float) -> str:
    return f"{value:.2%}"


def render_report(
    summaries: list[dict[str, object]],
    raw_summaries: list[dict[str, object]],
    gap_distribution: pd.DataFrame,
    cross: dict[str, object],
    contracts: pd.DataFrame,
) -> str:
    lines = [
        "# FinData Native Gap Audit",
        "",
        "This audit uses only the newly collected, pruned FinData 15-minute and one-hour "
        "files. It measures internal gaps between each contract's first and last observation; "
        "it does not infer unobserved time before the first or after the last row.",
        "",
        "## Dataset-level gaps",
        "",
        "| Resolution | Observed rows | Missing internal rows | Coverage | Gap events | One-bar gaps | Gaps >4 bars | Cap-1 coverage of missing | Cap-4 coverage of missing | Maximum missing run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['resolution']} | {row['observed_rows']:,} | "
            f"{row['internal_missing_rows']:,} | {pct(row['pooled_internal_coverage_fraction'])} | "
            f"{row['gap_events']:,} | {row['one_bar_gap_events']:,} | "
            f"{row['over_four_bar_gap_events']:,} | {pct(row['cap1_fraction_of_missing'])} | "
            f"{pct(row['cap4_fraction_of_missing'])} | {row['maximum_missing_bars']:,} |"
        )
    lines.extend(
        [
            "",
            "The longest missing run is 1,106 quarter-hours (276.5 hours) at 15m and "
            "275 hours at 1h. In both cases, the bounding observations are about 11.5 days "
            "apart.",
            "",
            "## Effect of anomaly pruning",
            "",
            "| Resolution | Raw rows | Clean rows | Quarantined | Raw missing | Clean missing | Coverage change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for raw, clean in zip(raw_summaries, summaries, strict=True):
        delta_points = 100 * (
            clean["pooled_internal_coverage_fraction"]
            - raw["pooled_internal_coverage_fraction"]
        )
        lines.append(
            f"| {clean['resolution']} | {raw['observed_rows']:,} | "
            f"{clean['observed_rows']:,} | "
            f"{raw['observed_rows'] - clean['observed_rows']:,} | "
            f"{raw['internal_missing_rows']:,} | {clean['internal_missing_rows']:,} | "
            f"{delta_points:+.3f} pp |"
        )
    lines.extend(
        [
            "",
            "Pruning accounts for only 116 additional missing 15-minute slots and 147 "
            "additional missing one-hour slots. The large majority of gaps were already "
            "present in the raw FinData response.",
        ]
    )
    lines.extend(
        [
            "",
            "## Gap-length distribution",
            "",
            "| Resolution | Missing bars per gap | Gap events | Missing rows | Share of missing rows |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in gap_distribution.itertuples(index=False):
        lines.append(
            f"| {row.resolution} | {row.gap_band} | {row.gap_events:,} | "
            f"{row.missing_rows:,} | {pct(row.missing_row_fraction)} |"
        )
    lines.extend(
        [
            "",
            "## Cross-resolution evidence",
            "",
            f"Of {cross['missing_native_1h_slots']:,} missing native-hour slots, "
            f"{cross['missing_1h_with_any_15m_activity']:,} "
            f"({pct(cross['missing_1h_with_any_15m_activity_fraction'])}) contain at least one "
            "native 15-minute candle, and "
            f"{cross['missing_1h_with_four_15m_bars']:,} "
            f"({pct(cross['missing_1h_with_four_15m_bars_fraction'])}) contain all four.",
            "",
            f"Conversely, {cross['missing_15m_inside_present_native_1h']:,}/"
            f"{cross['missing_native_15m_slots']:,} missing 15-minute slots "
            f"({pct(cross['missing_15m_inside_present_native_1h_fraction'])}) fall inside an hour "
            "for which the native one-hour endpoint returned a candle.",
            "",
            "This shows that missing rows are not always equivalent to a verified no-trade interval; "
            "the API resolutions can emit different sparse views.",
            "",
            "## Per-contract concentration",
            "",
            f"At 15m, median per-contract coverage is {pct(summaries[0]['median_contract_coverage'])}; "
            f"{summaries[0]['contracts_below_75pct_coverage']}/50 contracts are below 75% and "
            f"{summaries[0]['contracts_below_50pct_coverage']}/50 are below 50%. At 1h, median "
            f"coverage is {pct(summaries[1]['median_contract_coverage'])}; "
            f"{summaries[1]['contracts_below_75pct_coverage']}/50 are below 75% and "
            f"{summaries[1]['contracts_below_50pct_coverage']}/50 are below 50%.",
            "",
        ]
    )
    for resolution in ("15m", "1h"):
        part = contracts.loc[contracts["resolution"].eq(resolution)].nsmallest(
            5, "internal_coverage_fraction"
        )
        lines.extend(
            [
                f"Lowest-coverage {resolution} contracts:",
                "",
                "| Coverage | Observed | Missing | Max missing run | Contract |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for row in part.itertuples(index=False):
            lines.append(
                f"| {pct(row.internal_coverage_fraction)} | {row.observed_rows:,} | "
                f"{row.internal_missing_rows:,} | {row.maximum_missing_bars:,} | "
                f"{row.question or row.condition_id} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "The stored clean files have not been forward-filled. One-bar deterministic filling is "
            "a reasonable sensitivity when flat OHLC, zero volume, `is_imputed`, and time-since-last-"
            "observation are retained. Filling every long gap is not recommended: it would create "
            "large synthetic stale blocks, including runs lasting multiple days, and cross-resolution "
            "evidence shows that some absent rows occur despite activity at the other resolution. "
            "Long gaps should remain gaps or cause windows to be excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare_output(args.output_dir, args.overwrite)

    metadata = pd.read_parquet(args.input_dir / "market_metadata.parquet")
    title_column = "question_snapshot" if "question_snapshot" in metadata else "title"
    title_by_condition = metadata.drop_duplicates("condition_id").set_index("condition_id")[title_column]
    clean_15m = normalize(pd.read_parquet(args.input_dir / "candles_15min_clean.parquet"))
    clean_1h = normalize(pd.read_parquet(args.input_dir / "candles_1h_clean.parquet"))
    raw_15m = normalize(pd.read_parquet(args.input_dir / "candles_15min.parquet"))
    raw_1h = normalize(pd.read_parquet(args.input_dir / "candles_1h.parquet"))

    gaps_15m, contracts_15m, missing_15m = gap_tables(
        clean_15m,
        resolution="15m",
        interval_minutes=15,
        title_by_condition=title_by_condition,
    )
    gaps_1h, contracts_1h, missing_1h = gap_tables(
        clean_1h,
        resolution="1h",
        interval_minutes=60,
        title_by_condition=title_by_condition,
    )
    gaps = pd.concat([gaps_15m, gaps_1h], ignore_index=True)
    contracts = pd.concat([contracts_15m, contracts_1h], ignore_index=True)
    gap_distribution = distribution(gaps)
    summaries = [
        dataset_summary(
            clean_15m, contracts_15m, gaps_15m, resolution="15m", interval_minutes=15
        ),
        dataset_summary(clean_1h, contracts_1h, gaps_1h, resolution="1h", interval_minutes=60),
    ]
    raw_gaps_15m, raw_contracts_15m, _ = gap_tables(
        raw_15m,
        resolution="15m",
        interval_minutes=15,
        title_by_condition=title_by_condition,
    )
    raw_gaps_1h, raw_contracts_1h, _ = gap_tables(
        raw_1h,
        resolution="1h",
        interval_minutes=60,
        title_by_condition=title_by_condition,
    )
    raw_summaries = [
        dataset_summary(
            raw_15m,
            raw_contracts_15m,
            raw_gaps_15m,
            resolution="15m",
            interval_minutes=15,
        ),
        dataset_summary(
            raw_1h,
            raw_contracts_1h,
            raw_gaps_1h,
            resolution="1h",
            interval_minutes=60,
        ),
    ]
    cross = cross_resolution(missing_1h, missing_15m, clean_1h, clean_15m)
    raw_clean = {
        "15m_raw_rows": len(raw_15m),
        "15m_clean_rows": len(clean_15m),
        "15m_quarantined_rows": len(raw_15m) - len(clean_15m),
        "1h_raw_rows": len(raw_1h),
        "1h_clean_rows": len(clean_1h),
        "1h_quarantined_rows": len(raw_1h) - len(clean_1h),
    }

    gaps.to_parquet(args.output_dir / "gap_events.parquet", index=False)
    contracts.to_csv(args.output_dir / "contract_gap_summary.csv", index=False)
    gap_distribution.to_csv(args.output_dir / "gap_length_distribution.csv", index=False)
    missing_15m.to_parquet(args.output_dir / "missing_15m_slots.parquet", index=False)
    missing_1h.to_parquet(args.output_dir / "missing_1h_slots.parquet", index=False)
    report = render_report(summaries, raw_summaries, gap_distribution, cross, contracts)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "new FinData cohort only; clean native 15m and 1h internal gaps",
        "raw_clean_rows": raw_clean,
        "dataset_summaries": summaries,
        "raw_dataset_summaries": raw_summaries,
        "cross_resolution": cross,
        "endpoint_gaps_outside_first_last_observation_measured": False,
        "source_files_modified": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
