"""Analyze completeness and movement content of a collected FinData cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_new/findata/polymarket/mvp_2025-12-01_2026-08-31"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, record in manifest["artifacts"].items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"manifest artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path}")
    return manifest


def interval_quality(frame: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    expected_delta = pd.Timedelta(minutes=interval_minutes)
    for condition_id, market in frame.groupby("condition_id", sort=True):
        market = market.sort_values("date")
        dates = market["date"]
        gaps = dates.diff().dropna()
        expected_rows = int((dates.iloc[-1] - dates.iloc[0]) / expected_delta) + 1
        ohlcv = market[["open", "high", "low", "close", "volume"]].to_numpy(np.float64)
        records.append(
            {
                "condition_id": condition_id,
                "interval_minutes": interval_minutes,
                "rows": len(market),
                "start": dates.iloc[0],
                "end": dates.iloc[-1],
                "expected_grid_rows": expected_rows,
                "observed_grid_fraction": len(market) / expected_rows,
                "exact_interval_gap_fraction": float((gaps == expected_delta).mean()) if len(gaps) else math.nan,
                "median_gap_minutes": float(gaps.median() / pd.Timedelta(minutes=1)) if len(gaps) else math.nan,
                "p95_gap_minutes": float(gaps.quantile(0.95) / pd.Timedelta(minutes=1)) if len(gaps) else math.nan,
                "max_gap_hours": float(gaps.max() / pd.Timedelta(hours=1)) if len(gaps) else math.nan,
                "finite_ohlcv": bool(np.isfinite(ohlcv).all()),
                "positive_volume_fraction": float((market["volume"] > 0).mean()),
                "unchanged_close_fraction": float(market["close"].diff().eq(0).mean()),
            }
        )
    return pd.DataFrame(records)


def consecutive_four_hour_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_records: list[dict[str, object]] = []
    eligible_parts: list[pd.DataFrame] = []
    four_hours = pd.Timedelta(hours=4)
    for condition_id, market in frame.groupby("condition_id", sort=True):
        market = market.sort_values("date").reset_index(drop=True)
        dates = market["date"]
        close = market["close"].to_numpy(np.float64)
        complete = market["complete_1h_coverage"].to_numpy(bool)
        eligible = np.zeros(len(market), dtype=bool)
        if len(market) > 2:
            eligible[:-2] = (
                (dates.shift(-1) - dates == four_hours)
                & (dates.shift(-2) - dates == 2 * four_hours)
                & pd.Series(complete).rolling(3).sum().shift(-2).eq(3)
            ).iloc[:-2].to_numpy(bool)
        indices = np.flatnonzero(eligible)
        deltas = close[indices + 2] - close[indices]

        context_eligible = 0
        if len(market) >= 66:
            expected_span = 65 * four_hours
            for target_index in range(65, len(market)):
                input_start = target_index - 65
                decision_index = target_index - 2
                block = market.iloc[input_start : target_index + 1]
                if (
                    block["complete_1h_coverage"].all()
                    and block["date"].iloc[-1] - block["date"].iloc[0] == expected_span
                    and decision_index - input_start + 1 == 64
                ):
                    context_eligible += 1

        market_records.append(
            {
                "condition_id": condition_id,
                "four_hour_rows": len(market),
                "complete_four_hour_rows": int(complete.sum()),
                "complete_four_hour_fraction": float(complete.mean()),
                "consecutive_h2_targets": len(deltas),
                "seq64_h2_rows_without_imputation": context_eligible,
                "exact_zero_fraction": float((deltas == 0).mean()) if len(deltas) else math.nan,
                "stable_fraction_tau005": float((np.abs(deltas) <= 0.005).mean()) if len(deltas) else math.nan,
                "mean_absolute_movement": float(np.abs(deltas).mean()) if len(deltas) else math.nan,
                "movement_std": float(deltas.std()) if len(deltas) else math.nan,
                "zero_baseline_mae": float(np.abs(deltas).mean()) if len(deltas) else math.nan,
                "zero_baseline_rmse": float(np.sqrt(np.mean(np.square(deltas)))) if len(deltas) else math.nan,
            }
        )
        if len(indices):
            eligible_parts.append(
                pd.DataFrame(
                    {
                        "condition_id": condition_id,
                        "decision_date": dates.iloc[indices].to_numpy(),
                        "target_date": dates.iloc[indices + 2].to_numpy(),
                        "current_close": close[indices],
                        "future_close": close[indices + 2],
                        "delta": deltas,
                    }
                )
            )
    eligible_rows = pd.concat(eligible_parts, ignore_index=True) if eligible_parts else pd.DataFrame()
    return pd.DataFrame(market_records), eligible_rows


def native_eight_hour_analysis(
    frame: pd.DataFrame,
    *,
    resolution: str,
    interval_minutes: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    steps = 8 * 60 // interval_minutes
    expected_span = pd.Timedelta(hours=8)
    records: list[dict[str, object]] = []
    pooled: list[np.ndarray] = []
    for condition_id, market in frame.groupby("condition_id", sort=True):
        market = market.sort_values("date").reset_index(drop=True)
        eligible = market["date"].shift(-steps).sub(market["date"]).eq(expected_span)
        indices = np.flatnonzero(eligible.fillna(False).to_numpy())
        close = market["close"].to_numpy(np.float64)
        delta = close[indices + steps] - close[indices]
        if len(delta):
            pooled.append(delta)
        records.append(
            {
                "resolution": resolution,
                "condition_id": condition_id,
                "targets": len(delta),
                "exact_zero_fraction": float((delta == 0).mean()) if len(delta) else math.nan,
                "stable_fraction_tau005": (
                    float((np.abs(delta) <= 0.005).mean()) if len(delta) else math.nan
                ),
                "mean_absolute_movement": float(np.abs(delta).mean()) if len(delta) else math.nan,
            }
        )
    by_market = pd.DataFrame(records)
    delta = np.concatenate(pooled) if pooled else np.empty(0, dtype=np.float64)
    summary = {
        "resolution": resolution,
        "markets": int(by_market["targets"].gt(0).sum()),
        "targets": len(delta),
        "exact_zero_fraction": float((delta == 0).mean()) if len(delta) else None,
        "stable_fraction_tau005": float((np.abs(delta) <= 0.005).mean()) if len(delta) else None,
        "mean_absolute_movement": float(np.abs(delta).mean()) if len(delta) else None,
        "contract_macro_exact_zero_fraction": float(by_market["exact_zero_fraction"].mean()),
        "contract_macro_stable_fraction_tau005": float(by_market["stable_fraction_tau005"].mean()),
        "contract_macro_mean_absolute_movement": float(by_market["mean_absolute_movement"].mean()),
    }
    return by_market, summary


def aligned_hourly_consistency(candles_15m: pd.DataFrame, candles_1h: pd.DataFrame) -> dict[str, object]:
    parts: list[pd.DataFrame] = []
    for condition_id, market in candles_15m.groupby("condition_id", sort=True):
        market = market.sort_values("date").copy()
        market["hour"] = market["date"].dt.floor("1h")
        aggregate = market.groupby("hour", sort=True).agg(
            open_15m=("open", "first"),
            high_15m=("high", "max"),
            low_15m=("low", "min"),
            close_15m=("close", "last"),
            volume_15m=("volume", "sum"),
            observed_15m_bars=("date", "nunique"),
        ).reset_index(names="date")
        aggregate.insert(0, "condition_id", condition_id)
        parts.append(aggregate)
    derived = pd.concat(parts, ignore_index=True)
    native = candles_1h.rename(
        columns={
            "open": "open_1h",
            "high": "high_1h",
            "low": "low_1h",
            "close": "close_1h",
            "volume": "volume_1h",
        }
    )
    aligned = derived.merge(native, on=["condition_id", "date"], how="inner")
    complete = aligned.loc[aligned["observed_15m_bars"].eq(4)].copy()
    if complete.empty:
        return {"aligned_complete_hours": 0}
    close_diff = np.abs(complete["close_15m"] - complete["close_1h"])
    volume_diff = np.abs(complete["volume_15m"] - complete["volume_1h"])
    return {
        "aligned_complete_hours": len(complete),
        "exact_close_match_fraction": float((close_diff <= 1e-12).mean()),
        "close_mae": float(close_diff.mean()),
        "close_p99_absolute_difference": float(close_diff.quantile(0.99)),
        "volume_mae": float(volume_diff.mean()),
    }


def category_movement_summary(
    eligible: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "category",
        "markets",
        "targets",
        "exact_zero_fraction",
        "stable_fraction_tau005",
        "mean_absolute_movement",
        "movement_std",
    ]
    if eligible.empty or "category" not in metadata.columns:
        return pd.DataFrame(columns=columns)
    merged = eligible.merge(
        metadata[["condition_id", "category"]].drop_duplicates("condition_id"),
        on="condition_id",
        how="left",
        validate="many_to_one",
    )
    records: list[dict[str, object]] = []
    for category, group in merged.groupby("category", dropna=False, sort=True):
        delta = group["delta"].to_numpy(np.float64)
        records.append(
            {
                "category": category,
                "markets": int(group["condition_id"].nunique()),
                "targets": len(group),
                "exact_zero_fraction": float((delta == 0).mean()),
                "stable_fraction_tau005": float((np.abs(delta) <= 0.005).mean()),
                "mean_absolute_movement": float(np.abs(delta).mean()),
                "movement_std": float(delta.std()),
            }
        )
    return pd.DataFrame(records, columns=columns)


def outcome_orientation_diagnostic(frame: pd.DataFrame) -> dict[str, object]:
    ordered = frame.sort_values(["condition_id", "date"]).copy()
    previous = ordered.groupby("condition_id")["close"].shift()
    jump = (ordered["close"] - previous).abs()
    complement_error = (ordered["close"] + previous - 1.0).abs()
    likely_flip = jump.gt(0.5) & complement_error.le(0.02)
    wide_bar = (ordered["high"] - ordered["low"]).gt(0.5)
    affected = ordered.loc[likely_flip | wide_bar, "condition_id"].nunique()
    return {
        "likely_complement_close_flips": int(likely_flip.sum()),
        "wide_complement_spanning_bars": int(wide_bar.sum()),
        "affected_markets": int(affected),
        "warning": (
            "A positive diagnostic demonstrates mixed outcome orientation; a zero count would not prove "
            "that a condition-level series is token-consistent."
        ),
    }


def render_report(
    manifest: dict[str, object],
    metadata: pd.DataFrame,
    quality: pd.DataFrame,
    movement: pd.DataFrame,
    eligible: pd.DataFrame,
    categories: pd.DataFrame,
    native_movement: list[dict[str, object]],
    orientation: dict[str, object],
    consistency: dict[str, object],
) -> str:
    complete_4h = int(movement["complete_four_hour_rows"].sum())
    all_4h = int(movement["four_hour_rows"].sum())
    deltas = eligible["delta"].to_numpy(np.float64)
    lines = [
        "# FinData Recent Polymarket Cohort Audit",
        "",
        f"Requested interval: `{manifest['requested_interval']['start_inclusive']}` to "
        f"`{manifest['requested_interval']['end_exclusive']}` (half-open, UTC).",
        "",
        "## Scope",
        "",
        f"The cohort contains {len(metadata)} Polymarket conditions whose candle eligibility was "
        "probed inside the manifest's selection interval. Catalog volume and complete market metadata "
        "remain retrospective inputs, so this is an exploratory study cohort rather than a production "
        "cutoff-local universe.",
        "",
        "Selected markets:",
        "",
    ]
    rank_column = "selection_rank" if "selection_rank" in metadata.columns else "mvp_rank"
    for row in metadata.sort_values(rank_column).itertuples(index=False):
        lines.append(f"- `{row.condition_id}` — {row.question_snapshot}")
    lines.extend(
        [
            "",
            "## Acquisition findings",
            "",
            f"- Native 15-minute rows: {manifest['intervals']['15m']['rows']:,}.",
            f"- Native one-hour rows: {manifest['intervals']['1h']['rows']:,}.",
            f"- Derived four-hour bins: {manifest['intervals']['4h']['rows']:,}.",
            f"- Fully observed four-hour bins: {complete_4h:,}/{all_4h:,} ({complete_4h / all_4h:.2%}).",
            "- The native endpoints are sparse: absent intervals are not emitted as zero-volume candles. "
            "The stored raw files deliberately do not forward-fill them.",
            "- The documented native `4h` query was not usable in live probes, so four-hour rows are "
            "UTC-aligned aggregates of native one-hour rows and carry explicit completeness fields.",
            "",
            "## Cross-resolution check",
            "",
            f"On {consistency.get('aligned_complete_hours', 0):,} hours containing four observed 15-minute bars, "
            f"the 15-minute-derived and native-hourly close match rate is "
            f"{consistency.get('exact_close_match_fraction', math.nan):.2%}, with close MAE "
            f"{consistency.get('close_mae', math.nan):.6g}. This should be retained as a source-consistency "
            "diagnostic rather than assuming different API resolutions are identical.",
            "",
            "## Outcome-orientation audit",
            "",
            f"The native hourly series contains {orientation['likely_complement_close_flips']:,} likely "
            f"complementary close flips and {orientation['wide_complement_spanning_bars']:,} bars whose "
            f"high-low range exceeds 0.5, affecting {orientation['affected_markets']} markets. FinData's "
            "condition-level candle rows do not identify the outcome token. These files therefore cannot be "
            "treated as a canonical YES-probability history; the movement statistics below are retained only "
            "as contaminated source diagnostics.",
            "",
            "## Four-hour movement feasibility",
            "",
            f"Strictly consecutive, fully observed eight-hour targets: {len(eligible):,}.",
            f"Exact-zero share: {float((deltas == 0).mean()) if len(deltas) else math.nan:.2%}.",
            f"Stable share at `|delta| <= 0.005`: "
            f"{float((np.abs(deltas) <= 0.005).mean()) if len(deltas) else math.nan:.2%}.",
            f"Zero-movement baseline MAE: {float(np.abs(deltas).mean()) if len(deltas) else math.nan:.6f}.",
            f"Rows with a complete consecutive 64-bar context and two-bar target, without imputation: "
            f"{int(movement['seq64_h2_rows_without_imputation'].sum()):,}.",
            "",
            "Contract-macro means (each eligible contract has equal weight):",
            "",
            f"- Exact-zero share: {movement['exact_zero_fraction'].dropna().mean():.2%}.",
            f"- Stable share at `|delta| <= 0.005`: "
            f"{movement['stable_fraction_tau005'].dropna().mean():.2%}.",
            f"- Mean absolute movement: {movement['mean_absolute_movement'].dropna().mean():.6f}.",
            "",
        ]
    )
    if not categories.empty:
        lines.extend(
            [
                "## Heuristic category diagnostics",
                "",
                "Categories are approximate sampling aids, not ground-truth research labels. The movement "
                "statistics below remain descriptive and row-weighted within each category.",
                "",
                "| Category | Markets | Targets | Exact zero | Stable (0.005) | Mean absolute move |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in categories.itertuples(index=False):
            lines.append(
                f"| {row.category} | {row.markets} | {row.targets:,} | "
                f"{row.exact_zero_fraction:.2%} | {row.stable_fraction_tau005:.2%} | "
                f"{row.mean_absolute_movement:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Native-resolution eight-hour check",
            "",
            "Only uninterrupted native bars are included. These are diagnostics of the downloaded source, "
            "not interchangeable training datasets.",
            "",
            "| Resolution | Markets | Targets | Exact zero | Stable (0.005) | Mean absolute move |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in native_movement:
        lines.append(
            f"| {row['resolution']} | {row['markets']} | {row['targets']:,} | "
            f"{row['exact_zero_fraction']:.2%} | {row['stable_fraction_tau005']:.2%} | "
            f"{row['mean_absolute_movement']:.6f} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "The condition-level candles are not verified canonical YES-probability histories because outcome-token "
            "identity is absent and sometimes mixed. The project accepts them only as an explicitly exploratory "
            "Phase 5 source after the approved quarantine, causal bounded-gap treatment, and source-limitation "
            "reporting. This raw audit does not clear training: walk-local decision availability, activity and target "
            "eligibility, identical baseline rows, and replayable manifests still have to be enforced. Rows from "
            "related contracts are not independent market regimes, so pooled row counts must not be treated as "
            "independent evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    manifest = verify_manifest(args.input_dir)
    metadata = pd.read_parquet(args.input_dir / "market_metadata.parquet")
    candles_15m = pd.read_parquet(args.input_dir / "candles_15min.parquet")
    candles_1h = pd.read_parquet(args.input_dir / "candles_1h.parquet")
    candles_4h = pd.read_parquet(args.input_dir / "candles_4h_derived.parquet")

    quality = pd.concat(
        [interval_quality(candles_15m, 15), interval_quality(candles_1h, 60)],
        ignore_index=True,
    )
    movement, eligible = consecutive_four_hour_analysis(candles_4h)
    categories = category_movement_summary(eligible, metadata)
    movement_15m, summary_15m = native_eight_hour_analysis(
        candles_15m, resolution="15m", interval_minutes=15
    )
    movement_1h, summary_1h = native_eight_hour_analysis(
        candles_1h, resolution="1h", interval_minutes=60
    )
    native_movement = [summary_15m, summary_1h]
    orientation = outcome_orientation_diagnostic(candles_1h)
    consistency = aligned_hourly_consistency(candles_15m, candles_1h)

    analysis_dir = args.input_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    quality.to_parquet(analysis_dir / "interval_quality.parquet", index=False)
    movement.to_parquet(analysis_dir / "four_hour_movement_by_market.parquet", index=False)
    eligible.to_parquet(analysis_dir / "eligible_four_hour_targets.parquet", index=False)
    categories.to_parquet(analysis_dir / "four_hour_movement_by_category.parquet", index=False)
    pd.concat([movement_15m, movement_1h], ignore_index=True).to_parquet(
        analysis_dir / "native_eight_hour_movement_by_market.parquet", index=False
    )

    deltas = eligible["delta"].to_numpy(np.float64)
    summary = {
        "markets": len(metadata),
        "interval_quality": quality.to_dict(orient="records"),
        "four_hour_by_market": movement.to_dict(orient="records"),
        "cross_resolution": consistency,
        "outcome_orientation": orientation,
        "four_hour_pooled": {
            "eligible_consecutive_complete_h2_targets": len(eligible),
            "seq64_h2_rows_without_imputation": int(movement["seq64_h2_rows_without_imputation"].sum()),
            "exact_zero_fraction": float((deltas == 0).mean()) if len(deltas) else None,
            "stable_fraction_tau005": float((np.abs(deltas) <= 0.005).mean()) if len(deltas) else None,
            "mean_absolute_movement": float(np.abs(deltas).mean()) if len(deltas) else None,
            "movement_std": float(deltas.std()) if len(deltas) else None,
            "zero_baseline_rmse": float(np.sqrt(np.mean(np.square(deltas)))) if len(deltas) else None,
        },
        "four_hour_contract_macro": {
            "eligible_markets": int(movement["consecutive_h2_targets"].gt(0).sum()),
            "exact_zero_fraction": float(movement["exact_zero_fraction"].dropna().mean()),
            "stable_fraction_tau005": float(movement["stable_fraction_tau005"].dropna().mean()),
            "mean_absolute_movement": float(movement["mean_absolute_movement"].dropna().mean()),
        },
        "four_hour_by_category": categories.to_dict(orient="records"),
        "native_eight_hour_movement": native_movement,
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    report = render_report(
        manifest,
        metadata,
        quality,
        movement,
        eligible,
        categories,
        native_movement,
        orientation,
        consistency,
    )
    (analysis_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
