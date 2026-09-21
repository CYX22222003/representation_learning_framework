"""Plot clean, bounded-forward-filled FinData OHLCV by contract.

The source Parquet files are never modified.  Each contract-resolution plot
uses its actual first and last clean candle, complete isolated native gaps are
filled with a flat zero-volume candle, and longer gaps remain NaN so the
plotted lines break rather than implying continuous observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/representation_learning_framework_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data_new/findata/polymarket/historical_diverse_top50_2025-12-01_2026-08-31"
)
DEFAULT_OUTPUT = DEFAULT_INPUT / "analysis/native_ohlcv_contract_plots"
FEATURES = ("open", "high", "low", "close", "volume")
RESOLUTIONS = {
    "15m": ("candles_15min_clean.parquet", "candles_15min_clean_ffill1.parquet", 15),
    "1h": ("candles_1h_clean.parquet", "candles_1h_clean_ffill1.parquet", 60),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path, *, overwrite: bool) -> None:
    allowed = (ROOT / "data_new").resolve()
    resolved = path.resolve()
    if allowed != resolved and allowed not in resolved.parents:
        raise ValueError(f"output must be inside {allowed}: {resolved}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite occupied output: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_slug(value: object, *, fallback: str = "contract") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return (slug or fallback)[:96]


def load_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "condition_id",
        "selection_rank",
        "title",
        "slug_search",
        "requested_overlap_start",
        "requested_overlap_end",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")
    result = frame.copy()
    result["condition_id"] = result["condition_id"].astype(str)
    result["requested_overlap_start"] = pd.to_datetime(
        result["requested_overlap_start"], utc=True, errors="raise"
    )
    result["requested_overlap_end"] = pd.to_datetime(
        result["requested_overlap_end"], utc=True, errors="raise"
    )
    if result["condition_id"].duplicated().any():
        raise ValueError("metadata contains duplicate condition_id values")
    if (result["requested_overlap_start"] >= result["requested_overlap_end"]).any():
        raise ValueError("metadata contains an empty or reversed market interval")
    return result.sort_values("selection_rank").reset_index(drop=True)


def load_clean_candles(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"condition_id", "date", *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"clean candle file is missing required columns: {missing}")
    result = frame[["condition_id", "date", *FEATURES]].copy()
    result["condition_id"] = result["condition_id"].astype(str)
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
    for column in FEATURES:
        result[column] = pd.to_numeric(result[column], errors="raise")
    values = result[list(FEATURES)].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"clean candle file contains non-finite OHLCV: {path}")
    if (result["volume"] < 0).any():
        raise ValueError(f"clean candle file contains negative volume: {path}")
    invalid_ohlc = (
        (result["high"] < result[["open", "close", "low"]].max(axis=1))
        | (result["low"] > result[["open", "close", "high"]].min(axis=1))
    )
    if invalid_ohlc.any():
        raise ValueError(f"clean candle file contains inconsistent OHLC rows: {path}")
    result = result.sort_values(["condition_id", "date"]).reset_index(drop=True)
    if result.duplicated(["condition_id", "date"]).any():
        raise ValueError(f"clean candle file contains duplicate identities: {path}")
    return result


def load_filled_candles(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"condition_id", "date", *FEATURES, "is_observed", "is_imputed"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"filled candle file is missing required columns: {missing}")
    result = frame[["condition_id", "date", *FEATURES, "is_observed", "is_imputed"]].copy()
    result["condition_id"] = result["condition_id"].astype(str)
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
    result = result.sort_values(["condition_id", "date"], kind="stable").reset_index(drop=True)
    if result.duplicated(["condition_id", "date"]).any():
        raise ValueError(f"filled candle file contains duplicate identities: {path}")
    if (result["is_observed"].astype(bool) == result["is_imputed"].astype(bool)).any():
        raise ValueError(f"filled candle rows must be exactly observed or imputed: {path}")
    values = result[list(FEATURES)].to_numpy(np.float64)
    if not np.isfinite(values).all() or (result["volume"] < 0).any():
        raise ValueError(f"filled candle file contains invalid OHLCV: {path}")
    return result


def plotting_grid_from_filled(
    frame: pd.DataFrame,
    *,
    interval_minutes: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    observed = frame.loc[frame["is_observed"].astype(bool)]
    if observed.empty:
        raise ValueError("filled contract has no observed rows")
    start, end = observed["date"].iloc[0], observed["date"].iloc[-1]
    grid_index = pd.date_range(start, end, freq=f"{interval_minutes}min")
    grid = frame.set_index("date")[[*FEATURES, "is_observed", "is_imputed"]].reindex(grid_index)
    absent = grid["close"].isna()
    grid.loc[absent, ["is_observed", "is_imputed"]] = False
    grid[["is_observed", "is_imputed"]] = grid[["is_observed", "is_imputed"]].astype(bool)
    grid.index.name = "date"
    grid = grid.reset_index()
    return grid, {
        "internal_grid_rows": len(grid),
        "internal_missing_rows": int(absent.sum() + frame["is_imputed"].sum()),
        "imputed_rows": int(frame["is_imputed"].sum()),
        "unfilled_missing_rows": int(absent.sum()),
    }


def bounded_fill_ohlcv(
    frame: pd.DataFrame,
    *,
    interval_minutes: int,
    maximum_fill_bars: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a regular plotting grid with only bounded complete gaps filled."""

    if frame.empty:
        raise ValueError("cannot build a plotting grid from an empty contract")
    if interval_minutes <= 0 or maximum_fill_bars < 0:
        raise ValueError("interval_minutes must be positive and maximum_fill_bars non-negative")
    ordered = frame.sort_values("date").reset_index(drop=True)
    if ordered["date"].duplicated().any():
        raise ValueError("contract contains duplicate timestamps")
    step_ns = int(pd.Timedelta(minutes=interval_minutes).value)
    timestamps_ns = ordered["date"].astype("int64").to_numpy()
    if ((timestamps_ns - timestamps_ns[0]) % step_ns != 0).any():
        raise ValueError("contract timestamps do not share the declared native grid")

    grid_index = pd.date_range(
        ordered["date"].iloc[0],
        ordered["date"].iloc[-1],
        freq=f"{interval_minutes}min",
    )
    grid = ordered.set_index("date")[list(FEATURES)].reindex(grid_index)
    grid.index.name = "date"
    missing = grid["close"].isna().to_numpy()
    if not grid[list(FEATURES)].isna().all(axis=1).equals(grid["close"].isna()):
        raise ValueError("contract has partially missing OHLCV rows")

    is_imputed = np.zeros(len(grid), dtype=bool)
    filled_rows = 0
    index = 0
    while index < len(grid):
        if not missing[index]:
            index += 1
            continue
        end = index + 1
        while end < len(grid) and missing[end]:
            end += 1
        length = end - index
        complete_internal_gap = index > 0 and end < len(grid)
        if complete_internal_gap and length <= maximum_fill_bars:
            previous_close = float(grid.iloc[index - 1]["close"])
            grid.iloc[index:end, grid.columns.get_indexer(["open", "high", "low", "close"])] = (
                previous_close
            )
            grid.iloc[index:end, grid.columns.get_loc("volume")] = 0.0
            is_imputed[index:end] = True
            filled_rows += length
        index = end

    grid["is_imputed"] = is_imputed
    grid["is_observed"] = ~missing
    grid = grid.reset_index()
    total_missing = int(missing.sum())
    return grid, {
        "internal_grid_rows": len(grid),
        "internal_missing_rows": total_missing,
        "imputed_rows": filled_rows,
        "unfilled_missing_rows": total_missing - filled_rows,
    }


def _price_limits(grid: pd.DataFrame) -> tuple[float, float]:
    values = grid[["open", "high", "low", "close"]].to_numpy(np.float64)
    finite = values[np.isfinite(values)]
    low = float(finite.min())
    high = float(finite.max())
    padding = max((high - low) * 0.05, 0.01)
    return max(0.0, low - padding), min(1.0, high + padding)


def plot_contract(
    grid: pd.DataFrame,
    metadata: pd.Series,
    *,
    resolution: str,
    market_start: pd.Timestamp,
    market_end: pd.Timestamp,
    output_path: Path,
    counts: dict[str, int],
) -> None:
    colors = {
        "open": "#1f77b4",
        "high": "#2ca02c",
        "low": "#d62728",
        "close": "#9467bd",
        "volume": "#4c566a",
    }
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    price_limits = _price_limits(grid)
    imputed = grid["is_imputed"].to_numpy(bool)
    for axis, feature in zip(axes, FEATURES, strict=True):
        axis.plot(
            grid["date"],
            grid[feature],
            color=colors[feature],
            linewidth=0.65,
            label=feature.capitalize(),
        )
        if imputed.any():
            axis.scatter(
                grid.loc[imputed, "date"],
                grid.loc[imputed, feature],
                s=9,
                color="#ff8c00",
                marker="x",
                linewidths=0.7,
                label="Isolated-gap fill",
                zorder=3,
            )
        axis.set_ylabel(feature.capitalize())
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.margins(x=0)
        if feature != "volume":
            axis.set_ylim(*price_limits)
        axis.legend(loc="upper left", fontsize=8, framealpha=0.8)

    for axis in axes:
        axis.set_xlim(market_start, market_end)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("UTC time")

    title = textwrap.fill(str(metadata["title"]), width=105)
    interval_text = (
        f"{market_start.isoformat()} → {market_end.isoformat()} UTC"
        f"  |  observed={counts['observed_rows']:,}"
        f"  |  imputed={counts['imputed_rows']:,}"
        f"  |  unfilled missing slots={counts['unfilled_missing_rows']:,}"
    )
    fig.suptitle(
        f"Rank {int(metadata['selection_rank']):02d} · {resolution} · {title}\n{interval_text}",
        fontsize=12,
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        f"condition_id={metadata['condition_id']} · clean native candles · one-bar bounded forward fill",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.955))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def markdown_report(records: pd.DataFrame) -> str:
    lines = [
        "# Clean FinData Native OHLCV Contract Plots",
        "",
        "Each figure contains separate Open, High, Low, Close, and Volume panels. "
        "Sources are the separate post-quarantine bounded-fill native files. Complete isolated "
        "one-bar gaps are flat-filled from the previous observed close with zero volume and marked "
        "in orange; longer gaps remain visible breaks.",
        "",
        "Every x-axis is dynamically restricted to that contract-resolution's actual first "
        "and last clean candle. Metadata overlap bounds are retained in the manifest for "
        "reference, but they do not remove collected end-date rows.",
        "",
        "| Resolution | Contracts | Clean observed rows | Imputed rows | Unfilled missing slots | Rows at/after metadata overlap end |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for resolution, group in records.groupby("resolution", sort=False):
        lines.append(
            f"| {resolution} | {len(group):,} | "
            f"{int(group['observed_rows'].sum()):,} | "
            f"{int(group['imputed_rows'].sum()):,} | "
            f"{int(group['unfilled_missing_rows'].sum()):,} | "
            f"{int(group['rows_at_or_after_metadata_end'].sum()):,} |"
        )
    lines.extend(["", "## Plots", ""])
    for resolution, group in records.groupby("resolution", sort=False):
        lines.extend([f"### {resolution}", ""])
        for row in group.sort_values("selection_rank").itertuples(index=False):
            label = str(row.title).replace("[", "\\[").replace("]", "\\]")
            lines.append(f"- {int(row.selection_rank):02d}. [{label}]({row.plot_path})")
        lines.append("")
    return "\n".join(lines)


def generate_plots(
    input_dir: Path,
    output_dir: Path,
    *,
    maximum_fill_bars: int,
    overwrite: bool,
) -> pd.DataFrame:
    if maximum_fill_bars != 1:
        raise ValueError("the frozen exploratory plotting contract requires exactly one fill bar")
    metadata_path = input_dir / "market_metadata.parquet"
    metadata = load_metadata(metadata_path)
    prepare_output(output_dir, overwrite=overwrite)
    records: list[dict[str, object]] = []
    source_hashes: dict[str, str] = {}

    for resolution, (clean_filename, filled_filename, interval_minutes) in RESOLUTIONS.items():
        clean_path = input_dir / clean_filename
        source_path = input_dir / filled_filename
        source_hashes[clean_filename] = sha256_file(clean_path)
        source_hashes[filled_filename] = sha256_file(source_path)
        clean = load_clean_candles(clean_path)
        candles = load_filled_candles(source_path)
        replayed_clean = candles.loc[candles["is_observed"], ["condition_id", "date", *FEATURES]]
        if not replayed_clean.reset_index(drop=True).equals(clean.reset_index(drop=True)):
            raise AssertionError(f"{filled_filename} observed rows do not replay {clean_filename}")
        unknown = sorted(set(candles["condition_id"]) - set(metadata["condition_id"]))
        if unknown:
            raise ValueError(f"{filled_filename} contains condition IDs missing from metadata: {unknown[:3]}")
        by_condition = {key: group for key, group in candles.groupby("condition_id", sort=False)}
        for row in metadata.itertuples(index=False):
            condition_id = str(row.condition_id)
            if condition_id not in by_condition:
                raise ValueError(f"{filled_filename} has no rows for condition {condition_id}")
            contract = by_condition[condition_id].reset_index(drop=True)
            observed_contract = contract.loc[contract["is_observed"]].reset_index(drop=True)
            metadata_start = pd.Timestamp(row.requested_overlap_start)
            metadata_end = pd.Timestamp(row.requested_overlap_end)
            plot_start = observed_contract["date"].iloc[0]
            plot_end = observed_contract["date"].iloc[-1]
            rows_before_metadata_start = int(observed_contract["date"].lt(metadata_start).sum())
            rows_at_or_after_metadata_end = int(observed_contract["date"].ge(metadata_end).sum())
            grid, gap_counts = plotting_grid_from_filled(
                contract,
                interval_minutes=interval_minutes,
            )
            rank = int(row.selection_rank)
            slug = safe_slug(row.slug_search, fallback=condition_id[:12])
            relative_path = Path(resolution) / f"{rank:02d}_{slug}_{condition_id[2:10]}.png"
            counts = {
                **gap_counts,
                "observed_rows": len(observed_contract),
            }
            plot_contract(
                grid,
                pd.Series(row._asdict()),
                resolution=resolution,
                market_start=plot_start,
                market_end=plot_end,
                output_path=output_dir / relative_path,
                counts=counts,
            )
            record = {
                "resolution": resolution,
                "interval_minutes": interval_minutes,
                "selection_rank": rank,
                "condition_id": condition_id,
                "title": str(row.title),
                "metadata_overlap_start": metadata_start,
                "metadata_overlap_end": metadata_end,
                "plot_start": plot_start,
                "plot_end": plot_end,
                "observed_rows": len(observed_contract),
                "rows_before_metadata_start": rows_before_metadata_start,
                "rows_at_or_after_metadata_end": rows_at_or_after_metadata_end,
                **gap_counts,
                "plot_path": relative_path.as_posix(),
                "plot_sha256": sha256_file(output_dir / relative_path),
            }
            records.append(record)
            print(
                f"[{resolution}] {rank:02d}/{len(metadata):02d} {row.title} "
                f"(observed={len(observed_contract):,}, filled={gap_counts['imputed_rows']:,})",
                flush=True,
            )

    manifest_frame = pd.DataFrame(records).sort_values(
        ["resolution", "selection_rank"], kind="stable"
    )
    manifest_frame.to_csv(output_dir / "plot_manifest.csv", index=False)
    report = markdown_report(manifest_frame)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "per-contract OHLCV visualization of clean plus bounded-fill native data",
        "input_dir": str(input_dir.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "source_sha256": source_hashes,
        "plot_period": "actual first clean candle through actual last clean candle",
        "metadata_overlap_bounds": (
            "retained for audit only; no collected row is excluded from a plot by these fields"
        ),
        "bounded_fill_policy": {
            "maximum_missing_native_bars": maximum_fill_bars,
            "fill_ohlc": "previous observed close for open/high/low/close",
            "fill_volume": 0.0,
            "longer_gaps": "left missing and rendered as line breaks",
            "leading_or_trailing_period_gaps": "not filled",
            "source_files_modified": False,
        },
        "plots": json.loads(manifest_frame.to_json(orient="records", date_format="iso")),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-fill-bars", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    frame = generate_plots(
        args.input_dir,
        args.output_dir,
        maximum_fill_bars=args.maximum_fill_bars,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {len(frame):,} contract-resolution figures to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
