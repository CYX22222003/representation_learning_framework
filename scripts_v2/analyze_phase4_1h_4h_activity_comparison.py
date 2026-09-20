"""Compare duration-matched one-hour and four-hour top-50/top-80 activity.

Exploratory data analysis only. No model is trained or selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/representation_learning_framework_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data/phase3/processed/market_4h_seq64_top50.npz.manifest.json"
WALK_SOURCE = ROOT / "experiments/phase4/data_analysis/top80_1h_timestamp_capacity/walk_capacity.csv"
DEFAULT_OUTPUT = ROOT / "experiments/phase4/data_analysis/top50_top80_1h_4h_activity_comparison"
THRESHOLD = 0.005
CONFIGS = {
    "1h": {"bar_hours": 1, "sequence_length": 256, "horizon": 8, "activity_bars": 24},
    "4h": {"bar_hours": 4, "sequence_length": 64, "horizon": 2, "activity_bars": 6},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path, overwrite: bool) -> None:
    resolved = path.resolve()
    required = (ROOT / "experiments/phase4/data_analysis").resolve()
    if required not in resolved.parents:
        raise ValueError(f"output must be below {required}: {resolved}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite occupied output: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ranked_contracts() -> list[dict[str, object]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    records = [
        row for row in manifest["universe_selection"]["candidate_records"]
        if row.get("eligible") and row.get("rank") is not None and int(row["rank"]) <= 80
    ]
    records.sort(key=lambda row: int(row["rank"]))
    if [int(row["rank"]) for row in records] != list(range(1, 81)):
        raise ValueError("source manifest does not contain contiguous ranks 1--80")
    return records


def longest_run(values: np.ndarray) -> int:
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    return int(np.diff(np.r_[starts, len(values)]).max())


def trailing_run(values: np.ndarray) -> int:
    changes = np.flatnonzero(values[1:] != values[:-1])
    return len(values) if len(changes) == 0 else int(len(values) - changes[-1] - 1)


def rolling_any(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values.astype(np.int8)).rolling(window, min_periods=1).sum().to_numpy() > 0
    )


def load_timeframe(timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = CONFIGS[timeframe]
    contract_records: list[dict[str, object]] = []
    row_parts: list[pd.DataFrame] = []
    for source in ranked_contracts():
        filename = str(source["filename"]).replace("-4h.feather", f"-{timeframe}.feather")
        path = ROOT / "data" / filename
        frame = pd.read_feather(path)
        timestamps = pd.to_datetime(frame["date"], errors="raise")
        close = pd.to_numeric(frame["close"], errors="raise").to_numpy(np.float64)
        volume = pd.to_numeric(frame["volume"], errors="raise").to_numpy(np.float64)
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise ValueError(f"invalid timestamps: {path}")
        if not np.isfinite(close).all() or not np.isfinite(volume).all():
            raise ValueError(f"non-finite OHLCV: {path}")

        seq_len = int(cfg["sequence_length"])
        horizon = int(cfg["horizon"])
        decisions = np.arange(seq_len - 1, len(close) - horizon, dtype=np.int64)
        if not len(decisions):
            raise ValueError(f"insufficient rows for duration-matched task: {path}")
        targets = decisions + horizon
        delta = close[targets] - close[decisions]
        changed = np.r_[False, np.diff(close) != 0]
        recent_change = rolling_any(changed, int(cfg["activity_bars"]))
        recent_volume = rolling_any(volume > 0, int(cfg["activity_bars"]))
        tail = trailing_run(close)
        tail_start = len(close) - tail

        contract_records.append({
            "timeframe": timeframe,
            "contract_rank": int(source["rank"]),
            "filename": filename,
            "raw_rows": len(close),
            "lifespan_hours": float((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600),
            "model_eligible_rows": len(delta),
            "close_std": float(close.std()),
            "close_range": float(close.max() - close.min()),
            "near_boundary_fraction": float(((close <= 0.05) | (close >= 0.95)).mean()),
            "bar_zero_move_fraction": float((np.diff(close) == 0).mean()),
            "eight_hour_zero_move_fraction": float((delta == 0).mean()),
            "eight_hour_stable_fraction": float((np.abs(delta) <= THRESHOLD).mean()),
            "eight_hour_mean_abs_move": float(np.abs(delta).mean()),
            "eight_hour_median_abs_move": float(np.median(np.abs(delta))),
            "eight_hour_p90_abs_move": float(np.quantile(np.abs(delta), 0.90)),
            "eight_hour_meaningful_005_fraction": float((np.abs(delta) > THRESHOLD).mean()),
            "positive_volume_bar_fraction": float((volume > 0).mean()),
            "longest_constant_close_run_hours": longest_run(close) * int(cfg["bar_hours"]),
            "trailing_constant_close_run_hours": tail * int(cfg["bar_hours"]),
            "trailing_constant_close_fraction": float(tail / len(close)),
            "trailing_positive_volume_fraction": float((volume[tail_start:] > 0).mean()),
            "source_sha256": sha256_file(path),
        })
        row_parts.append(pd.DataFrame({
            "timeframe": timeframe,
            "contract_rank": int(source["rank"]),
            "window_start_time": timestamps.iloc[decisions - seq_len + 1].to_numpy(),
            "decision_time": (
                timestamps.iloc[decisions] + pd.Timedelta(hours=int(cfg["bar_hours"]))
            ).to_numpy(),
            "target_time": (
                timestamps.iloc[targets] + pd.Timedelta(hours=int(cfg["bar_hours"]))
            ).to_numpy(),
            "relative_position": decisions / (len(close) - 1),
            "current_close": close[decisions],
            "delta": delta,
            "abs_delta": np.abs(delta),
            "recent_price_change_24h": recent_change[decisions],
            "recent_positive_volume_24h": recent_volume[decisions],
        }))
    return pd.DataFrame(contract_records), pd.concat(row_parts, ignore_index=True)


def aggregate_summary(contracts: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for timeframe in ("1h", "4h"):
        for top_k in (50, 80):
            c = contracts[(contracts["timeframe"] == timeframe) & (contracts["contract_rank"] <= top_k)]
            r = rows[(rows["timeframe"] == timeframe) & (rows["contract_rank"] <= top_k)]
            records.append({
                "timeframe": timeframe,
                "top_k": top_k,
                "contracts": len(c),
                "raw_rows": int(c["raw_rows"].sum()),
                "model_eligible_rows": len(r),
                "minimum_lifespan_hours": float(c["lifespan_hours"].min()),
                "median_lifespan_hours": float(c["lifespan_hours"].median()),
                "median_contract_close_std": float(c["close_std"].median()),
                "median_contract_meaningful_fraction": float(
                    c["eight_hour_meaningful_005_fraction"].median()
                ),
                "pooled_zero_move_fraction": float((r["delta"] == 0).mean()),
                "pooled_stable_fraction": float((r["abs_delta"] <= THRESHOLD).mean()),
                "pooled_mean_absolute_move": float(r["abs_delta"].mean()),
                "pooled_median_absolute_move": float(r["abs_delta"].median()),
                "pooled_p90_absolute_move": float(r["abs_delta"].quantile(0.90)),
                "contracts_meaningful_lt_10pct": int(
                    (c["eight_hour_meaningful_005_fraction"] < 0.10).sum()
                ),
                "contracts_close_std_lt_001": int((c["close_std"] < 0.01).sum()),
                "contracts_flat_run_ge_24h": int(
                    (c["longest_constant_close_run_hours"] >= 24).sum()
                ),
                "contracts_flat_run_ge_72h": int(
                    (c["longest_constant_close_run_hours"] >= 72).sum()
                ),
                "contracts_trailing_flat_ge_25pct": int(
                    (c["trailing_constant_close_fraction"] >= 0.25).sum()
                ),
            })
    return pd.DataFrame(records)


def eligibility_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for timeframe in ("1h", "4h"):
        for top_k in (50, 80):
            base = rows[(rows["timeframe"] == timeframe) & (rows["contract_rank"] <= top_k)]
            policies = {
                "all_rows": pd.Series(True, index=base.index),
                "recent_positive_volume_24h": base["recent_positive_volume_24h"],
                "recent_price_change_24h": base["recent_price_change_24h"],
                "recent_change_and_volume_24h": (
                    base["recent_price_change_24h"] & base["recent_positive_volume_24h"]
                ),
            }
            for name, eligible in policies.items():
                part = base[eligible]
                records.append({
                    "timeframe": timeframe,
                    "top_k": top_k,
                    "eligibility": name,
                    "rows": len(part),
                    "retained_fraction": float(len(part) / len(base)),
                    "contracts": part["contract_rank"].nunique(),
                    "zero_move_fraction": float((part["delta"] == 0).mean()),
                    "stable_fraction": float((part["abs_delta"] <= THRESHOLD).mean()),
                    "mean_absolute_move": float(part["abs_delta"].mean()),
                    "median_absolute_move": float(part["abs_delta"].median()),
                    "p90_absolute_move": float(part["abs_delta"].quantile(0.90)),
                })
    return pd.DataFrame(records)


def lifecycle_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for timeframe in ("1h", "4h"):
        for top_k in (50, 80):
            base = rows[(rows["timeframe"] == timeframe) & (rows["contract_rank"] <= top_k)].copy()
            base["lifecycle_stage"] = pd.cut(
                base["relative_position"], [0, 1 / 3, 2 / 3, 1],
                labels=["early", "middle", "late"], include_lowest=True, right=False,
            )
            for stage, part in base.groupby("lifecycle_stage", observed=True):
                records.append({
                    "timeframe": timeframe,
                    "top_k": top_k,
                    "lifecycle_stage": str(stage),
                    "rows": len(part),
                    "contracts": part["contract_rank"].nunique(),
                    "zero_move_fraction": float((part["delta"] == 0).mean()),
                    "stable_fraction": float((part["abs_delta"] <= THRESHOLD).mean()),
                    "mean_absolute_move": float(part["abs_delta"].mean()),
                    "p90_absolute_move": float(part["abs_delta"].quantile(0.90)),
                    "near_boundary_fraction": float(
                        ((part["current_close"] <= 0.05) | (part["current_close"] >= 0.95)).mean()
                    ),
                })
    return pd.DataFrame(records)


def aligned_target_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for top_k in (50, 80):
        hourly = rows[(rows["timeframe"] == "1h") & (rows["contract_rank"] <= top_k)][[
            "contract_rank", "decision_time", "target_time", "delta",
            "recent_price_change_24h",
        ]]
        four_hour = rows[(rows["timeframe"] == "4h") & (rows["contract_rank"] <= top_k)][[
            "contract_rank", "decision_time", "target_time", "delta",
            "recent_price_change_24h",
        ]]
        paired = four_hour.merge(
            hourly,
            on=["contract_rank", "decision_time", "target_time"],
            suffixes=("_4h", "_1h"),
            validate="one_to_one",
        )
        difference = paired["delta_1h"] - paired["delta_4h"]
        records.append({
            "top_k": top_k,
            "aligned_targets": len(paired),
            "fraction_of_4h_targets_aligned": float(len(paired) / len(four_hour)),
            "exact_delta_agreement": float(np.isclose(difference, 0, rtol=0, atol=1e-12).mean()),
            "delta_mean_absolute_difference": float(np.abs(difference).mean()),
            "delta_correlation": float(paired["delta_1h"].corr(paired["delta_4h"])),
            "activity_rule_agreement": float(
                (paired["recent_price_change_24h_1h"] == paired["recent_price_change_24h_4h"]).mean()
            ),
        })
    return pd.DataFrame(records)


def calendar_summary(rows: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(WALK_SOURCE)
    walks = source[
        (source["walk_count"] == 2)
        & (source["scheme"] == "calendar_window")
        & (source["sequence_length"] == 256)
    ].sort_values("walk")
    if len(walks) != 2:
        raise ValueError("expected two canonical calendar-window records")
    records = []
    for timeframe in ("1h", "4h"):
        for top_k in (50, 80):
            base = rows[(rows["timeframe"] == timeframe) & (rows["contract_rank"] <= top_k)]
            for walk in walks.itertuples(index=False):
                train_start = pd.Timestamp(walk.train_start)
                cutoff = pd.Timestamp(walk.cutoff)
                end = pd.Timestamp(walk.evaluation_end)
                train_interval = (
                    (base["window_start_time"] >= train_start) & (base["target_time"] < cutoff)
                )
                evaluation_interval = (
                    (base["decision_time"] >= cutoff) & (base["target_time"] < end)
                )
                for policy, eligible in {
                    "all_rows": pd.Series(True, index=base.index),
                    "recent_price_change_24h": base["recent_price_change_24h"],
                }.items():
                    train = base[train_interval & eligible]
                    evaluation = base[evaluation_interval & eligible]
                    records.append({
                        "timeframe": timeframe,
                        "top_k": top_k,
                        "walk": int(walk.walk),
                        "eligibility": policy,
                        "train_start": train_start,
                        "cutoff": cutoff,
                        "evaluation_end": end,
                        "task_train_rows": len(train),
                        "task_train_contracts": train["contract_rank"].nunique(),
                        "evaluation_rows": len(evaluation),
                        "evaluation_contracts": evaluation["contract_rank"].nunique(),
                        "zero_move_fraction": float((evaluation["delta"] == 0).mean()),
                        "stable_fraction": float((evaluation["abs_delta"] <= THRESHOLD).mean()),
                        "mean_absolute_move": float(evaluation["abs_delta"].mean()),
                        "p90_absolute_move": float(evaluation["abs_delta"].quantile(0.90)),
                    })
    return pd.DataFrame(records)


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def plot_comparison(aggregate: pd.DataFrame, eligibility: pd.DataFrame, output: Path) -> None:
    labels = [f"{row.timeframe} top-{row.top_k}" for row in aggregate.itertuples(index=False)]
    x = np.arange(len(labels))
    active = eligibility[eligibility["eligibility"] == "recent_price_change_24h"].reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    axes[0, 0].bar(x - 0.18, aggregate["pooled_zero_move_fraction"], 0.36, label="exact zero")
    axes[0, 0].bar(x + 0.18, aggregate["pooled_stable_fraction"], 0.36, label="stable")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("All-row eight-hour target staleness")
    axes[0, 0].legend()
    axes[0, 1].bar(x, active["retained_fraction"])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Rows retained by causal 24h activity rule")
    axes[1, 0].bar(x, active["mean_absolute_move"])
    axes[1, 0].set_title("Active-row zero-baseline MAE")
    axes[1, 0].set_ylabel("mean |8h probability change|")
    axes[1, 1].bar(x, aggregate["contracts_trailing_flat_ge_25pct"])
    axes[1, 1].set_title("Contracts with >=25% trailing flat tail")
    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=25, ha="right")
    fig.suptitle("Duration-matched one-hour versus four-hour activity suitability")
    fig.tight_layout()
    fig.savefig(output / "timeframe_activity_comparison.png", dpi=180)
    plt.close(fig)


def write_report(
    output: Path,
    aggregate: pd.DataFrame,
    eligibility: pd.DataFrame,
    lifecycle: pd.DataFrame,
    calendar: pd.DataFrame,
    alignment: pd.DataFrame,
) -> None:
    active = eligibility[eligibility["eligibility"] == "recent_price_change_24h"]
    calendar_focus = calendar[calendar["eligibility"] == "recent_price_change_24h"]
    lines = [
        "# Top-50/Top-80 One-Hour versus Four-Hour Activity Comparison", "",
        "Status: exploratory data analysis only; no model training or selection.", "",
        "The comparison holds temporal meaning fixed: 256 hours of input context, an "
        "eight-hour future probability-change target, and a causal prior-24h price-change "
        "activity rule. The same four-hour early-prefix rank defines contract identity.", "",
        "## All-row comparison", "", markdown_table(aggregate), "",
        "## Causal prior-24h activity comparison", "", markdown_table(active), "",
        "## Timestamp-aligned target agreement", "", markdown_table(alignment), "",
        "Raw timestamps are treated as bar starts, so decision and target availability "
        "times are shifted by one hour for 1h bars and four hours for 4h bars. This also "
        "aligns a 4h close with the corresponding final 1h close in its bar.", "",
        "## Lifecycle comparison", "", markdown_table(lifecycle), "",
        "## Two-walk global-calendar active-row comparison", "",
        markdown_table(calendar_focus), "",
        "![Timeframe comparison](timeframe_activity_comparison.png)", "",
        "One-hour rows overlap more heavily than four-hour rows. Row-count ratios are "
        "therefore computational-capacity comparisons, not ratios of independent information.", "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(output: Path, overwrite: bool) -> None:
    prepare_output(output, overwrite)
    contract_parts = []
    row_parts = []
    for timeframe in ("1h", "4h"):
        contracts, rows = load_timeframe(timeframe)
        contract_parts.append(contracts)
        row_parts.append(rows)
    contracts = pd.concat(contract_parts, ignore_index=True)
    rows = pd.concat(row_parts, ignore_index=True)
    aggregate = aggregate_summary(contracts, rows)
    eligibility = eligibility_summary(rows)
    lifecycle = lifecycle_summary(rows)
    calendar = calendar_summary(rows)
    alignment = aligned_target_summary(rows)
    contracts.to_csv(output / "contract_activity.csv", index=False)
    aggregate.to_csv(output / "aggregate_activity.csv", index=False)
    eligibility.to_csv(output / "causal_eligibility.csv", index=False)
    lifecycle.to_csv(output / "lifecycle_activity.csv", index=False)
    calendar.to_csv(output / "calendar_walk_activity.csv", index=False)
    alignment.to_csv(output / "aligned_target_agreement.csv", index=False)
    plot_comparison(aggregate, eligibility, output)
    write_report(output, aggregate, eligibility, lifecycle, calendar, alignment)
    summary = {
        "analysis_only": True,
        "model_training_launched": False,
        "timeframes": ["1h", "4h"],
        "top_k_values": [50, 80],
        "context_hours": 256,
        "target_horizon_hours": 8,
        "activity_lookback_hours": 24,
        "movement_threshold": THRESHOLD,
        "universe_policy": "shared_4h_256_row_early_prefix_rank",
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "walk_source": str(WALK_SOURCE),
        "walk_source_sha256": sha256_file(WALK_SOURCE),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args.output, args.overwrite)
        print(f"Completed one-hour/four-hour comparison: {args.output}")
        return 0
    except Exception as exc:
        print(f"One-hour/four-hour comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
