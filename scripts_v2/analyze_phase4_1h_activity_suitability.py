"""Audit whether the top-80 one-hour cohort is informative enough for Phase 4.

This is exploratory data analysis only. It does not prepare model artifacts or
train/select any model.
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
CAPACITY_ROOT = ROOT / "experiments/phase4/data_analysis/top80_1h_timestamp_capacity"
DEFAULT_OUTPUT = ROOT / "experiments/phase4/data_analysis/top80_1h_activity_suitability"
TOP_K = 80
SEQUENCE_LENGTH = 256
HORIZON = 8
THRESHOLD = 0.005


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


def selected_contracts() -> list[dict[str, object]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    records = [
        row for row in manifest["universe_selection"]["candidate_records"]
        if row.get("eligible") and row.get("rank") is not None and int(row["rank"]) <= TOP_K
    ]
    records.sort(key=lambda row: int(row["rank"]))
    if len(records) != TOP_K:
        raise ValueError(f"expected {TOP_K} selected contract identities, found {len(records)}")
    return records


def longest_constant_run(values: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    return int(np.diff(np.r_[starts, len(values)]).max())


def trailing_constant_run(values: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    changes = np.flatnonzero(values[1:] != values[:-1])
    return len(values) if len(changes) == 0 else int(len(values) - (changes[-1] + 1))


def rolling_any(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values.astype(np.int8))
        .rolling(window=window, min_periods=1)
        .sum()
        .to_numpy()
        > 0
    )


def quantiles(values: pd.Series, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_p10": float(values.quantile(0.10)),
        f"{prefix}_p25": float(values.quantile(0.25)),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_p75": float(values.quantile(0.75)),
        f"{prefix}_p90": float(values.quantile(0.90)),
    }


def load_and_measure() -> tuple[pd.DataFrame, pd.DataFrame]:
    contract_records: list[dict[str, object]] = []
    row_parts: list[pd.DataFrame] = []
    for record in selected_contracts():
        filename = str(record["filename"]).replace("-4h.feather", "-1h.feather")
        path = ROOT / "data" / filename
        frame = pd.read_feather(path)
        timestamps = pd.to_datetime(frame["date"], errors="raise")
        close = pd.to_numeric(frame["close"], errors="raise").to_numpy(np.float64)
        volume = pd.to_numeric(frame["volume"], errors="raise").to_numpy(np.float64)
        if not np.isfinite(close).all() or not np.isfinite(volume).all():
            raise ValueError(f"non-finite values: {path}")
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise ValueError(f"invalid timestamps: {path}")
        if len(close) <= SEQUENCE_LENGTH + HORIZON:
            raise ValueError(f"insufficient rows: {path}")

        one_hour_delta = np.diff(close)
        all_eight_hour_delta = close[HORIZON:] - close[:-HORIZON]
        decisions = np.arange(SEQUENCE_LENGTH - 1, len(close) - HORIZON, dtype=np.int64)
        targets = decisions + HORIZON
        delta = close[targets] - close[decisions]
        relative_position = decisions / (len(close) - 1)
        labels = np.where(delta < -THRESHOLD, "DOWN", np.where(delta > THRESHOLD, "UP", "STABLE"))
        positive_volume = volume[volume > 0]
        changed = np.r_[False, one_hour_delta != 0]
        recent_change_24h = rolling_any(changed, 24)
        recent_change_72h = rolling_any(changed, 72)
        recent_volume_24h = rolling_any(volume > 0, 24)
        trailing_run = trailing_constant_run(close)
        trailing_start = len(close) - trailing_run

        contract_records.append({
            "contract_rank": int(record["rank"]),
            "filename": filename,
            "raw_rows": len(close),
            "lifespan_hours": float((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600),
            "model_eligible_rows": len(delta),
            "close_mean": float(close.mean()),
            "close_std": float(close.std()),
            "close_iqr": float(np.quantile(close, 0.75) - np.quantile(close, 0.25)),
            "close_range": float(close.max() - close.min()),
            "near_boundary_fraction": float(((close <= 0.05) | (close >= 0.95)).mean()),
            "unique_close_values": int(np.unique(close).size),
            "one_hour_zero_move_fraction": float((one_hour_delta == 0).mean()),
            "one_hour_abs_move_mean": float(np.abs(one_hour_delta).mean()),
            "all_eight_hour_zero_move_fraction": float((all_eight_hour_delta == 0).mean()),
            "eight_hour_zero_move_fraction": float((delta == 0).mean()),
            "eight_hour_stable_fraction": float((np.abs(delta) <= THRESHOLD).mean()),
            "eight_hour_mean_abs_move": float(np.abs(delta).mean()),
            "eight_hour_median_abs_move": float(np.median(np.abs(delta))),
            "eight_hour_p90_abs_move": float(np.quantile(np.abs(delta), 0.90)),
            "eight_hour_delta_std": float(delta.std()),
            "eight_hour_meaningful_005_fraction": float((np.abs(delta) > 0.005).mean()),
            "eight_hour_meaningful_010_fraction": float((np.abs(delta) > 0.010).mean()),
            "down_fraction": float((labels == "DOWN").mean()),
            "up_fraction": float((labels == "UP").mean()),
            "positive_volume_fraction": float((volume > 0).mean()),
            "zero_volume_fraction": float((volume == 0).mean()),
            "median_log1p_positive_volume": float(np.median(np.log1p(positive_volume))),
            "total_log1p_volume": float(np.log1p(np.maximum(volume, 0)).sum()),
            "longest_constant_close_run_hours": longest_constant_run(close),
            "trailing_constant_close_run_hours": trailing_run,
            "trailing_constant_close_fraction": float(trailing_run / len(close)),
            "trailing_constant_close": float(close[-1]),
            "trailing_positive_volume_fraction": float((volume[trailing_start:] > 0).mean()),
            "last_price_change_time": (
                timestamps.iloc[trailing_start] if trailing_start > 0 else timestamps.iloc[0]
            ),
            "source_sha256": sha256_file(path),
        })
        row_parts.append(pd.DataFrame({
            "contract_rank": int(record["rank"]),
            "window_start_time": timestamps.iloc[decisions - SEQUENCE_LENGTH + 1].to_numpy(),
            "decision_time": (timestamps.iloc[decisions] + pd.Timedelta(hours=1)).to_numpy(),
            "target_time": (timestamps.iloc[targets] + pd.Timedelta(hours=1)).to_numpy(),
            "relative_position": relative_position,
            "current_close": close[decisions],
            "delta": delta,
            "abs_delta": np.abs(delta),
            "label": labels,
            "recent_price_change_24h": recent_change_24h[decisions],
            "recent_price_change_72h": recent_change_72h[decisions],
            "recent_positive_volume_24h": recent_volume_24h[decisions],
        }))
    return pd.DataFrame(contract_records), pd.concat(row_parts, ignore_index=True)


def lifecycle_summary(rows: pd.DataFrame) -> pd.DataFrame:
    stage = pd.cut(
        rows["relative_position"], bins=[0, 1 / 3, 2 / 3, 1],
        labels=["early", "middle", "late"], include_lowest=True, right=False,
    )
    work = rows.assign(lifecycle_stage=stage)
    records = []
    for name, part in work.groupby("lifecycle_stage", observed=True):
        records.append({
            "lifecycle_stage": str(name),
            "rows": len(part),
            "contracts": part["contract_rank"].nunique(),
            "zero_move_fraction": float((part["delta"] == 0).mean()),
            "stable_fraction": float((part["abs_delta"] <= THRESHOLD).mean()),
            "mean_absolute_move": float(part["abs_delta"].mean()),
            "median_absolute_move": float(part["abs_delta"].median()),
            "p90_absolute_move": float(part["abs_delta"].quantile(0.90)),
            "near_boundary_fraction": float(
                ((part["current_close"] <= 0.05) | (part["current_close"] >= 0.95)).mean()
            ),
        })
    return pd.DataFrame(records)


def calendar_evaluation_summary(rows: pd.DataFrame) -> pd.DataFrame:
    capacity = pd.read_csv(CAPACITY_ROOT / "walk_capacity.csv")
    walks = capacity[
        (capacity["walk_count"] == 2)
        & (capacity["scheme"] == "calendar_window")
        & (capacity["sequence_length"] == SEQUENCE_LENGTH)
    ].sort_values("walk")
    if len(walks) != 2:
        raise ValueError("expected two duration-matched calendar-window records")
    policies = {
        "all_rows": pd.Series(True, index=rows.index),
        "recent_price_change_24h": rows["recent_price_change_24h"],
        "recent_price_change_72h": rows["recent_price_change_72h"],
        "recent_change_and_volume_24h": (
            rows["recent_price_change_24h"] & rows["recent_positive_volume_24h"]
        ),
    }
    records = []
    for row in walks.itertuples(index=False):
        train_start = pd.Timestamp(row.train_start)
        cutoff = pd.Timestamp(row.cutoff)
        end = pd.Timestamp(row.evaluation_end)
        training_interval = (
            (rows["window_start_time"] >= train_start) & (rows["target_time"] < cutoff)
        )
        interval = (rows["decision_time"] >= cutoff) & (rows["target_time"] < end)
        for policy, eligible in policies.items():
            training = rows[training_interval & eligible]
            part = rows[interval & eligible]
            records.append({
                "walk": int(row.walk),
                "eligibility": policy,
                "train_start": train_start,
                "cutoff": cutoff,
                "evaluation_end": end,
                "task_train_rows": len(training),
                "task_train_contracts": training["contract_rank"].nunique(),
                "rows": len(part),
                "contracts": part["contract_rank"].nunique(),
                "zero_move_fraction": float((part["delta"] == 0).mean()),
                "stable_fraction": float((part["abs_delta"] <= THRESHOLD).mean()),
                "mean_absolute_move": float(part["abs_delta"].mean()),
                "median_absolute_move": float(part["abs_delta"].median()),
                "p90_absolute_move": float(part["abs_delta"].quantile(0.90)),
                "down_fraction": float((part["delta"] < -THRESHOLD).mean()),
                "up_fraction": float((part["delta"] > THRESHOLD).mean()),
            })
    return pd.DataFrame(records)


def causal_eligibility_summary(rows: pd.DataFrame) -> pd.DataFrame:
    policies = {
        "all_rows": pd.Series(True, index=rows.index),
        "recent_positive_volume_24h": rows["recent_positive_volume_24h"],
        "recent_price_change_72h": rows["recent_price_change_72h"],
        "recent_price_change_24h": rows["recent_price_change_24h"],
        "recent_change_and_volume_24h": (
            rows["recent_price_change_24h"] & rows["recent_positive_volume_24h"]
        ),
    }
    records = []
    for name, eligible in policies.items():
        part = rows[eligible]
        records.append({
            "eligibility": name,
            "rows": len(part),
            "retained_fraction": float(len(part) / len(rows)),
            "contracts": part["contract_rank"].nunique(),
            "zero_move_fraction": float((part["delta"] == 0).mean()),
            "stable_fraction": float((part["abs_delta"] <= THRESHOLD).mean()),
            "mean_absolute_move": float(part["abs_delta"].mean()),
            "median_absolute_move": float(part["abs_delta"].median()),
            "p90_absolute_move": float(part["abs_delta"].quantile(0.90)),
        })
    return pd.DataFrame(records)


def suitability_summary(contracts: pd.DataFrame, rows: pd.DataFrame) -> dict[str, object]:
    volume_rank = contracts["positive_volume_fraction"].rank(method="average")
    movement_rank = contracts["eight_hour_meaningful_005_fraction"].rank(method="average")
    volume_movement_spearman = float(volume_rank.corr(movement_rank))
    low_information = (
        (contracts["one_hour_zero_move_fraction"] >= 0.80)
        & (contracts["eight_hour_meaningful_005_fraction"] < 0.10)
    )
    return {
        "contracts": len(contracts),
        "raw_rows": int(contracts["raw_rows"].sum()),
        "model_eligible_rows": int(len(rows)),
        "pooled_zero_move_fraction": float((rows["delta"] == 0).mean()),
        "pooled_stable_fraction": float((rows["abs_delta"] <= THRESHOLD).mean()),
        "pooled_mean_absolute_move": float(rows["abs_delta"].mean()),
        "pooled_median_absolute_move": float(rows["abs_delta"].median()),
        "pooled_p90_absolute_move": float(rows["abs_delta"].quantile(0.90)),
        "contracts_one_hour_zero_ge_80pct": int((contracts["one_hour_zero_move_fraction"] >= 0.80).sum()),
        "contracts_one_hour_zero_ge_90pct": int((contracts["one_hour_zero_move_fraction"] >= 0.90).sum()),
        "contracts_eight_hour_meaningful_lt_10pct": int(
            (contracts["eight_hour_meaningful_005_fraction"] < 0.10).sum()
        ),
        "contracts_eight_hour_meaningful_lt_5pct": int(
            (contracts["eight_hour_meaningful_005_fraction"] < 0.05).sum()
        ),
        "contracts_close_std_lt_001": int((contracts["close_std"] < 0.01).sum()),
        "contracts_positive_volume_lt_50pct": int((contracts["positive_volume_fraction"] < 0.50).sum()),
        "contracts_positive_volume_lt_25pct": int((contracts["positive_volume_fraction"] < 0.25).sum()),
        "contracts_flat_run_ge_24h": int((contracts["longest_constant_close_run_hours"] >= 24).sum()),
        "contracts_flat_run_ge_72h": int((contracts["longest_constant_close_run_hours"] >= 72).sum()),
        "contracts_trailing_flat_run_ge_24h": int(
            (contracts["trailing_constant_close_run_hours"] >= 24).sum()
        ),
        "contracts_trailing_flat_fraction_ge_25pct": int(
            (contracts["trailing_constant_close_fraction"] >= 0.25).sum()
        ),
        "low_information_composite_contracts": int(low_information.sum()),
        "positive_volume_vs_meaningful_movement_spearman": volume_movement_spearman,
        **quantiles(contracts["one_hour_zero_move_fraction"], "contract_one_hour_zero_fraction"),
        **quantiles(contracts["eight_hour_meaningful_005_fraction"], "contract_meaningful_8h_fraction"),
        **quantiles(contracts["close_std"], "contract_close_std"),
        **quantiles(contracts["positive_volume_fraction"], "contract_positive_volume_fraction"),
        **quantiles(contracts["longest_constant_close_run_hours"], "contract_longest_flat_run_hours"),
    }


def plot_diagnostics(contracts: pd.DataFrame, lifecycle: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    scatter = axes[0, 0].scatter(
        contracts["positive_volume_fraction"],
        contracts["eight_hour_meaningful_005_fraction"],
        c=contracts["near_boundary_fraction"], cmap="viridis", alpha=0.8,
    )
    axes[0, 0].set_xlabel("positive-volume hour fraction")
    axes[0, 0].set_ylabel("fraction with |8h move| > 0.005")
    axes[0, 0].set_title("Trading activity versus target movement")
    fig.colorbar(scatter, ax=axes[0, 0], label="near-boundary price fraction")

    axes[0, 1].hist(contracts["one_hour_zero_move_fraction"], bins=16, alpha=0.75)
    axes[0, 1].axvline(0.8, color="red", linestyle="--", label="80% unchanged")
    axes[0, 1].set_xlabel("unchanged one-hour close fraction")
    axes[0, 1].set_ylabel("contracts")
    axes[0, 1].set_title("Per-contract one-hour staleness")
    axes[0, 1].legend()

    axes[1, 0].hist(contracts["longest_constant_close_run_hours"], bins=16, alpha=0.75)
    axes[1, 0].axvline(24, color="red", linestyle="--", label="24 hours")
    axes[1, 0].set_xlabel("longest identical-close run (hours)")
    axes[1, 0].set_ylabel("contracts")
    axes[1, 0].set_title("Longest stale run")
    axes[1, 0].legend()

    x = np.arange(len(lifecycle))
    axes[1, 1].bar(x - 0.18, lifecycle["zero_move_fraction"], 0.36, label="exact zero")
    axes[1, 1].bar(x + 0.18, lifecycle["stable_fraction"], 0.36, label="|move| <= 0.005")
    axes[1, 1].set_xticks(x, lifecycle["lifecycle_stage"])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("row fraction")
    axes[1, 1].set_title("Staleness across contract lifecycle")
    axes[1, 1].legend()

    fig.suptitle("Top-80 one-hour activity and movement suitability")
    fig.tight_layout()
    fig.savefig(output / "activity_suitability.png", dpi=180)
    plt.close(fig)


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


def write_report(
    output: Path,
    summary: dict[str, object],
    lifecycle: pd.DataFrame,
    calendar: pd.DataFrame,
    eligibility: pd.DataFrame,
    contracts: pd.DataFrame,
) -> None:
    most_stale = contracts.nlargest(10, "one_hour_zero_move_fraction")[[
        "contract_rank", "filename", "raw_rows", "positive_volume_fraction",
        "one_hour_zero_move_fraction", "eight_hour_meaningful_005_fraction",
        "longest_constant_close_run_hours", "trailing_constant_close_run_hours",
        "trailing_constant_close_fraction",
    ]]
    lines = [
        "# Top-80 One-Hour Activity Suitability Audit", "",
        "Status: exploratory data analysis only; no model training or selection.", "",
        "The cohort uses the same retrospectively identified, early-prefix-ranked top-80 "
        "contract identities as the four-hour lifecycle analysis. Metrics use a 256-hour "
        "input eligibility rule and an eight-hour future probability-movement target.", "",
        "## Aggregate findings", "",
        f"- Raw rows: `{summary['raw_rows']}`; model-eligible rows: `{summary['model_eligible_rows']}`.",
        f"- Pooled exact-zero eight-hour targets: `{summary['pooled_zero_move_fraction']:.2%}`.",
        f"- Pooled stable targets (`|delta| <= 0.005`): `{summary['pooled_stable_fraction']:.2%}`.",
        f"- Pooled mean / median / p90 absolute eight-hour move: "
        f"`{summary['pooled_mean_absolute_move']:.6f}` / "
        f"`{summary['pooled_median_absolute_move']:.6f}` / "
        f"`{summary['pooled_p90_absolute_move']:.6f}`.",
        f"- Contracts with at least 80% unchanged one-hour closes: "
        f"`{summary['contracts_one_hour_zero_ge_80pct']}` / `{TOP_K}`.",
        f"- Contracts with fewer than 10% meaningful eight-hour moves: "
        f"`{summary['contracts_eight_hour_meaningful_lt_10pct']}` / `{TOP_K}`.",
        f"- Contracts with a constant-close run of at least 24h / 72h: "
        f"`{summary['contracts_flat_run_ge_24h']}` / `{summary['contracts_flat_run_ge_72h']}`.",
        f"- Contracts whose trailing constant-price tail occupies at least 25% of the file: "
        f"`{summary['contracts_trailing_flat_fraction_ge_25pct']}` / `{TOP_K}`.",
        f"- Low-information composite contracts: `{summary['low_information_composite_contracts']}` / `{TOP_K}`.",
        f"- Rank correlation between positive-volume fraction and meaningful-movement fraction: "
        f"`{summary['positive_volume_vs_meaningful_movement_spearman']:.4f}`.", "",
        "The low-information composite is an exploratory flag, not an exclusion rule: at "
        "least 80% unchanged one-hour closes and fewer than 10% of eligible eight-hour "
        "targets moving by more than 0.005.", "",
        "## Causal activity-filter sensitivity", "", markdown_table(eligibility), "",
        "The recent-activity filters use only information available at each decision time. "
        "They are diagnostics, not yet frozen inclusion rules. Trailing-flat statistics are "
        "retrospective diagnostics and must not themselves be used as causal filters.", "",
        "## Lifecycle distribution", "", markdown_table(lifecycle), "",
        "## Two-walk global-calendar evaluation distribution", "", markdown_table(calendar), "",
        "## Ten stalest contracts by one-hour close", "", markdown_table(most_stale), "",
        "![Activity suitability](activity_suitability.png)", "",
        "Large row counts do not by themselves establish learnability. The target variance, "
        "contract coverage, lifecycle composition, and performance against an exact-zero "
        "movement baseline must all be reported in later Phase 4 experiments.", "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(output: Path, overwrite: bool) -> None:
    prepare_output(output, overwrite)
    contracts, rows = load_and_measure()
    lifecycle = lifecycle_summary(rows)
    calendar = calendar_evaluation_summary(rows)
    eligibility = causal_eligibility_summary(rows)
    summary = suitability_summary(contracts, rows)
    summary.update({
        "analysis_only": True,
        "model_training_launched": False,
        "timeframe": "1h",
        "top_k": TOP_K,
        "sequence_length": SEQUENCE_LENGTH,
        "context_hours": SEQUENCE_LENGTH,
        "horizon_steps": HORIZON,
        "horizon_hours": HORIZON,
        "movement_threshold": THRESHOLD,
        "selection_note": "same prefix-ranked contract identities as top-80 4h analysis",
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "capacity_artifact_sha256": sha256_file(CAPACITY_ROOT / "walk_capacity.csv"),
    })
    contracts.to_csv(output / "contract_activity.csv", index=False)
    lifecycle.to_csv(output / "lifecycle_activity.csv", index=False)
    calendar.to_csv(output / "calendar_evaluation_activity.csv", index=False)
    eligibility.to_csv(output / "causal_eligibility_sensitivity.csv", index=False)
    plot_diagnostics(contracts, lifecycle, output)
    write_report(output, summary, lifecycle, calendar, eligibility, contracts)
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
        print(f"Completed one-hour activity suitability audit: {args.output}")
        return 0
    except Exception as exc:
        print(f"One-hour activity suitability audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
