"""Audit recent FinData capacity under causal global-calendar walks.

This script is diagnostic only. It does not construct a training bundle or
launch a model. It measures how many one-hour sequences survive the selected
isolated-gap fill, observed endpoint, activity, target-maturity, and global
calendar-cutoff rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data_new/findata/polymarket/"
    "historical_diverse_top50_2025-12-01_2026-08-31"
)
DEFAULT_OUTPUT = DEFAULT_INPUT / "analysis/phase5_walk_capacity"
REFERENCE_PHASE3_ROWS = 21_696
STEP = pd.Timedelta(hours=1)
ACTIVITY_HOURS = 24


@dataclass(frozen=True)
class Walk:
    schedule: str
    walk: int
    train_start: pd.Timestamp
    cutoff: pd.Timestamp
    evaluation_end: pd.Timestamp


def utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def candidate_walks() -> list[Walk]:
    """Return capacity candidates fixed from calendar duration, not outcomes."""
    output = [
        Walk(
            "balanced_two_120d",
            1,
            utc("2025-12-02"),
            utc("2026-04-01"),
            utc("2026-06-16"),
        ),
        Walk(
            "balanced_two_120d",
            2,
            utc("2026-02-16"),
            utc("2026-06-16"),
            utc("2026-09-01"),
        ),
    ]
    for walk, cutoff_text in enumerate(
        ("2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01"),
        start=1,
    ):
        cutoff = utc(cutoff_text)
        output.append(
            Walk(
                "monthly_120d",
                walk,
                cutoff - pd.Timedelta(days=120),
                cutoff,
                min(cutoff + pd.offsets.MonthBegin(1), utc("2026-09-01")),
            )
        )
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candles = pd.read_parquet(input_dir / "candles_1h_clean.parquet")
    metadata = pd.read_parquet(input_dir / "market_metadata.parquet")
    quarantine = pd.read_parquet(input_dir / "candle_quarantine.parquet")
    required = {"condition_id", "date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(candles.columns))
    if missing:
        raise ValueError(f"clean hourly candles missing columns: {missing}")
    candles = candles.copy()
    candles["date"] = pd.to_datetime(candles["date"], utc=True, errors="raise")
    candles = candles.sort_values(["condition_id", "date"]).reset_index(drop=True)
    if candles.duplicated(["condition_id", "date"]).any():
        raise ValueError("duplicate condition_id/date rows in clean hourly candles")
    numeric = candles[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite clean hourly OHLCV values")
    quarantine = quarantine.loc[quarantine["resolution"].eq("1h")].copy()
    quarantine["date"] = pd.to_datetime(quarantine["date"], utc=True, errors="raise")
    quarantine["quarantine_available_at"] = pd.to_datetime(
        quarantine["quarantine_available_at"], utc=True, errors="raise"
    )
    return candles, metadata, quarantine


def complete_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill only a complete isolated missing hour and mark longer-gap segments."""
    records: list[dict[str, object]] = []
    segment = 0
    previous: pd.Series | None = None
    for _, row in frame.iterrows():
        if previous is not None:
            gap = row["date"] - previous["date"]
            if gap <= pd.Timedelta(0) or gap % STEP != pd.Timedelta(0):
                raise ValueError(f"invalid hourly step {gap} for {row['condition_id']}")
            missing = int(gap / STEP) - 1
            if missing == 1:
                records.append(
                    {
                        "condition_id": row["condition_id"],
                        "date": previous["date"] + STEP,
                        "close": float(previous["close"]),
                        "volume": 0.0,
                        "observed": False,
                        "is_imputed": True,
                        "segment": segment,
                    }
                )
            elif missing > 1:
                segment += 1
        records.append(
            {
                "condition_id": row["condition_id"],
                "date": row["date"],
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "observed": True,
                "is_imputed": False,
                "segment": segment,
            }
        )
        previous = row
    result = pd.DataFrame.from_records(records)
    result["availability"] = result["date"] + STEP
    result["price_changed"] = (
        result.groupby("segment", sort=False)["close"].diff().abs().fillna(0).gt(0)
    )
    result["active_24h"] = (
        result.groupby("segment", sort=False)["price_changed"]
        .rolling(ACTIVITY_HOURS, min_periods=ACTIVITY_HOURS)
        .max()
        .reset_index(level=0, drop=True)
        .eq(1.0)
    )
    result["segment_position"] = result.groupby("segment", sort=False).cumcount()
    result["imputed_prefix"] = result["is_imputed"].astype(int).cumsum()
    return result


def build_contracts(candles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(condition_id): complete_contract(frame.reset_index(drop=True))
        for condition_id, frame in candles.groupby("condition_id", sort=True)
    }


def sequence_rows(
    frame: pd.DataFrame,
    *,
    context_hours: int,
    horizon_hours: int,
) -> pd.DataFrame:
    count = len(frame)
    if count <= context_hours + horizon_hours:
        return pd.DataFrame()
    decision = np.arange(context_hours - 1, count - horizon_hours, dtype=np.int64)
    start = decision - context_hours + 1
    target = decision + horizon_hours
    same_context_segment = (
        frame["segment"].to_numpy()[start] == frame["segment"].to_numpy()[decision]
    )
    same_target_segment = (
        frame["segment"].to_numpy()[decision] == frame["segment"].to_numpy()[target]
    )
    observed = frame["observed"].to_numpy(bool)
    valid = same_context_segment & same_target_segment & observed[decision] & observed[target]
    decision = decision[valid]
    start = start[valid]
    target = target[valid]
    if not len(decision):
        return pd.DataFrame()
    prefix = frame["imputed_prefix"].to_numpy(np.int64)
    before = np.where(start > 0, prefix[start - 1], 0)
    context_imputed = prefix[decision] - before
    close = frame["close"].to_numpy(float)
    return pd.DataFrame(
        {
            "condition_id": frame["condition_id"].iloc[0],
            "window_start": frame["date"].to_numpy()[start],
            "decision_date": frame["date"].to_numpy()[decision],
            "decision_availability": frame["availability"].to_numpy()[decision],
            "target_date": frame["date"].to_numpy()[target],
            "target_availability": frame["availability"].to_numpy()[target],
            "active_24h": frame["active_24h"].to_numpy(bool)[decision],
            "context_imputed_rows": context_imputed,
            "delta": close[target] - close[decision],
        }
    )


def nonoverlapping_count(frame: pd.DataFrame) -> int:
    """Greedily count rows whose complete input+target intervals do not overlap."""
    selected = 0
    last_target: pd.Timestamp | None = None
    for row in frame.sort_values("target_availability").itertuples(index=False):
        start = pd.Timestamp(row.window_start)
        if last_target is None or start >= last_target:
            selected += 1
            last_target = pd.Timestamp(row.target_availability)
    return selected


def concentration(frame: pd.DataFrame) -> dict[str, float | int]:
    counts = frame.groupby("condition_id").size().sort_values(ascending=False)
    if counts.empty:
        return {
            "contracts": 0,
            "rows_per_contract_min": 0,
            "rows_per_contract_median": 0.0,
            "top1_row_share": 0.0,
            "top5_row_share": 0.0,
            "effective_contracts": 0.0,
        }
    shares = counts / counts.sum()
    return {
        "contracts": int(len(counts)),
        "rows_per_contract_min": int(counts.min()),
        "rows_per_contract_median": float(counts.median()),
        "top1_row_share": float(shares.iloc[:1].sum()),
        "top5_row_share": float(shares.iloc[:5].sum()),
        "effective_contracts": float(1.0 / np.square(shares).sum()),
    }


def analyze_partition(
    rows: pd.DataFrame,
    walk: Walk,
    *,
    affected: set[str],
    exclude_affected: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    allowed = rows.loc[~rows["condition_id"].isin(affected)].copy() if exclude_affected else rows
    train_all = allowed.loc[
        allowed["window_start"].ge(walk.train_start)
        & allowed["target_availability"].lt(walk.cutoff)
    ].copy()
    train = train_all.loc[train_all["active_24h"]].copy()
    train_counts = train.groupby("condition_id").size()
    supported = set(train_counts.index)
    well_supported = set(train_counts[train_counts.ge(256)].index)
    evaluation_all = allowed.loc[
        allowed["decision_availability"].ge(walk.cutoff)
        & allowed["target_availability"].lt(walk.evaluation_end)
    ].copy()
    evaluation = evaluation_all.loc[
        evaluation_all["active_24h"] & evaluation_all["condition_id"].isin(supported)
    ].copy()
    evaluation_well_supported = evaluation.loc[
        evaluation["condition_id"].isin(well_supported)
    ].copy()
    train_concentration = concentration(train)
    eval_concentration = concentration(evaluation)
    record: dict[str, object] = {
        "schedule": walk.schedule,
        "walk": walk.walk,
        "train_start": walk.train_start,
        "cutoff": walk.cutoff,
        "evaluation_end": walk.evaluation_end,
        "exclude_affected_contracts": exclude_affected,
        "train_all_rows": len(train_all),
        "train_active_rows": len(train),
        "train_unimputed_context_rows": int(train["context_imputed_rows"].eq(0).sum()),
        "train_active_fraction": len(train) / len(train_all) if len(train_all) else np.nan,
        "train_nonoverlapping_rows": int(
            sum(nonoverlapping_count(group) for _, group in train.groupby("condition_id"))
        )
        if len(train)
        else 0,
        "train_rows_vs_phase3_reference": len(train) / REFERENCE_PHASE3_ROWS,
        "train_contracts": train_concentration["contracts"],
        "train_contracts_ge_256_rows": len(well_supported),
        "train_rows_per_contract_min": train_concentration["rows_per_contract_min"],
        "train_rows_per_contract_median": train_concentration["rows_per_contract_median"],
        "train_top1_row_share": train_concentration["top1_row_share"],
        "train_top5_row_share": train_concentration["top5_row_share"],
        "train_effective_contracts": train_concentration["effective_contracts"],
        "train_categories": int(train["category"].nunique()) if len(train) else 0,
        "train_event_families": int(train["event_family"].nunique()) if len(train) else 0,
        "train_context_imputed_fraction": float(train["context_imputed_rows"].gt(0).mean())
        if len(train)
        else np.nan,
        "train_zero_target_fraction": float(train["delta"].eq(0).mean()) if len(train) else np.nan,
        "evaluation_all_rows": len(evaluation_all),
        "evaluation_active_supported_rows": len(evaluation),
        "evaluation_unimputed_context_rows": int(
            evaluation["context_imputed_rows"].eq(0).sum()
        ),
        "evaluation_well_supported_rows": len(evaluation_well_supported),
        "evaluation_contracts": eval_concentration["contracts"],
        "evaluation_rows_per_contract_min": eval_concentration["rows_per_contract_min"],
        "evaluation_rows_per_contract_median": eval_concentration["rows_per_contract_median"],
        "evaluation_top5_row_share": eval_concentration["top5_row_share"],
        "evaluation_effective_contracts": eval_concentration["effective_contracts"],
        "evaluation_categories": int(evaluation["category"].nunique()) if len(evaluation) else 0,
        "evaluation_event_families": int(evaluation["event_family"].nunique())
        if len(evaluation)
        else 0,
        "evaluation_context_imputed_fraction": float(
            evaluation["context_imputed_rows"].gt(0).mean()
        )
        if len(evaluation)
        else np.nan,
        "evaluation_zero_target_fraction": float(evaluation["delta"].eq(0).mean())
        if len(evaluation)
        else np.nan,
    }
    contract_rows = []
    identities = sorted(set(train_counts.index).union(evaluation["condition_id"].unique()))
    for condition_id in identities:
        train_contract = train.loc[train["condition_id"].eq(condition_id)]
        evaluation_contract = evaluation.loc[evaluation["condition_id"].eq(condition_id)]
        contract_rows.append(
            {
                "schedule": walk.schedule,
                "walk": walk.walk,
                "exclude_affected_contracts": exclude_affected,
                "condition_id": condition_id,
                "train_rows": len(train_contract),
                "train_nonoverlapping_rows": nonoverlapping_count(train_contract),
                "evaluation_rows": len(evaluation_contract),
                "well_supported": condition_id in well_supported,
                "category": train_contract["category"].iloc[0]
                if len(train_contract)
                else evaluation_contract["category"].iloc[0],
                "event_family": train_contract["event_family"].iloc[0]
                if len(train_contract)
                else evaluation_contract["event_family"].iloc[0],
            }
        )
    return record, pd.DataFrame(contract_rows)


def quarantine_audit(quarantine: pd.DataFrame, walks: list[Walk]) -> pd.DataFrame:
    records = []
    for walk in walks:
        training = quarantine.loc[
            quarantine["date"].add(STEP).ge(walk.train_start)
            & quarantine["date"].add(STEP).lt(walk.cutoff)
        ]
        evaluation = quarantine.loc[
            quarantine["date"].add(STEP).ge(walk.cutoff)
            & quarantine["date"].add(STEP).lt(walk.evaluation_end)
        ]
        records.append(
            {
                "schedule": walk.schedule,
                "walk": walk.walk,
                "training_quarantines": len(training),
                "training_decisions_unavailable_at_cutoff": int(
                    training["quarantine_available_at"].ge(walk.cutoff).sum()
                ),
                "evaluation_quarantines": len(evaluation),
                "evaluation_preconfirmation_decision_slots_upper_bound": int(
                    (
                        evaluation["quarantine_available_at"]
                        - evaluation["date"].add(STEP)
                    )
                    .dt.total_seconds()
                    .div(3600)
                    .clip(lower=0)
                    .sum()
                ),
                "maximum_decision_delay_hours": float(
                    (
                        pd.concat([training, evaluation])["quarantine_available_at"]
                        - pd.concat([training, evaluation])["date"].add(STEP)
                    ).dt.total_seconds().div(3600).max()
                )
                if len(training) + len(evaluation)
                else 0.0,
            }
        )
    return pd.DataFrame(records)


def write_report(summary: pd.DataFrame, quarantine: pd.DataFrame, output: Path) -> None:
    primary = summary.loc[
        summary["schedule"].eq("balanced_two_120d")
        & summary["context_hours"].eq(64)
        & summary["horizon_hours"].eq(1)
    ].copy()
    columns = [
        "walk",
        "exclude_affected_contracts",
        "train_active_rows",
        "train_unimputed_context_rows",
        "train_nonoverlapping_rows",
        "train_contracts",
        "train_contracts_ge_256_rows",
        "train_rows_vs_phase3_reference",
        "evaluation_active_supported_rows",
        "evaluation_unimputed_context_rows",
        "evaluation_contracts",
        "evaluation_categories",
        "evaluation_event_families",
        "train_top5_row_share",
        "train_context_imputed_fraction",
    ]
    report = f"""# Recent FinData Walk-Forward Capacity Audit

This is a data-capacity diagnostic, not a training bundle or model result. The
50-contract cohort was selected retrospectively for source auditing, so these
counts establish feasibility only. They do not validate the still-required
cutoff-local candidate discovery and universe selection.

## Balanced two-walk candidate (`seq64`, one-hour target)

{markdown_table(primary[columns])}

## Interpretation rules

- Training uses a fixed rolling calendar interval and targets that mature
  strictly before the shared cutoff.
- Only complete isolated one-hour gaps are filled. Longer gaps split sequences.
- Decision and target endpoints must be observed.
- The primary row population requires a price change in the preceding 24
  hours, calculated causally inside the current uninterrupted segment.
- Evaluation contracts must have active training rows before the cutoff.
- The affected-contract exclusion is reported separately.
- `train_nonoverlapping_rows` greedily counts disjoint input-plus-target
  intervals within each contract and is a conservative antidote to treating
  stride-one windows as independent samples.

## Quarantine availability at the cutoffs

{markdown_table(quarantine)}

The final-clean files are conservative for capacity, but a production builder
must replay each quarantine decision only after `quarantine_available_at`.
The preconfirmation count is an upper bound on contract-hour decision slots
that the final-clean retrospective view can suppress before the corresponding
quarantine decision was knowable.

## Remaining validity boundary

The split mechanics are sound if replayed exactly, but this cohort cannot prove
the universe-selection policy: its 50 contracts were chosen with full-period
information. Before training, recollect or retain a broader candidate pool and
freeze the universe at each cutoff using only then-available history. The walk
builder must also save identical framework/baseline row identities and all
preprocessing/checkpoint hashes.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contexts", default="64,256")
    parser.add_argument("--horizons", default="1,2")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contexts = [int(value) for value in args.contexts.split(",") if value]
    horizons = [int(value) for value in args.horizons.split(",") if value]
    if any(value < ACTIVITY_HOURS for value in contexts):
        raise ValueError(f"contexts must be at least {ACTIVITY_HOURS} hours")
    if any(value < 1 for value in horizons):
        raise ValueError("horizons must be positive")
    prepare_output(args.output_dir, args.overwrite)
    candles, metadata, quarantine = load_inputs(args.input_dir)
    contracts = build_contracts(candles)
    affected = set(quarantine["condition_id"].astype(str))
    walks = candidate_walks()
    records: list[dict[str, object]] = []
    contract_records: list[pd.DataFrame] = []
    for context in contexts:
        for horizon in horizons:
            row_parts = [
                sequence_rows(frame, context_hours=context, horizon_hours=horizon)
                for frame in contracts.values()
            ]
            rows = pd.concat([part for part in row_parts if len(part)], ignore_index=True)
            rows = rows.merge(
                metadata[["condition_id", "category", "event_family"]].drop_duplicates(
                    "condition_id"
                ),
                on="condition_id",
                how="left",
                validate="many_to_one",
            )
            if rows[["category", "event_family"]].isna().any().any():
                raise ValueError("missing category/event-family metadata for sequence rows")
            for walk in walks:
                for exclude_affected in (False, True):
                    record, details = analyze_partition(
                        rows,
                        walk,
                        affected=affected,
                        exclude_affected=exclude_affected,
                    )
                    record["context_hours"] = context
                    record["horizon_hours"] = horizon
                    details["context_hours"] = context
                    details["horizon_hours"] = horizon
                    records.append(record)
                    contract_records.append(details)
    summary = pd.DataFrame(records).sort_values(
        ["schedule", "context_hours", "horizon_hours", "exclude_affected_contracts", "walk"]
    )
    contract_summary = pd.concat(contract_records, ignore_index=True)
    quarantine_summary = quarantine_audit(quarantine, walks)
    summary.to_csv(args.output_dir / "walk_capacity.csv", index=False)
    contract_summary.to_parquet(args.output_dir / "contract_capacity.parquet", index=False)
    quarantine_summary.to_csv(args.output_dir / "quarantine_cutoff_audit.csv", index=False)
    write_report(summary, quarantine_summary, args.output_dir)
    manifest = {
        "kind": "findata_walk_capacity_diagnostic",
        "input_dir": str(args.input_dir.resolve()),
        "source_hashes": {
            name: sha256(args.input_dir / name)
            for name in (
                "candles_1h_clean.parquet",
                "market_metadata.parquet",
                "candle_quarantine.parquet",
            )
        },
        "contracts": len(contracts),
        "clean_rows": len(candles),
        "affected_contracts": len(affected),
        "contexts": contexts,
        "horizons": horizons,
        "activity_hours": ACTIVITY_HOURS,
        "reference_phase3_rows": REFERENCE_PHASE3_ROWS,
        "walks": [asdict(walk) for walk in walks],
        "limitations": [
            "retrospectively selected 50-contract source-audit cohort",
            "final clean rows used conservatively; production must replay quarantine availability",
            "counts do not substitute for a replayable training bundle",
        ],
        "metadata_contracts": int(metadata["condition_id"].nunique()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
