"""Audit top-80 one-hour timestamps and walk-forward sample capacity.

This script is diagnostic only. It does not prepare model inputs or launch
encoder/downstream training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/representation_learning_framework_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data/phase3/processed/market_4h_seq64_top50.npz.manifest.json"
DEFAULT_OUTPUT = ROOT / "experiments/phase4/data_analysis/top80_1h_timestamp_capacity"
TOP_K = 80
HORIZON_STEPS = 8
TAU = 0.005
REFERENCE_4H_ENCODER_TRAIN_ROWS = 21696


@dataclass(frozen=True)
class CalendarWalk:
    walk: int
    train_start: pd.Timestamp
    cutoff: pd.Timestamp
    evaluation_end: pd.Timestamp


@dataclass(frozen=True)
class RelativeWalk:
    walk: int
    train_start_fraction: float
    cutoff_fraction: float
    evaluation_end_fraction: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_contracts() -> tuple[list[dict[str, object]], pd.DataFrame]:
    contracts: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for record in selected_contracts():
        one_hour_name = str(record["filename"]).replace("-4h.feather", "-1h.feather")
        path = ROOT / "data" / one_hour_name
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_feather(path)
        required = ["date", "open", "high", "low", "close", "volume"]
        missing_columns = sorted(set(required).difference(frame.columns))
        if missing_columns:
            raise ValueError(f"missing columns {missing_columns}: {path}")
        timestamps = pd.to_datetime(frame["date"], errors="raise")
        differences = timestamps.diff().dropna().dt.total_seconds().to_numpy() / 3600.0
        numeric = frame[["open", "high", "low", "close", "volume"]].apply(
            pd.to_numeric, errors="coerce"
        )
        nonfinite = int((~np.isfinite(numeric.to_numpy(np.float64))).sum())
        on_hour = (
            (timestamps.dt.minute == 0)
            & (timestamps.dt.second == 0)
            & (timestamps.dt.microsecond == 0)
        )
        audit.append({
            "contract_rank": int(record["rank"]),
            "filename": one_hour_name,
            "rows": len(frame),
            "timestamp_dtype": str(frame["date"].dtype),
            "timestamp_start": timestamps.iloc[0],
            "timestamp_end": timestamps.iloc[-1],
            "strictly_increasing": bool(timestamps.is_monotonic_increasing and not timestamps.duplicated().any()),
            "duplicate_timestamps": int(timestamps.duplicated().sum()),
            "non_hour_aligned_timestamps": int((~on_hour).sum()),
            "non_1h_steps": int(np.sum(differences != 1.0)),
            "maximum_step_hours": float(np.max(differences)) if len(differences) else None,
            "missing_or_nonfinite_ohlcv_cells": nonfinite,
            "source_sha256": sha256_file(path),
        })
        contracts.append({
            "contract_rank": int(record["rank"]),
            "filename": one_hour_name,
            "timestamps": timestamps.to_numpy(dtype="datetime64[ns]"),
            "close": numeric["close"].to_numpy(np.float64),
            "rows": len(frame),
        })
    audit_frame = pd.DataFrame(audit)
    invalid = audit_frame[
        (~audit_frame["strictly_increasing"])
        | (audit_frame["duplicate_timestamps"] != 0)
        | (audit_frame["non_hour_aligned_timestamps"] != 0)
        | (audit_frame["non_1h_steps"] != 0)
        | (audit_frame["missing_or_nonfinite_ohlcv_cells"] != 0)
    ]
    if len(invalid):
        names = ", ".join(invalid["filename"].astype(str).tolist())
        raise ValueError(f"timestamp/OHLCV audit failed for: {names}")
    return contracts, audit_frame


def build_rows(contracts: list[dict[str, object]], seq_len: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for contract in contracts:
        timestamps = np.asarray(contract["timestamps"], dtype="datetime64[ns]")
        close = np.asarray(contract["close"], dtype=np.float64)
        count = len(timestamps)
        if count <= seq_len + HORIZON_STEPS:
            raise ValueError(f"insufficient rows: {contract['filename']}")
        decision = np.arange(seq_len - 1, count - HORIZON_STEPS, dtype=np.int64)
        window_start = decision - seq_len + 1
        target = decision + HORIZON_STEPS
        delta = close[target] - close[decision]
        parts.append(pd.DataFrame({
            "contract_rank": int(contract["contract_rank"]),
            "filename": str(contract["filename"]),
            "window_start_time": timestamps[window_start],
            "decision_time": timestamps[decision] + np.timedelta64(1, "h"),
            "target_time": timestamps[target] + np.timedelta64(1, "h"),
            "window_start_fraction": window_start / (count - 1),
            "decision_fraction": decision / (count - 1),
            "target_fraction": target / (count - 1),
            "delta": delta,
            "label": np.where(delta < -TAU, "DOWN", np.where(delta > TAU, "UP", "STABLE")),
        }))
    rows = pd.concat(parts, ignore_index=True)
    rows.sort_values(["decision_time", "contract_rank"], inplace=True, ignore_index=True)
    return rows


def calendar_walks(rows: pd.DataFrame, walk_count: int, scheme: str) -> list[CalendarWalk]:
    start = pd.Timestamp(rows["window_start_time"].min())
    observed = np.sort(rows["decision_time"].drop_duplicates().to_numpy(dtype="datetime64[ns]"))
    end = pd.Timestamp(observed[-1]) + pd.Timedelta(hours=1)
    total = end - start

    def snap(value: pd.Timestamp) -> pd.Timestamp:
        target = np.datetime64(value.to_datetime64(), "ns").astype(np.int64)
        raw = observed.astype(np.int64)
        index = int(np.searchsorted(raw, target, side="left"))
        choices = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(raw)]
        return pd.Timestamp(observed[min(choices, key=lambda candidate: abs(raw[candidate] - target))])

    cutoffs = [snap(start + total * (0.5 + i * 0.5 / walk_count)) for i in range(walk_count)]
    initial_duration = cutoffs[0] - start
    output = []
    for index, cutoff in enumerate(cutoffs):
        train_start = start if scheme == "calendar_anchored" else cutoff - initial_duration
        evaluation_end = end if index == walk_count - 1 else cutoffs[index + 1]
        output.append(CalendarWalk(index + 1, train_start, cutoff, evaluation_end))
    return output


def relative_walks(walk_count: int, scheme: str) -> list[RelativeWalk]:
    step = 0.5 / walk_count
    output = []
    for index in range(walk_count):
        cutoff = 0.5 + index * step
        train_start = 0.0 if scheme == "relative_anchored" else index * step
        output.append(RelativeWalk(index + 1, train_start, cutoff, cutoff + step))
    return output


def summarize_partition(frame: pd.DataFrame) -> dict[str, object]:
    counts = frame.groupby("contract_rank").size()
    return {
        "rows": len(frame),
        "contracts": frame["contract_rank"].nunique(),
        "rows_per_contract_min": int(counts.min()) if len(counts) else 0,
        "rows_per_contract_median": float(counts.median()) if len(counts) else 0.0,
        "contracts_ge_64_rows": int((counts >= 64).sum()),
        "contracts_ge_256_rows": int((counts >= 256).sum()),
    }


def analyze_scheme(
    rows: pd.DataFrame,
    seq_len: int,
    walk_count: int,
    scheme: str,
) -> list[dict[str, object]]:
    output = []
    if scheme.startswith("calendar_"):
        walks = calendar_walks(rows, walk_count, scheme)
        for walk in walks:
            complete = rows["window_start_time"] >= walk.train_start
            encoder = rows[complete & (rows["decision_time"] < walk.cutoff)]
            task = rows[complete & (rows["target_time"] < walk.cutoff)]
            evaluation = rows[
                (rows["decision_time"] >= walk.cutoff)
                & (rows["target_time"] < walk.evaluation_end)
            ]
            output.append({
                "timeframe": "1h", "sequence_length": seq_len,
                "context_hours": seq_len, "horizon_steps": HORIZON_STEPS,
                "horizon_hours": HORIZON_STEPS, "walk_count": walk_count,
                "scheme": scheme, **asdict(walk),
                **{f"encoder_train_{key}": value for key, value in summarize_partition(encoder).items()},
                **{f"task_train_{key}": value for key, value in summarize_partition(task).items()},
                **{f"evaluation_{key}": value for key, value in summarize_partition(evaluation).items()},
                "task_train_to_phase3_4h_ratio": len(task) / REFERENCE_4H_ENCODER_TRAIN_ROWS,
                "evaluation_zero_move_fraction": float((evaluation["delta"] == 0).mean()),
                "evaluation_stable_fraction": float((evaluation["label"] == "STABLE").mean()),
            })
    else:
        walks = relative_walks(walk_count, scheme)
        for walk in walks:
            complete = rows["window_start_fraction"] >= walk.train_start_fraction
            encoder = rows[complete & (rows["decision_fraction"] < walk.cutoff_fraction)]
            task = rows[complete & (rows["target_fraction"] < walk.cutoff_fraction)]
            evaluation = rows[
                (rows["decision_fraction"] >= walk.cutoff_fraction)
                & (rows["target_fraction"] < walk.evaluation_end_fraction)
            ]
            output.append({
                "timeframe": "1h", "sequence_length": seq_len,
                "context_hours": seq_len, "horizon_steps": HORIZON_STEPS,
                "horizon_hours": HORIZON_STEPS, "walk_count": walk_count,
                "scheme": scheme, **asdict(walk),
                **{f"encoder_train_{key}": value for key, value in summarize_partition(encoder).items()},
                **{f"task_train_{key}": value for key, value in summarize_partition(task).items()},
                **{f"evaluation_{key}": value for key, value in summarize_partition(evaluation).items()},
                "task_train_to_phase3_4h_ratio": len(task) / REFERENCE_4H_ENCODER_TRAIN_ROWS,
                "evaluation_zero_move_fraction": float((evaluation["delta"] == 0).mean()),
                "evaluation_stable_fraction": float((evaluation["label"] == "STABLE").mean()),
            })
    return output


def plot_capacity(capacity: pd.DataFrame, output: Path) -> None:
    selected = capacity[(capacity["walk_count"] == 3)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=False)
    for row_index, seq_len in enumerate((64, 256)):
        for column_index, family in enumerate(("calendar", "relative")):
            ax = axes[row_index, column_index]
            part = selected[
                (selected["sequence_length"] == seq_len)
                & selected["scheme"].str.startswith(family)
            ]
            for scheme, group in part.groupby("scheme"):
                ax.plot(group["walk"], group["task_train_rows"], marker="o", label=scheme)
            ax.axhline(
                REFERENCE_4H_ENCODER_TRAIN_ROWS, color="black", linestyle="--", linewidth=1,
                label="completed 4h train rows",
            )
            ax.set_title(f"{family}: sequence {seq_len} ({seq_len}h context)")
            ax.set_xlabel("walk")
            ax.set_ylabel("supervised training rows")
            ax.legend(fontsize=8)
    fig.suptitle("Top-80 one-hour sample capacity under candidate walk policies")
    fig.tight_layout()
    fig.savefig(output / "training_capacity.png", dpi=180)
    plt.close(fig)


def write_report(output: Path, audit: pd.DataFrame, capacity: pd.DataFrame) -> None:
    timestamp_summary = pd.DataFrame([{
        "contracts": len(audit),
        "total_rows": int(audit["rows"].sum()),
        "minimum_rows": int(audit["rows"].min()),
        "median_rows": float(audit["rows"].median()),
        "maximum_rows": int(audit["rows"].max()),
        "earliest_timestamp": audit["timestamp_start"].min(),
        "latest_timestamp": audit["timestamp_end"].max(),
        "strictly_increasing_contracts": int(audit["strictly_increasing"].sum()),
        "duplicate_timestamps": int(audit["duplicate_timestamps"].sum()),
        "non_hour_aligned_timestamps": int(audit["non_hour_aligned_timestamps"].sum()),
        "non_1h_steps": int(audit["non_1h_steps"].sum()),
        "missing_or_nonfinite_ohlcv_cells": int(audit["missing_or_nonfinite_ohlcv_cells"].sum()),
    }])
    focus = capacity[
        (capacity["walk_count"] == 3)
        & capacity["scheme"].isin(["calendar_window", "relative_window"])
    ][[
        "sequence_length", "scheme", "walk", "task_train_rows", "task_train_contracts",
        "task_train_rows_per_contract_median", "evaluation_rows", "evaluation_contracts",
        "task_train_to_phase3_4h_ratio",
    ]]
    lines = [
        "# Top-80 One-Hour Timestamp and Training-Capacity Audit",
        "",
        "Status: data analysis only; no model training or Phase 4 bundle generation.",
        "",
        "## Timestamp audit",
        "",
        markdown_table(timestamp_summary),
        "",
        "The same 80 contract identities used by the four-hour analysis all have matching "
        "one-hour feather files. The `date` field is checked contract by contract for type, "
        "ordering, duplicates, hourly alignment, and step regularity.",
        "",
        "## Three-walk capacity focus",
        "",
        markdown_table(focus),
        "",
        f"The ratio uses `{REFERENCE_4H_ENCODER_TRAIN_ROWS}` rows as the reference workload "
        "from the completed Phase 3 four-hour encoder bundle. It is a scale comparison, not "
        "a proof that two datasets have equal information content.",
        "",
        "Sequence length 64 gives 64 hours of one-hour context. Sequence length 256 preserves "
        "the 256-hour duration of the existing four-hour, length-64 inputs. Both are reported "
        "because comparing only sequence length 64 would silently change the temporal context.",
        "",
        "Contract-relative rows are retrospective lifecycle evidence. Calendar rows establish "
        "raw capacity for a deployable split, but the top-80 cohort itself remains retrospective "
        "until a cutoff-local universe rule is frozen.",
        "",
        "![Training capacity](training_capacity.png)",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(output: Path, overwrite: bool) -> None:
    prepare_output(output, overwrite)
    contracts, audit = load_contracts()
    records: list[dict[str, object]] = []
    for seq_len in (64, 256):
        rows = build_rows(contracts, seq_len)
        for walk_count in (2, 3):
            for scheme in (
                "calendar_anchored", "calendar_window",
                "relative_anchored", "relative_window",
            ):
                records.extend(analyze_scheme(rows, seq_len, walk_count, scheme))
    capacity = pd.DataFrame(records)
    audit.to_csv(output / "contract_timestamp_audit.csv", index=False)
    capacity.to_csv(output / "walk_capacity.csv", index=False)
    plot_capacity(capacity, output)
    write_report(output, audit, capacity)
    summary = {
        "analysis_only": True,
        "timeframe": "1h",
        "contracts": TOP_K,
        "source_contract_identity_manifest": str(SOURCE_MANIFEST),
        "source_contract_identity_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "selection_note": "same prefix-ranked contract identities as top-80 4h analysis",
        "sequence_lengths": [64, 256],
        "context_hours": [64, 256],
        "horizon_steps": HORIZON_STEPS,
        "horizon_hours": HORIZON_STEPS,
        "walk_counts": [2, 3],
        "schemes": [
            "calendar_anchored", "calendar_window",
            "relative_anchored", "relative_window",
        ],
        "reference_4h_encoder_train_rows": REFERENCE_4H_ENCODER_TRAIN_ROWS,
        "model_training_launched": False,
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
        print(f"Completed one-hour timestamp/capacity audit: {args.output}")
        return 0
    except Exception as exc:
        print(f"One-hour timestamp/capacity audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
