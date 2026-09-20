"""Analyze targets under one contract-relative walk-forward scheme.

This is a data-analysis entry point, not a training launcher. It compares the
available encoder and supervised rows and describes probability-point movement,
arithmetic return, and DOWN/STABLE/UP labels for a retrospective top-K cohort.
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
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/representation_learning_framework_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = (
    ROOT / "data/phase3/processed/market_4h_seq64_top50.npz.manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "experiments/phase4/data_analysis/top80_contract_relative_walk_forward"
)
SCHEMES = ("relative_anchored", "relative_window")
STAGES = ("early", "middle", "late")
CLASSES = ("DOWN", "STABLE", "UP")


@dataclass(frozen=True)
class Walk:
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


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


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
    required = DEFAULT_OUTPUT_ROOT.resolve()
    if resolved != required and required not in resolved.parents:
        raise ValueError(f"output must be under {required}: {resolved}")
    if any(part.endswith("_old") for part in resolved.parts):
        raise ValueError(f"legacy output path is prohibited: {resolved}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite occupied output: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def select_contracts(manifest_path: Path, top_k: int) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = manifest.get("universe_selection", {})
    if selection.get("selection_prefix_rows") != 256:
        raise ValueError("expected the Phase 3 256-row prefix-ranking provenance")
    records = [
        row for row in selection.get("candidate_records", [])
        if row.get("eligible") and row.get("rank") is not None and int(row["rank"]) <= top_k
    ]
    records.sort(key=lambda row: int(row["rank"]))
    if len(records) != top_k:
        raise ValueError(f"manifest provides {len(records)} eligible ranks, expected {top_k}")
    selected = pd.DataFrame(records)
    expected = np.arange(1, top_k + 1)
    if not np.array_equal(selected["rank"].to_numpy(dtype=np.int64), expected):
        raise ValueError("selected contract ranks are not contiguous")
    return selected


def load_rows(
    selected: pd.DataFrame, seq_len: int, horizon: int, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    contract_rows: list[dict[str, object]] = []
    for record in selected.to_dict("records"):
        path = ROOT / "data" / str(record["filename"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != record["source_sha256"]:
            raise ValueError(f"raw source hash mismatch: {path}")
        frame = pd.read_feather(path, columns=["date", "close"])
        timestamps = pd.to_datetime(frame["date"], errors="raise")
        close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(np.float64)
        if len(frame) != int(record["raw_row_count"]):
            raise ValueError(f"raw row count mismatch: {path}")
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise ValueError(f"timestamps are not strictly increasing: {path}")
        if not np.isfinite(close).all():
            raise ValueError(f"non-finite close values: {path}")
        if np.any((close < 0.0) | (close > 1.0)):
            raise ValueError(f"close outside [0, 1]: {path}")
        if len(frame) <= seq_len + horizon:
            raise ValueError(f"insufficient rows after selection: {path}")

        decision_index = np.arange(seq_len - 1, len(frame) - horizon, dtype=np.int64)
        window_start_index = decision_index - seq_len + 1
        target_index = decision_index + horizon
        current = close[decision_index]
        future = close[target_index]
        delta = future - current
        valid_return = current != 0.0
        arithmetic_return = np.full(len(delta), np.nan, dtype=np.float64)
        arithmetic_return[valid_return] = delta[valid_return] / current[valid_return]
        lifecycle_fraction = decision_index / (len(frame) - 1)
        window_start_fraction = window_start_index / (len(frame) - 1)
        target_fraction = target_index / (len(frame) - 1)
        lifecycle_id = np.minimum((lifecycle_fraction * 3).astype(np.int64), 2)
        label = np.full(len(delta), "STABLE", dtype="U6")
        label[delta < -threshold] = "DOWN"
        label[delta > threshold] = "UP"

        parts.append(pd.DataFrame({
            "contract_rank": int(record["rank"]),
            "filename": str(record["filename"]),
            "window_start_time": timestamps.iloc[window_start_index].to_numpy(),
            "decision_time": timestamps.iloc[decision_index].to_numpy(),
            "target_time": timestamps.iloc[target_index].to_numpy(),
            "current_close": current,
            "future_close": future,
            "probability_movement": delta,
            "absolute_probability_movement": np.abs(delta),
            "arithmetic_return": arithmetic_return,
            "return_is_defined": valid_return,
            "movement_label": label,
            "window_start_fraction": window_start_fraction,
            "lifecycle_fraction": lifecycle_fraction,
            "target_fraction": target_fraction,
            "lifecycle_stage": np.asarray(STAGES, dtype=object)[lifecycle_id],
        }))
        contract_rows.append({
            "contract_rank": int(record["rank"]),
            "filename": str(record["filename"]),
            "raw_rows": len(frame),
            "raw_start": timestamps.iloc[0],
            "raw_end": timestamps.iloc[-1],
            "eligible_target_rows": len(decision_index),
            "source_sha256": actual_hash,
        })
    rows = pd.concat(parts, ignore_index=True)
    rows.sort_values(["decision_time", "contract_rank"], inplace=True, ignore_index=True)
    return rows, pd.DataFrame(contract_rows).sort_values("contract_rank")


def build_walks(
    scheme: str,
    walk_count: int,
    initial_train_fraction: float,
) -> list[Walk]:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme: {scheme}")
    if walk_count < 2:
        raise ValueError("walk_count must be at least 2")
    if not 0.0 < initial_train_fraction < 1.0:
        raise ValueError("initial_train_fraction must lie in (0, 1)")
    evaluation_fraction = (1.0 - initial_train_fraction) / walk_count
    walks: list[Walk] = []
    for index in range(walk_count):
        cutoff = initial_train_fraction + index * evaluation_fraction
        evaluation_end = initial_train_fraction + (index + 1) * evaluation_fraction
        train_start = 0.0 if scheme == "relative_anchored" else index * evaluation_fraction
        walks.append(Walk(index + 1, train_start, cutoff, evaluation_end))
    return walks


def quantile(series: pd.Series, q: float) -> float | None:
    values = series.to_numpy(np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else None


def frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for column in columns:
        values = frame[column]
        if pd.api.types.is_datetime64_any_dtype(values):
            array = values.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        else:
            array = values.to_numpy()
        contiguous = np.ascontiguousarray(array)
        digest.update(column.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def target_summary(part: pd.DataFrame, scheme: str, walk: int, partition: str) -> dict[str, object]:
    delta = part["probability_movement"]
    returns = part.loc[part["return_is_defined"], "arithmetic_return"]
    return {
        "scheme": scheme,
        "walk": walk,
        "partition": partition,
        "rows": len(part),
        "contracts": part["contract_rank"].nunique(),
        "identity_sha256": frame_hash(
            part, ("contract_rank", "decision_time", "target_time")
        ),
        "probability_movement_sha256": frame_hash(
            part, ("probability_movement",)
        ),
        "probability_movement_mean": float(delta.mean()) if len(part) else None,
        "probability_movement_std": float(delta.std(ddof=0)) if len(part) else None,
        "zero_movement_fraction": float((delta == 0.0).mean()) if len(part) else None,
        "stable_tau005_fraction": float((delta.abs() <= 0.005).mean()) if len(part) else None,
        "zero_baseline_mae": float(delta.abs().mean()) if len(part) else None,
        "zero_baseline_rmse": float(np.sqrt(np.mean(delta.to_numpy() ** 2))) if len(part) else None,
        "abs_movement_q50": quantile(delta.abs(), 0.50),
        "abs_movement_q90": quantile(delta.abs(), 0.90),
        "abs_movement_q95": quantile(delta.abs(), 0.95),
        "abs_movement_q99": quantile(delta.abs(), 0.99),
        "return_defined_rows": len(returns),
        "return_undefined_zero_price_rows": int((~part["return_is_defined"]).sum()),
        "return_defined_fraction": float(part["return_is_defined"].mean()) if len(part) else None,
        "arithmetic_return_median": quantile(returns, 0.50),
        "absolute_return_q50": quantile(returns.abs(), 0.50),
        "absolute_return_q90": quantile(returns.abs(), 0.90),
        "absolute_return_q95": quantile(returns.abs(), 0.95),
        "absolute_return_q99": quantile(returns.abs(), 0.99),
        "absolute_return_gt_1_fraction": float((returns.abs() > 1.0).mean()) if len(returns) else None,
        "absolute_return_gt_10_fraction": float((returns.abs() > 10.0).mean()) if len(returns) else None,
        "near_boundary_fraction": float(
            ((part["current_close"] <= 0.01) | (part["current_close"] >= 0.99)).mean()
        ) if len(part) else None,
    }


def rows_by_contract(
    frame: pd.DataFrame, scheme: str, walk: int, partition: str
) -> list[dict[str, object]]:
    grouped = frame.groupby(["contract_rank", "filename"], observed=True)
    return [
        {
            "scheme": scheme,
            "walk": walk,
            "partition": partition,
            "contract_rank": int(rank),
            "filename": filename,
            "rows": len(part),
        }
        for (rank, filename), part in grouped
    ]


def label_rows(frame: pd.DataFrame, scheme: str, walk: int, partition: str) -> list[dict[str, object]]:
    counts = frame["movement_label"].value_counts()
    total = len(frame)
    return [
        {
            "scheme": scheme,
            "walk": walk,
            "partition": partition,
            "class": label,
            "rows": int(counts.get(label, 0)),
            "fraction": float(counts.get(label, 0) / total) if total else None,
        }
        for label in CLASSES
    ]


def lifecycle_rows(
    frame: pd.DataFrame, scheme: str, walk: int, partition: str
) -> list[dict[str, object]]:
    counts = frame["lifecycle_stage"].value_counts()
    total = len(frame)
    return [
        {
            "scheme": scheme,
            "walk": walk,
            "partition": partition,
            "lifecycle_stage": stage,
            "rows": int(counts.get(stage, 0)),
            "fraction": float(counts.get(stage, 0) / total) if total else None,
        }
        for stage in STAGES
    ]


def return_band_rows(
    frame: pd.DataFrame, scheme: str, walk: int, partition: str
) -> list[dict[str, object]]:
    bands = (
        ("zero", lambda x: x == 0.0),
        ("(0,.01]", lambda x: (x > 0.0) & (x <= 0.01)),
        ("(.01,.10]", lambda x: (x > 0.01) & (x <= 0.10)),
        ("(.10,.90)", lambda x: (x > 0.10) & (x < 0.90)),
        ("[.90,.99)", lambda x: (x >= 0.90) & (x < 0.99)),
        ("[.99,1]", lambda x: x >= 0.99),
    )
    output: list[dict[str, object]] = []
    for name, predicate in bands:
        part = frame[predicate(frame["current_close"])]
        valid = part.loc[part["return_is_defined"], "arithmetic_return"]
        output.append({
            "scheme": scheme,
            "walk": walk,
            "partition": partition,
            "current_price_band": name,
            "rows": len(part),
            "fraction": float(len(part) / len(frame)) if len(frame) else None,
            "return_defined_rows": len(valid),
            "zero_baseline_mae_probability_points": (
                float(part["absolute_probability_movement"].mean()) if len(part) else None
            ),
            "absolute_return_q50": quantile(valid.abs(), 0.50),
            "absolute_return_q95": quantile(valid.abs(), 0.95),
            "absolute_return_q99": quantile(valid.abs(), 0.99),
        })
    return output


def analyze(
    rows: pd.DataFrame,
    walks: Iterable[Walk],
    scheme: str,
) -> dict[str, pd.DataFrame]:
    walk_records: list[dict[str, object]] = []
    target_records: list[dict[str, object]] = []
    label_records: list[dict[str, object]] = []
    lifecycle_records: list[dict[str, object]] = []
    band_records: list[dict[str, object]] = []
    contract_records: list[dict[str, object]] = []

    for walk in walks:
        complete_window = rows["window_start_fraction"] >= walk.train_start_fraction
        before_cutoff = rows["lifecycle_fraction"] < walk.cutoff_fraction
        encoder_train = rows[complete_window & before_cutoff]
        # A supervised row is usable only when its future target lies strictly
        # before the contract-relative cutoff.
        task_train = rows[
            complete_window & (rows["target_fraction"] < walk.cutoff_fraction)
        ]
        evaluation = rows[
            (rows["lifecycle_fraction"] >= walk.cutoff_fraction)
            & (rows["lifecycle_fraction"] < walk.evaluation_end_fraction)
        ]
        if len(task_train) == 0 or len(evaluation) == 0:
            raise ValueError(f"walk {walk.walk} has an empty train or evaluation partition")
        overlap = task_train.merge(
            evaluation[["contract_rank", "decision_time"]],
            on=["contract_rank", "decision_time"], how="inner",
        )
        if len(overlap):
            raise AssertionError(f"walk {walk.walk} train/evaluation decision overlap")
        if not (task_train["target_fraction"] < walk.cutoff_fraction).all():
            raise AssertionError(f"walk {walk.walk} contains an immature training target")

        per_contract_train = task_train.groupby("contract_rank").size()
        per_contract_eval = evaluation.groupby("contract_rank").size()
        walk_records.append({
            "scheme": scheme,
            **asdict(walk),
            "encoder_train_rows": len(encoder_train),
            "task_train_rows": len(task_train),
            "evaluation_rows": len(evaluation),
            "encoder_train_contracts": encoder_train["contract_rank"].nunique(),
            "task_train_contracts": task_train["contract_rank"].nunique(),
            "evaluation_contracts": evaluation["contract_rank"].nunique(),
            "task_train_contracts_ge_64_rows": int((per_contract_train >= 64).sum()),
            "task_train_contracts_ge_256_rows": int((per_contract_train >= 256).sum()),
            "evaluation_contracts_ge_16_rows": int((per_contract_eval >= 16).sum()),
            "task_train_rows_median_per_active_contract": float(per_contract_train.median()),
            "evaluation_rows_median_per_active_contract": float(per_contract_eval.median()),
            "immature_rows_excluded_from_task_train": len(encoder_train) - len(task_train),
        })
        for name, frame in (("train", task_train), ("evaluation", evaluation)):
            target_records.append(target_summary(frame, scheme, walk.walk, name))
            label_records.extend(label_rows(frame, scheme, walk.walk, name))
            lifecycle_records.extend(lifecycle_rows(frame, scheme, walk.walk, name))
            band_records.extend(return_band_rows(frame, scheme, walk.walk, name))
            contract_records.extend(rows_by_contract(frame, scheme, walk.walk, name))

    return {
        "walk_summary": pd.DataFrame(walk_records),
        "target_summary": pd.DataFrame(target_records),
        "label_distribution": pd.DataFrame(label_records),
        "lifecycle_distribution": pd.DataFrame(lifecycle_records),
        "return_by_price_band": pd.DataFrame(band_records),
        "sample_counts_by_contract": pd.DataFrame(contract_records),
    }


def plot_results(tables: dict[str, pd.DataFrame], out_dir: Path, scheme: str) -> None:
    walks = tables["walk_summary"]
    targets = tables["target_summary"]
    labels = tables["label_distribution"]
    lifecycle = tables["lifecycle_distribution"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    x = np.arange(len(walks))
    axes[0, 0].bar(x - 0.25, walks["encoder_train_rows"], width=0.25, label="encoder train")
    axes[0, 0].bar(x, walks["task_train_rows"], width=0.25, label="task train")
    axes[0, 0].bar(x + 0.25, walks["evaluation_rows"], width=0.25, label="evaluation")
    axes[0, 0].set_xticks(x, [f"walk {value}" for value in walks["walk"]])
    axes[0, 0].set_title("Available samples")
    axes[0, 0].legend(fontsize=8)

    for partition, marker in (("train", "o"), ("evaluation", "s")):
        part = targets[targets["partition"] == partition]
        axes[0, 1].plot(part["walk"], part["zero_movement_fraction"], marker=marker, label=partition)
    axes[0, 1].set_title("Exact-zero probability movement")
    axes[0, 1].set_xlabel("walk")
    axes[0, 1].set_ylabel("fraction")
    axes[0, 1].legend(fontsize=8)

    evaluation_labels = labels[labels["partition"] == "evaluation"]
    bottoms = np.zeros(len(walks))
    for label in CLASSES:
        values = evaluation_labels[evaluation_labels["class"] == label].sort_values("walk")["fraction"].to_numpy()
        axes[1, 0].bar(x, values, bottom=bottoms, label=label)
        bottoms += values
    axes[1, 0].set_xticks(x, [f"walk {value}" for value in walks["walk"]])
    axes[1, 0].set_title("Evaluation movement-label balance")
    axes[1, 0].set_ylabel("fraction")
    axes[1, 0].legend(fontsize=8)

    evaluation_lifecycle = lifecycle[lifecycle["partition"] == "evaluation"]
    bottoms = np.zeros(len(walks))
    for stage in STAGES:
        values = evaluation_lifecycle[
            evaluation_lifecycle["lifecycle_stage"] == stage
        ].sort_values("walk")["fraction"].to_numpy()
        axes[1, 1].bar(x, values, bottom=bottoms, label=stage)
        bottoms += values
    axes[1, 1].set_xticks(x, [f"walk {value}" for value in walks["walk"]])
    axes[1, 1].set_title("Evaluation lifecycle composition")
    axes[1, 1].set_ylabel("fraction")
    axes[1, 1].legend(fontsize=8)

    title = (
        "Contract-relative anchored"
        if scheme == "relative_anchored"
        else "Contract-relative fixed-length"
    )
    fig.suptitle(f"Top-80 4-hour lifecycle analysis: {title}")
    fig.tight_layout()
    fig.savefig(out_dir / "walk_data_overview.png", dpi=180)
    plt.close(fig)


def markdown_report(
    out_dir: Path,
    scheme: str,
    config: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> None:
    walks = tables["walk_summary"]
    targets = tables["target_summary"]
    labels = tables["label_distribution"]
    lines = [
        f"# Phase 4 Top-80 Walk Data Analysis — `{scheme}`",
        "",
        "Status: exploratory data feasibility analysis; no model was trained.",
        "",
        "## Contract",
        "",
        f"- Timeframe: `{config['timeframe']}`",
        f"- Sequence length: `{config['sequence_length']}`",
        f"- Horizon: `{config['horizon_steps']}` steps (`{config['horizon_hours']}` hours)",
        f"- Label threshold: `{config['threshold']}` probability points",
        f"- Contracts: `{config['top_k']}` retrospective prefix-ranked contracts",
        f"- Contract-relative walk scheme: `{scheme}`",
        "",
        "The top-80 cohort is reused from the Phase 3 per-contract 256-row prefix ranking. "
        "Every boundary is a fraction of each contract's own observed length. This is a "
        "retrospective lifecycle analysis, not a causally deployable pooled-model split: "
        "relative cutoffs map to different calendar dates and observed final length is future "
        "information unless the termination boundary was known at decision time.",
        "",
        "## Walk sample availability",
        "",
        markdown_table(walks),
        "",
        "## Probability movement and return diagnostics",
        "",
        markdown_table(targets),
        "",
        "Arithmetic return is defined as `(future_close - current_close) / current_close`. "
        "Rows with current close equal to zero are retained for probability movement but "
        "reported as undefined for arithmetic return.",
        "",
        "## DOWN/STABLE/UP label distribution",
        "",
        markdown_table(labels),
        "",
        "Training rows require the complete 64-step input window to lie inside the walk's "
        "relative training interval and require the target position to be strictly before the "
        "relative cutoff. The analysis does not make a no-future-leak deployment claim.",
        "",
        "![Walk data overview](walk_data_overview.png)",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = args.output_root / args.scheme
    prepare_output(out_dir, args.overwrite)
    selected = select_contracts(args.source_manifest, args.top_k)
    rows, contract_summary = load_rows(
        selected, args.sequence_length, args.horizon, args.threshold
    )
    walks = build_walks(args.scheme, args.walk_count, args.initial_train_fraction)
    tables = analyze(rows, walks, args.scheme)

    config = {
        "analysis_only": True,
        "scheme": args.scheme,
        "timeframe": "4h",
        "top_k": args.top_k,
        "sequence_length": args.sequence_length,
        "horizon_steps": args.horizon,
        "horizon_hours": 4 * args.horizon,
        "threshold": args.threshold,
        "walk_count": args.walk_count,
        "initial_train_fraction": args.initial_train_fraction,
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "universe_policy": "retrospective_phase3_256_row_prefix_rank_top_k",
        "universe_is_phase4_causal": False,
        "training_window_policy": (
            "contract_relative_anchored_start"
            if args.scheme == "relative_anchored"
            else "contract_relative_fixed_length"
        ),
        "boundary_basis": "raw_row_position_divided_by_contract_final_index",
        "uses_retrospective_final_contract_length": True,
        "training_input_policy": "complete_sequence_inside_relative_training_interval",
        "training_target_policy": "target_fraction_strictly_before_relative_cutoff",
        "evaluation_policy": "disjoint_half_open_contract_relative_intervals",
    }
    write_json(out_dir / "config.json", config)
    selected.to_csv(out_dir / "selected_contracts.csv", index=False)
    contract_summary.to_csv(out_dir / "contract_source_summary.csv", index=False)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    plot_results(tables, out_dir, args.scheme)
    markdown_report(out_dir, args.scheme, config, tables)
    write_json(out_dir / "artifact_hashes.json", {
        path.name: sha256_file(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    })
    return out_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--walk-count", type=int, default=3)
    parser.add_argument("--initial-train-fraction", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.threshold != 0.005:
        raise ValueError("the initial movement-label analysis is fixed to tau=0.005")
    if args.top_k != 80 or args.sequence_length != 64 or args.horizon != 2:
        raise ValueError("the requested exploratory contract is fixed to top80/seq64/h2")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        output = run(args)
        print(f"Completed Phase 4 {args.scheme} data analysis: {output}")
        return 0
    except Exception as exc:
        print(f"Phase 4 walk data analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
