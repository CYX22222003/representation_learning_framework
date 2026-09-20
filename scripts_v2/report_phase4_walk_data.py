"""Compare completed contract-relative anchored and fixed-length analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    ROOT / "experiments/phase4/data_analysis/top80_contract_relative_walk_forward"
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified(root: Path, scheme: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    directory = root / scheme
    hashes = json.loads((directory / "artifact_hashes.json").read_text())
    for name, expected in hashes.items():
        path = directory / name
        if sha256_file(path) != expected:
            raise ValueError(f"artifact hash mismatch: {path}")
    config = json.loads((directory / "config.json").read_text())
    tables = {
        name: pd.read_csv(directory / f"{name}.csv")
        for name in (
            "walk_summary", "target_summary", "label_distribution",
            "lifecycle_distribution", "return_by_price_band",
        )
    }
    return config, tables


def run(root: Path, overwrite: bool) -> Path:
    report_dir = root / "comparison"
    if report_dir.exists() and any(report_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite occupied report: {report_dir}")
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    anchored_config, anchored = load_verified(root, "relative_anchored")
    window_config, window = load_verified(root, "relative_window")
    comparable_keys = (
        "top_k", "sequence_length", "horizon_steps", "threshold", "walk_count",
        "initial_train_fraction", "source_manifest_sha256",
    )
    for key in comparable_keys:
        if anchored_config[key] != window_config[key]:
            raise ValueError(f"analysis configurations differ for {key}")

    left_eval = anchored["target_summary"].query("partition == 'evaluation'").reset_index(drop=True)
    right_eval = window["target_summary"].query("partition == 'evaluation'").reset_index(drop=True)
    evaluation_columns = [
        "walk", "rows", "contracts", "identity_sha256",
        "probability_movement_sha256", "probability_movement_mean",
        "zero_movement_fraction", "stable_tau005_fraction", "zero_baseline_mae",
        "zero_baseline_rmse", "return_defined_rows",
    ]
    if not left_eval[evaluation_columns].equals(right_eval[evaluation_columns]):
        raise ValueError("the two schemes do not use identical evaluation targets")

    walk_comparison = anchored["walk_summary"].merge(
        window["walk_summary"], on="walk", suffixes=("_anchored", "_window")
    )
    walk_comparison["task_train_row_ratio_window_to_anchored"] = (
        walk_comparison["task_train_rows_window"]
        / walk_comparison["task_train_rows_anchored"]
    )
    walk_comparison.to_csv(report_dir / "walk_comparison.csv", index=False)
    left_eval.to_csv(report_dir / "shared_evaluation_target_summary.csv", index=False)

    compact_columns = [
        "walk",
        "train_start_fraction_anchored", "train_start_fraction_window",
        "cutoff_fraction_anchored", "evaluation_end_fraction_anchored",
        "task_train_rows_anchored", "task_train_rows_window",
        "task_train_contracts_anchored", "task_train_contracts_window",
        "evaluation_rows_anchored", "evaluation_contracts_anchored",
        "task_train_row_ratio_window_to_anchored",
    ]
    compact = walk_comparison[compact_columns].copy()
    compact.rename(columns={
        "cutoff_fraction_anchored": "cutoff_fraction",
        "evaluation_end_fraction_anchored": "evaluation_end_fraction",
        "evaluation_rows_anchored": "evaluation_rows",
        "evaluation_contracts_anchored": "evaluation_contracts",
    }, inplace=True)

    lines = [
        "# Phase 4 Top-80 Walk-Forward Data Analysis",
        "",
        "Status: exploratory data feasibility analysis; no model was trained.",
        "",
        "## Design",
        "",
        "The anchored and fixed-relative-length analyses use identical lifecycle evaluation "
        "fractions inside every contract. They differ only in whether the relative training "
        "start remains at zero or advances by one evaluation fraction per walk.",
        "",
        markdown_table(compact),
        "",
        "## Shared evaluation-target properties",
        "",
        markdown_table(left_eval),
        "",
        "Because the evaluation identities are hash-identical, descriptive differences between "
        "the two schemes arise from the relative training allocation rather than different "
        "evaluation targets.",
        "",
        "## Interpretation boundaries",
        "",
        "- Probability movement is the signed probability-point change `close[t+2] - close[t]`.",
        "- Return is the arithmetic ratio of that movement to `close[t]`; it is undefined at zero "
        "  and can become extreme near zero.",
        "- Labels use `DOWN < -0.005`, `STABLE` within `[-0.005, 0.005]`, and `UP > 0.005`.",
        "- Training targets must fall strictly before each contract-relative cutoff.",
        "- Relative positions use each contract's final observed length, which is retrospective "
        "  unless its termination boundary was known at decision time.",
        "- Relative cutoffs map to different calendar dates across contracts. These results "
        "  characterize lifecycle structure and are not a no-leak pooled deployment backtest.",
        "",
        "Individual reports:",
        "",
        "- [Contract-relative anchored analysis](../relative_anchored/report.md)",
        "- [Contract-relative fixed-length analysis](../relative_window/report.md)",
        "",
    ]
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "relative_anchored_artifact_hashes_sha256": sha256_file(
            root / "relative_anchored/artifact_hashes.json"
        ),
        "relative_window_artifact_hashes_sha256": sha256_file(
            root / "relative_window/artifact_hashes.json"
        ),
        "evaluation_identity_contract": "identical walk, rows, contracts, and target summaries",
        "model_training_launched": False,
    }
    (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return report_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = run(args.root, args.overwrite)
        print(f"Completed Phase 4 walk data comparison: {output}")
        return 0
    except Exception as exc:
        print(f"Phase 4 walk data comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
