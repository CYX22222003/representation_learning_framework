"""Run a train-only, Alpha101-style raw-OHLCV factor dry run.

The global per-contract test partition is never loaded into this experiment.
Each contract's original 80% training portion is divided chronologically into
60% discovery and 20% confirmation.  Formula direction and selection are
fixed from discovery; confirmation is reported without refitting.  This is a
factor diagnostic, not a cost-aware or executable trading backtest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from alpha.raw_ohlcv import FORMULA_NAMES, build_raw_ohlcv_formulae, score_cross_sectional_factor
from data_processing.file_list import list_top_k


def _load_panel(timeframe: str, top_k: int) -> pd.DataFrame:
    parts = []
    for filename, _ in list_top_k(timeframe, top_k):
        frame = pd.read_feather(os.path.join(ROOT, "data", filename))
        n_source = len(frame)
        train_end = int(n_source * 0.8)
        discovery_end = int(n_source * 0.6)
        # Slice before feature construction, so the locked source rows are not
        # even read by a rolling operator in this train-only diagnostic.
        values = build_raw_ohlcv_formulae(frame.iloc[:train_end].copy())
        n = len(values)
        values["contract"] = filename
        values["phase"] = np.where(np.arange(n) < discovery_end, "discovery", "confirmation")
        # Never allow the last source row of either train-only segment to use a
        # target from the following segment.
        values.loc[[max(0, discovery_end - 1), max(0, train_end - 1)], "forward_return_1"] = np.nan
        parts.append(values)
    return pd.concat(parts, ignore_index=True)


def _score(panel: pd.DataFrame, formula: str, min_assets: int) -> dict:
    score = score_cross_sectional_factor(panel, formula, min_assets=min_assets)
    return asdict(score)


def _markdown(results: dict) -> str:
    lines = [
        "# Raw-OHLCV alpha dry run",
        "",
        "This is a training-only factor diagnostic, not a profitability claim or cost-aware backtest.",
        "The original per-contract 20% test partition was not used. Each original training partition was split chronologically into 60% discovery and 20% confirmation.",
        "",
        "## Protocol",
        "",
        f"- Universe: top {results['top_k']} raw Polymarket {results['timeframe']} contracts by existing file-size activity proxy.",
        "- Target: next-bar close-to-close return; all factors use same-bar or earlier OHLCV only.",
        "- Selection: top three predeclared formulae by absolute discovery RankIC; direction is frozen from discovery.",
        "- Evaluation: timestamp-level cross-sectional IC/RankIC and equal-weight top-minus-bottom quintile return, without costs.",
        "",
        "## Results",
        "",
        "| Formula | Discovery RankIC | Direction | Confirmation RankIC | Confirmation IC | Top-bottom return | Dates |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for formula, row in results["formula_results"].items():
        confirmation = row["confirmation"]
        lines.append(
            f"| {formula} | {row['discovery']['mean_rank_ic']:.4f} | {row['direction']} | "
            f"{confirmation['oriented_mean_rank_ic']:.4f} | {confirmation['oriented_mean_ic']:.4f} | "
            f"{confirmation['oriented_mean_top_minus_bottom_return']:.6f} | {confirmation['oriented_n_dates']} |"
        )
    lines += ["", "## Selected for a future fresh holdout", ""]
    lines += [f"- `{name}`" for name in results["selected"]]
    lines += [
        "",
        "A selected result is only a candidate. Before any trading claim, evaluate these frozen formulae on a new later period and use contract-aware execution, liquidity, spread, and fee assumptions.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", default="4h", choices=("1h", "4h", "1d"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-assets", type=int, default=10)
    parser.add_argument("--out-dir", default="experiments/alpha/raw_ohlcv_4h_top50_dry_run")
    args = parser.parse_args()
    if args.top_k < args.min_assets:
        raise ValueError("top-k must be at least min-assets")

    panel = _load_panel(args.timeframe, args.top_k)
    discovery = panel.loc[panel["phase"] == "discovery"]
    confirmation = panel.loc[panel["phase"] == "confirmation"]
    discovery_scores = {formula: _score(discovery, formula, args.min_assets) for formula in FORMULA_NAMES}
    selected = sorted(FORMULA_NAMES, key=lambda name: abs(discovery_scores[name]["mean_rank_ic"]), reverse=True)[:3]
    formula_results = {}
    for formula in FORMULA_NAMES:
        direction = 1 if discovery_scores[formula]["mean_rank_ic"] >= 0 else -1
        oriented = confirmation.copy()
        oriented["oriented_factor"] = direction * oriented[formula]
        confirmation_score = _score(oriented, "oriented_factor", args.min_assets)
        confirmation_score = {f"oriented_{key}": value for key, value in confirmation_score.items()}
        formula_results[formula] = {
            "discovery": discovery_scores[formula],
            "direction": "long_high" if direction == 1 else "long_low",
            "confirmation": confirmation_score,
            "selected": formula in selected,
        }

    results = {
        "protocol": "train_only_discovery_confirmation_v1",
        "timeframe": args.timeframe,
        "top_k": args.top_k,
        "min_assets": args.min_assets,
        "formulae": list(FORMULA_NAMES),
        "selected": selected,
        "formula_results": formula_results,
        "rows_by_phase": panel["phase"].value_counts().to_dict(),
        "global_test_used": False,
    }
    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(_markdown(results))
    print(json.dumps({"out_dir": out_dir, "selected": selected, "global_test_used": False}, indent=2))


if __name__ == "__main__":
    main()
