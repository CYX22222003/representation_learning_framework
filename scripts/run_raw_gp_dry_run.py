"""Run a constrained, train-only GP search over causal raw-OHLCV terminals."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from alpha.formula import Expression
from alpha.gp import GPConfig, evolve_formulae
from alpha.raw_ohlcv import FORMULA_NAMES, build_raw_ohlcv_formulae, score_cross_sectional_factor
from data_processing.file_list import list_top_k


def _load_train_only_panel(timeframe: str, top_k: int) -> pd.DataFrame:
    parts = []
    for filename, _ in list_top_k(timeframe, top_k):
        frame = pd.read_feather(os.path.join(ROOT, "data", filename))
        train_end, discovery_end = int(len(frame) * .8), int(len(frame) * .6)
        values = build_raw_ohlcv_formulae(frame.iloc[:train_end].copy())
        values["phase"] = np.where(np.arange(len(values)) < discovery_end, "discovery", "confirmation")
        values.loc[[discovery_end - 1, train_end - 1], "forward_return_1"] = np.nan
        parts.append(values)
    return pd.concat(parts, ignore_index=True)


def _evaluate(panel: pd.DataFrame, expression: Expression, min_assets: int) -> dict:
    values = expression.evaluate({name: panel[name].to_numpy() for name in FORMULA_NAMES})
    candidate = panel.loc[:, ["date", "forward_return_1"]].copy()
    candidate["factor"] = values
    score = score_cross_sectional_factor(candidate, "factor", min_assets=min_assets)
    return {"mean_ic": score.mean_ic, "mean_rank_ic": score.mean_rank_ic,
            "rank_ic_ir": score.rank_ic_ir, "spread": score.mean_top_minus_bottom_return,
            "n_dates": score.n_dates, "block_rank_ic": list(score.block_rank_ic)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", choices=("1h", "4h", "1d"), default="4h")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-assets", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="experiments/alpha/raw_gp_4h_top50_dry_run")
    args = parser.parse_args()
    panel = _load_train_only_panel(args.timeframe, args.top_k)
    discovery = panel.loc[panel.phase == "discovery"].reset_index(drop=True)
    confirmation = panel.loc[panel.phase == "confirmation"].reset_index(drop=True)
    expressions: dict[str, Expression] = {}

    def fitness(expression: Expression) -> float:
        name = expression.to_string()
        expressions[name] = expression
        try:
            return abs(_evaluate(discovery, expression, args.min_assets)["mean_rank_ic"])
        except ValueError:
            # Degenerate GP trees such as x-x have no cross-sectional variance.
            return 0.0

    config = GPConfig(population_size=args.population_size, generations=args.generations,
                      max_depth=args.max_depth, elite_count=4, seed=args.seed)
    evolution = evolve_formulae(FORMULA_NAMES, fitness, config)
    ranked = evolution["ranked"]
    chosen = []
    discovery_terminal_values = {key: discovery[key].to_numpy() for key in FORMULA_NAMES}
    def has_binary_operator(expression: Expression) -> bool:
        return expression.op in {"add", "sub", "mul", "div"} or any(has_binary_operator(child) for child in expression.children)
    for name in ranked:
        expression = expressions[name]
        discovery_result = _evaluate(discovery, expression, args.min_assets)
        if not has_binary_operator(expression):
            continue
        candidate_values = expression.evaluate(discovery_terminal_values)
        def non_redundant(other: str) -> bool:
            other_values = expressions[other].evaluate(discovery_terminal_values)
            mask = np.isfinite(candidate_values) & np.isfinite(other_values)
            correlation = np.corrcoef(candidate_values[mask], other_values[mask])[0, 1] if mask.sum() > 1 else 1.0
            return abs(correlation) < .9
        if all(non_redundant(other) for other in chosen):
            direction = 1 if discovery_result["mean_rank_ic"] >= 0 else -1
            values = direction * expression.evaluate({key: confirmation[key].to_numpy() for key in FORMULA_NAMES})
            oriented = confirmation.loc[:, ["date", "forward_return_1"]].copy(); oriented["factor"] = values
            check = score_cross_sectional_factor(oriented, "factor", min_assets=args.min_assets)
            chosen.append(name)
            expressions[name] = expression
            if len(chosen) == 3:
                break
    results = {"protocol": "raw_ohlcv_gp_discovery_confirmation_v1", "global_test_used": False,
               "gp_config": config.__dict__, "terminals": list(FORMULA_NAMES), "history": evolution["history"],
               "selected": []}
    for name in chosen:
        expression = expressions[name]; discovery_result = _evaluate(discovery, expression, args.min_assets)
        direction = 1 if discovery_result["mean_rank_ic"] >= 0 else -1
        values = direction * expression.evaluate({key: confirmation[key].to_numpy() for key in FORMULA_NAMES})
        oriented = confirmation.loc[:, ["date", "forward_return_1"]].copy(); oriented["factor"] = values
        check = score_cross_sectional_factor(oriented, "factor", min_assets=args.min_assets)
        results["selected"].append({"formula": name, "depth": expression.depth,
            "discovery_rank_ic": discovery_result["mean_rank_ic"], "direction": "long_high" if direction == 1 else "long_low",
            "confirmation_rank_ic": check.mean_rank_ic, "confirmation_ic": check.mean_ic,
            "confirmation_spread": check.mean_top_minus_bottom_return, "confirmation_dates": check.n_dates})
    out_dir = os.path.join(ROOT, args.out_dir); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    lines = ["# Raw-OHLCV GP dry run", "", "Train-only constrained GP search; not a backtest.", "",
             "| Formula | Discovery RankIC | Direction | Confirmation RankIC | IC | Top-bottom return |", "|---|---:|---|---:|---:|---:|"]
    for row in results["selected"]:
        lines.append(f"| `{row['formula']}` | {row['discovery_rank_ic']:.4f} | {row['direction']} | {row['confirmation_rank_ic']:.4f} | {row['confirmation_ic']:.4f} | {row['confirmation_spread']:.6f} |")
    lines += ["", "Global test rows were sliced away before terminal construction. Formula selection used discovery only; confirmation is chronological and fixed-formula. Costs, spreads, liquidity and a fresh final holdout remain required."]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": out_dir, "selected": results["selected"], "global_test_used": False}, indent=2))


if __name__ == "__main__":
    main()
