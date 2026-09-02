"""Exploratory, train-only GP screen directly over saved representation coordinates.

This deliberately tests the user's hypothesis that representation coordinates may
themselves contain alpha. It is not the project's primary alpha protocol: these
coordinates are not economically interpretable, and a positive result requires
fresh-holdout confirmation before any claim.
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__)); sys.path.insert(0, os.path.join(ROOT, "src"))
from alpha.gp import GPConfig, evolve_formulae
from alpha.formula import Expression
from alpha.raw_ohlcv import score_cross_sectional_factor
from alpha.raw_ohlcv import FORMULA_NAMES, build_raw_ohlcv_formulae
from data_processing.file_list import list_top_k
from features.feature_store import NpzFeatureStore


def _panel_and_terminals(feature_path: str, top_k: int, include_raw_ohlcv: bool, all_coordinates: bool, deep_only: bool):
    bundle = NpzFeatureStore(feature_path).load().as_branch_dict()
    with np.load(f"{feature_path}.index.npz") as index:
        n_train = int(index["train_size"])
    # No target-based pre-screening.  The exhaustive mode exposes every saved
    # coordinate to GP; the small lattice remains useful for quick smoke tests.
    selected_branches = ("vae", "contrastive", "byol") if deep_only else tuple(bundle)
    positions = ({name: tuple(range(bundle[name].shape[1])) for name in selected_branches}
                 if all_coordinates else
                 {"statistical": (0, 23, 46, 69), "transformed": (0, 18, 36, 54),
                  "vae": (0, 21, 42, 63), "contrastive": (0, 42, 85, 127), "byol": (0, 42, 85, 127)})
    terminals = {f"{branch}_{i}": np.asarray(bundle[branch][:n_train, i], dtype=float)
                 for branch, indices in positions.items() for i in indices}
    parts = []; raw_parts = {name: [] for name in FORMULA_NAMES}; expected = 0
    for filename, _ in list_top_k("4h", top_k):
        raw = pd.read_feather(os.path.join(ROOT, "data", filename))
        n_seq = len(raw) - 64; train_end = int(.8 * n_seq); discovery_end = int(.6 * n_seq)
        if include_raw_ohlcv:
            # Feature row i represents raw bars i..i+63, so raw terminals must
            # be read at bar i+63—not at the feature-row ordinal i.
            raw_formulae = build_raw_ohlcv_formulae(raw.iloc[:train_end + 63].copy())
            for name in FORMULA_NAMES:
                raw_parts[name].append(raw_formulae[name].iloc[63:63 + train_end].to_numpy(dtype=float))
        values = pd.DataFrame({"date": raw["date"].iloc[63:63 + train_end].to_numpy(),
                               "forward_return_1": (raw["close"].iloc[64:64 + train_end].to_numpy() /
                                                    raw["close"].iloc[63:63 + train_end].to_numpy()) - 1.0,
                               "phase": np.where(np.arange(train_end) < discovery_end, "discovery", "confirmation")})
        values.loc[[discovery_end - 1, train_end - 1], "forward_return_1"] = np.nan
        parts.append(values); expected += train_end
    if expected != n_train:
        raise ValueError(f"Reconstructed {expected} train rows but bundle contains {n_train}; feature ordering cannot be trusted")
    panel = pd.concat(parts, ignore_index=True)
    if include_raw_ohlcv:
        terminals.update({f"ohlcv_{name}": np.concatenate(values) for name, values in raw_parts.items()})
    discovery = panel.phase.eq("discovery").to_numpy()
    # Fit coordinate scales on discovery only; arithmetic formulae then have a
    # stable, dimension-comparable scale in confirmation.
    for name, values in terminals.items():
        mean, std = values[discovery].mean(), values[discovery].std()
        terminals[name] = (values - mean) / (std if std > 1e-6 else 1.0)
    return panel, terminals


def _score(panel, values, min_assets):
    candidate = panel[["date", "forward_return_1"]].copy(); candidate["factor"] = values
    result = score_cross_sectional_factor(candidate, "factor", min_assets=min_assets)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-npz", default="data/features/features_4h_seq64_top50_phase1.npz")
    parser.add_argument("--top-k", type=int, default=50); parser.add_argument("--min-assets", type=int, default=10)
    parser.add_argument("--include-raw-ohlcv", action="store_true")
    parser.add_argument("--all-representation-features", action="store_true")
    parser.add_argument("--deep-learning-only", action="store_true", help="exclude statistical and transformed coordinates")
    parser.add_argument("--population-size", type=int, default=48); parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=4); parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out-dir", default="experiments/alpha/representation_gp_4h_top50_dry_run")
    args = parser.parse_args(); panel, terminals = _panel_and_terminals(args.features_npz, args.top_k, args.include_raw_ohlcv, args.all_representation_features, args.deep_learning_only)
    discovery = panel[panel.phase == "discovery"].reset_index(drop=True); confirmation = panel[panel.phase == "confirmation"].reset_index(drop=True)
    discovery_terms = {name: values[panel.phase.eq("discovery").to_numpy()] for name, values in terminals.items()}
    confirmation_terms = {name: values[panel.phase.eq("confirmation").to_numpy()] for name, values in terminals.items()}
    expressions = {}
    def fitness(expr):
        expressions[expr.to_string()] = expr
        try: return abs(_score(discovery, expr.evaluate(discovery_terms), args.min_assets).mean_rank_ic)
        except ValueError: return 0.0
    config = GPConfig(population_size=args.population_size, generations=args.generations,
                      max_depth=args.max_depth, elite_count=max(4, args.population_size // 8), seed=args.seed,
                      seed_all_terminals=args.all_representation_features)
    evolved = evolve_formulae(tuple(terminals), fitness, config)
    selected = []; selected_values = []
    for name in evolved["ranked"]:
        expr = expressions[name]
        if not any(node in name for node in (" add ", " sub ", " mul ", " div ")): continue
        try: discovery_score = _score(discovery, expr.evaluate(discovery_terms), args.min_assets)
        except ValueError: continue
        values_discovery = expr.evaluate(discovery_terms)
        if any(abs(np.corrcoef(values_discovery[np.isfinite(values_discovery) & np.isfinite(other)],
                               other[np.isfinite(values_discovery) & np.isfinite(other)])[0, 1]) >= .9
               for other in selected_values):
            continue
        direction = 1 if discovery_score.mean_rank_ic >= 0 else -1
        confirmation_score = _score(confirmation, direction * expr.evaluate(confirmation_terms), args.min_assets)
        selected.append({"formula": name, "depth": expr.depth, "discovery_rank_ic": discovery_score.mean_rank_ic,
                         "direction": "long_high" if direction == 1 else "long_low", "confirmation_rank_ic": confirmation_score.mean_rank_ic,
                         "confirmation_ic": confirmation_score.mean_ic, "confirmation_spread": confirmation_score.mean_top_minus_bottom_return,
                         "confirmation_dates": confirmation_score.n_dates})
        selected_values.append(values_discovery)
        if len(selected) == 3: break
    out_dir = os.path.join(ROOT, args.out_dir); os.makedirs(out_dir, exist_ok=True)
    payload = {"protocol": "exploratory_representation_ohlcv_gp_v1", "global_test_used": False,
               "coordinate_selection": "all_saved_deep_coordinates" if args.deep_learning_only else ("all_saved_coordinates" if args.all_representation_features else "fixed_spread_across_branches_no_target_prescreen"), "terminals": list(terminals),
               "include_raw_ohlcv": args.include_raw_ohlcv,
               "gp_config": config.__dict__, "history": evolved["history"], "selected": selected}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as handle: json.dump(payload, handle, indent=2)
    coordinate_scope = "all saved deep-learning coordinates" if args.deep_learning_only else ("all saved representation coordinates" if args.all_representation_features else "a fixed representation-coordinate lattice")
    lines = ["# Representation + OHLCV GP dry run", "", f"Exploratory only: {coordinate_scope} and causal OHLCV terminals are GP inputs.", "", "| Formula | Discovery RankIC | Direction | Confirmation RankIC | IC | Spread |", "|---|---:|---|---:|---:|---:|"]
    lines += [f"| `{x['formula']}` | {x['discovery_rank_ic']:.4f} | {x['direction']} | {x['confirmation_rank_ic']:.4f} | {x['confirmation_ic']:.4f} | {x['confirmation_spread']:.6f} |" for x in selected]
    lines += ["", "The global test partition was not used. This is not an interpretable or tradeable alpha claim; it is a direct empirical check of representation coordinates."]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as handle: handle.write("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": out_dir, "selected": selected, "global_test_used": False}, indent=2))
if __name__ == "__main__": main()
