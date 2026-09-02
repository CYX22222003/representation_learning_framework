import unittest
import numpy as np

from alpha.primitives import build_alpha_primitives
from alpha.oof import expanding_folds, generate_oof_predictions
from alpha.formula import Expression
from alpha.scoring import score_factor
from alpha.raw_ohlcv import build_raw_ohlcv_formulae, score_cross_sectional_factor
from alpha.gp import GPConfig, evolve_formulae
import pandas as pd


class AlphaSkeletonTests(unittest.TestCase):
    def test_primitives_align_and_add_confidence(self):
        t = build_alpha_primitives(price_prediction=np.arange(4),
                                   volatility_prediction=np.ones(4),
                                   trend_probabilities=np.tile([.6, .3, .1], (4, 1)))
        self.assertEqual(t.n_rows, 4)
        self.assertIn("trend_direction_margin", t.values)

    def test_oof_is_strictly_prior(self):
        folds = list(expanding_folds(20, n_folds=4, min_train_size=3))
        self.assertTrue(all(train[-1] < pred[0] for train, pred in folds))
        result = generate_oof_predictions(20, lambda tr, te: np.full(len(te), tr[-1]),
                                           n_folds=4, min_train_size=3)
        self.assertEqual(result.predictions.shape, (20,))

    def test_protected_formula_and_score(self):
        x = np.array([1., 2., 3., 4.]); y = x * 2
        expr = Expression("div", (Expression("primitive", name="x"),
                                    Expression("primitive", name="zero")))
        self.assertTrue(np.all(np.isfinite(expr.evaluate({"x": x, "zero": np.zeros(4)}))))
        self.assertAlmostEqual(score_factor(x, y).rank_ic, 1.0)

    def test_raw_ohlcv_formulae_are_causal_and_cross_sectional(self):
        dates = pd.date_range("2025-01-01", periods=24, freq="4h")
        rows = []
        for contract, offset in (("a", 0.0), ("b", 0.1), ("c", 0.2)):
            source = pd.DataFrame({"date": dates, "open": 0.3 + offset + np.arange(24) * .001,
                                   "high": 0.32 + offset + np.arange(24) * .001,
                                   "low": 0.29 + offset + np.arange(24) * .001,
                                   "close": 0.31 + offset + np.arange(24) * .001,
                                   "volume": np.arange(1, 25)})
            built = build_raw_ohlcv_formulae(source)
            built["contract"] = contract
            rows.append(built)
        panel = pd.concat(rows)
        self.assertTrue(np.isnan(panel.iloc[-1]["forward_return_1"]))
        score = score_cross_sectional_factor(panel, "reversal_1", min_assets=3)
        self.assertGreater(score.n_dates, 0)

        changed_future = source.copy()
        changed_future.loc[23, "volume"] = 1_000_000
        original_values = build_raw_ohlcv_formulae(source)
        changed_values = build_raw_ohlcv_formulae(changed_future)
        self.assertAlmostEqual(original_values.loc[22, "volume_weighted_momentum_3"],
                               changed_values.loc[22, "volume_weighted_momentum_3"])

    def test_gp_evolves_bounded_formulae(self):
        result = evolve_formulae(("x", "y"),
                                 lambda expression: len(expression.to_string()),
                                 GPConfig(population_size=12, generations=3, elite_count=2, seed=1))
        self.assertEqual(len(result["history"]), 3)
        self.assertTrue(result["ranked"])


if __name__ == "__main__":
    unittest.main()
