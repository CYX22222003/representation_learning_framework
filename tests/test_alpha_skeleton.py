import unittest
import numpy as np

from alpha.primitives import build_alpha_primitives
from alpha.oof import expanding_folds, generate_oof_predictions
from alpha.formula import Expression
from alpha.scoring import score_factor


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


if __name__ == "__main__":
    unittest.main()
