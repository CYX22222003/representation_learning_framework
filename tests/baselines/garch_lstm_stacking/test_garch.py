from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from baselines.garch_lstm_stacking.garch import GuardedGarchForecaster, aligned_realized_volatility_forecast


class GarchTests(unittest.TestCase):
    def test_aligned_formula_uses_known_returns_plus_next_variance(self) -> None:
        close = np.exp(np.asarray([0.0, 0.1, 0.3, 0.6], dtype=np.float64))
        got = aligned_realized_volatility_forecast(close, 0.16)
        expected = np.sqrt(((0.2**2) + (0.3**2) + 0.16) / 3.0)
        self.assertAlmostEqual(got, expected, places=12)

    def test_future_prices_do_not_change_earlier_forecast(self) -> None:
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0, 0.02, size=120)
        close = np.exp(np.cumsum(np.r_[0.0, returns]))
        forecaster = GuardedGarchForecaster()
        first = forecaster.fit_predict(close, np.array([30]), seq_len=10, fit_row_limit=50).prediction_guarded[0]
        changed = close.copy()
        changed[80:] *= 5.0
        second = forecaster.fit_predict(changed, np.array([30]), seq_len=10, fit_row_limit=50).prediction_guarded[0]
        self.assertAlmostEqual(float(first), float(second), places=10)

    def test_near_static_uses_finite_fallback(self) -> None:
        close = np.ones(80, dtype=np.float64)
        result = GuardedGarchForecaster().fit_predict(close, np.array([0, 1, 2]), seq_len=10, fit_row_limit=20)
        self.assertEqual(result.diagnostics.status, "fallback")
        self.assertEqual(result.diagnostics.fallback_reason, "near_static")
        self.assertTrue(np.all(np.isfinite(result.prediction_guarded)))
        self.assertTrue(np.all(result.fallback_flag))

    def test_optimizer_failure_falls_back(self) -> None:
        def failing_optimizer(*args, **kwargs):
            return SimpleNamespace(success=False, x=np.array([0.1, 0.1, 0.8]))

        close = np.exp(np.cumsum(np.r_[0.0, np.linspace(-0.02, 0.03, 80)]))
        result = GuardedGarchForecaster(optimizer=failing_optimizer).fit_predict(close, np.array([5]), seq_len=10, fit_row_limit=40)
        self.assertEqual(result.diagnostics.status, "fallback")
        self.assertEqual(result.diagnostics.fallback_reason, "optimizer_failed")
        self.assertTrue(np.isfinite(result.prediction_guarded[0]))

    def test_training_only_cap_is_applied(self) -> None:
        rng = np.random.default_rng(1)
        close = np.exp(np.cumsum(np.r_[0.0, rng.normal(0.0, 0.01, 100)]))
        close[70:80] *= np.linspace(1.0, 3.0, 10)
        result = GuardedGarchForecaster(cap_quantile=0.5).fit_predict(close, np.array([65, 66, 67]), seq_len=10, fit_row_limit=40)
        self.assertTrue(np.any(result.capped_flag))
        self.assertTrue(np.all(result.prediction_guarded <= result.diagnostics.volatility_cap + 1e-8))


if __name__ == "__main__":
    unittest.main()
