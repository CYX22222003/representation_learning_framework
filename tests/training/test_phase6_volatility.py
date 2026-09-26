from __future__ import annotations

import unittest

import numpy as np
import torch

from tasks.volatility_prediction import VolatilityRegressor
from training.phase6_volatility import (
    MODELS,
    Phase6VolatilityConfig,
    historical_persistence,
    smoke_test_volatility_models,
    volatility_metrics,
)


class Phase6VolatilityTrainingTests(unittest.TestCase):
    def test_active_round_excludes_deferred_garch_lstm(self) -> None:
        self.assertEqual(MODELS, ("framework_h0", "raw_ohlcv_mlp", "raw_lstm"))

    def test_positive_head_and_all_frozen_widths_smoke(self) -> None:
        result = smoke_test_volatility_models()
        self.assertTrue(result["valid"])
        for width in (445, 573):
            output = VolatilityRegressor(width)(torch.randn(4, width))
            self.assertTrue(torch.all(output >= 0.0))

    def test_historical_persistence_uses_last_eight_changes(self) -> None:
        sequences = np.zeros((2, 64, 5), dtype=np.float32)
        sequences[:, :, 3] = np.arange(64, dtype=np.float32)
        np.testing.assert_allclose(historical_persistence(sequences), [8.0, 8.0])

    def test_metrics_reject_negative_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            volatility_metrics(np.asarray([-1.0]), np.asarray([0.0]))

    def test_frozen_config_rejects_budget_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "50 epochs"):
            Phase6VolatilityConfig(model="framework_h0", walk=1, epochs=49)


if __name__ == "__main__":
    unittest.main()
