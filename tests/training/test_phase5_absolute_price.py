from __future__ import annotations

import unittest

import numpy as np
import torch

from training.phase5_absolute_price import (
    AbsolutePriceConfig,
    AbsolutePriceRegressor,
    price_breakdowns,
    relative_skill,
)


class Phase5AbsolutePriceTests(unittest.TestCase):
    def test_sigmoid_head_outputs_valid_probabilities(self) -> None:
        prediction = AbsolutePriceRegressor()(torch.randn(7, 445))
        self.assertEqual(tuple(prediction.shape), (7, 1))
        self.assertTrue(torch.all((prediction >= 0.0) & (prediction <= 1.0)))

    def test_price_breakdowns_and_skill(self) -> None:
        target = np.asarray([0.1, 0.2, 0.8, 0.9])
        prediction = np.asarray([0.11, 0.19, 0.79, 0.91])
        metadata = {
            "current_close": np.asarray([0.09, 0.18, 0.81, 0.91]),
            "context_imputed_rows": np.asarray([0, 1, 0, 1]),
            "lifecycle_stage": np.asarray([0, 1, 2, 2]),
            "condition_ids": np.asarray(["a", "a", "b", "b"]),
        }
        metrics, per_contract = price_breakdowns(prediction, target, metadata)
        self.assertEqual(metrics["overall"]["count"], 4)
        self.assertEqual(set(per_contract), {"a", "b"})
        skill = relative_skill(
            {"mae": 0.1, "mse": 0.01, "rmse": 0.1},
            {"mae": 0.2, "mse": 0.04, "rmse": 0.2},
        )
        self.assertAlmostEqual(skill["mae"], 0.5)
        self.assertAlmostEqual(skill["mse"], 0.75)

    def test_frozen_config_rejects_changed_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "50 epochs"):
            AbsolutePriceConfig(walk=1, epochs=15)


if __name__ == "__main__":
    unittest.main()
