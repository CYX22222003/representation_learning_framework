from __future__ import annotations

import unittest

import numpy as np

from training.phase5_regression_sensitivities import (
    RegressionSensitivityConfig,
    apply_target_transform,
    derive_scientific_target,
    fit_target_transform,
    invert_target_transform,
    reconstruct_probability,
)


class Phase5RegressionSensitivityTests(unittest.TestCase):
    def test_raw_delta_and_log_return_targets(self) -> None:
        current = np.asarray([0.1, 0.5])
        future = np.asarray([0.2, 0.4])
        np.testing.assert_allclose(
            derive_scientific_target("raw_delta_h8", current, future), [0.1, -0.1]
        )
        log_return = derive_scientific_target("log_return_h2", current, future)
        np.testing.assert_allclose(log_return, np.log(future / current))
        np.testing.assert_allclose(
            reconstruct_probability("log_return_h2", current, log_return), future
        )

    def test_log_return_rejects_nonpositive_prices(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            derive_scientific_target(
                "log_return_h2", np.asarray([0.0]), np.asarray([0.1])
            )

    def test_target_transforms_round_trip(self) -> None:
        for task, target in (
            ("raw_delta_h8", np.asarray([-0.02, 0.0, 0.03])),
            ("log_return_h2", np.asarray([-0.4, 0.1, 0.3])),
        ):
            with self.subTest(task=task):
                transform = fit_target_transform(task, target)
                encoded = apply_target_transform(target, transform)
                decoded = invert_target_transform(encoded, transform)
                np.testing.assert_allclose(decoded, target, rtol=1e-6, atol=1e-7)

    def test_frozen_config_rejects_changed_seed_or_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed0"):
            RegressionSensitivityConfig("raw_delta_h8", 1, seed=1)
        with self.assertRaisesRegex(ValueError, "50 epochs"):
            RegressionSensitivityConfig("log_return_h2", 1, epochs=15)


if __name__ == "__main__":
    unittest.main()
