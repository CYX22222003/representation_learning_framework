from __future__ import annotations

import unittest

import numpy as np

from training.phase5_downstream import (
    apply_feature_standardizer,
    fit_feature_standardizer,
    regression_breakdowns,
    smoke_test_heads,
)


class Phase5DownstreamTests(unittest.TestCase):
    def test_scaler_is_train_only_and_handles_constant_columns(self) -> None:
        rng = np.random.default_rng(3)
        train = rng.normal(size=(20, 445)).astype(np.float32)
        train[:, 9] = 4.0
        scaler = fit_feature_standardizer(train)
        self.assertEqual(float(scaler["scale"][9]), 1.0)
        evaluation = np.full((5, 445), 1e6, dtype=np.float32)
        transformed = apply_feature_standardizer(evaluation, scaler)
        self.assertTrue(np.all(transformed <= 10.0))
        self.assertTrue(np.all(transformed >= -10.0))
        replay = fit_feature_standardizer(train)
        for key in scaler:
            np.testing.assert_array_equal(scaler[key], replay[key])

    def test_regression_breakdowns_include_required_strata(self) -> None:
        target = np.asarray([0.0, 0.002, -0.003, 0.0005])
        prediction = np.asarray([0.0, 0.001, -0.002, -0.0002])
        metadata = {
            "condition_ids": np.asarray(["a", "a", "b", "b"]),
            "context_imputed_rows": np.asarray([0, 1, 0, 1]),
            "lifecycle_stage": np.asarray([0, 0, 1, 2]),
        }
        metrics, per_contract = regression_breakdowns(prediction, target, metadata)
        self.assertEqual(metrics["overall"]["count"], 4)
        self.assertEqual(metrics["exact_zero"]["count"], 1)
        self.assertEqual(metrics["threshold_exceeding"]["count"], 2)
        self.assertEqual(set(per_contract), {"a", "b"})

    def test_both_heads_and_losses_smoke(self) -> None:
        result = smoke_test_heads()
        self.assertTrue(result["valid"])
        self.assertEqual(result["regression_output_shape"], [16, 1])
        self.assertEqual(result["classification_output_shape"], [16, 3])


if __name__ == "__main__":
    unittest.main()
