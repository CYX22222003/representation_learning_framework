from __future__ import annotations

import unittest

import numpy as np

from tasks.phase2_classification.metrics import probabilistic_classification_metrics


class ProbabilisticMetricTests(unittest.TestCase):
    def test_perfect_three_class_predictions(self) -> None:
        labels = np.asarray([0, 1, 2, 0, 1, 2])
        logits = np.full((6, 3), -5.0, dtype=np.float32)
        logits[np.arange(6), labels] = 5.0
        metrics, arrays = probabilistic_classification_metrics(logits, labels, ["DOWN", "STABLE", "UP"])
        self.assertAlmostEqual(float(metrics["accuracy"]), 1.0)
        self.assertAlmostEqual(float(metrics["macro_f1"]), 1.0)
        self.assertAlmostEqual(float(metrics["balanced_accuracy"]), 1.0)
        self.assertAlmostEqual(float(metrics["macro_roc_auc"]), 1.0)
        np.testing.assert_allclose(arrays["probabilities"].sum(axis=1), 1.0)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            probabilistic_classification_metrics(np.zeros((2, 2)), np.asarray([0, 1]), ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
