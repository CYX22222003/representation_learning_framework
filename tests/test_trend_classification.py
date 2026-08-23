from __future__ import annotations

import unittest

import torch

from evaluation.metrics import multiclass_classification_metrics
from tasks.trend_classification import TrendClassifier


class TrendClassificationTests(unittest.TestCase):
    def test_trend_classifier_supports_triclass_logits(self) -> None:
        model = TrendClassifier(input_dim=8, hidden_dim=12, n_classes=3)
        logits = model(torch.randn(5, 8))

        self.assertEqual(logits.shape, (5, 3))

    def test_multiclass_metrics_report_macro_f1_and_confusion_matrix(self) -> None:
        logits = torch.tensor(
            [
                [4.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 2.0, 1.0],
                [0.0, 0.0, 5.0],
            ]
        )
        labels = torch.tensor([0, 1, 2, 2])

        metrics = multiclass_classification_metrics(logits, labels, n_classes=3)

        self.assertAlmostEqual(float(metrics["accuracy"]), 0.75)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 1, 1]])
        self.assertEqual(len(metrics["per_class"]), 3)


if __name__ == "__main__":
    unittest.main()
