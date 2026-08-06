from __future__ import annotations

import unittest

import numpy as np
import torch

from baselines.ginn_baseline.ginn_model import (
    GinnTensorDataset,
    LSTMVariancePredictor,
    evaluate_model,
    garch_fused_loss,
    make_train_loader,
    train_one_epoch,
)


class GinnModelTests(unittest.TestCase):
    def test_lstm_maps_sequences_to_single_prediction(self):
        model = LSTMVariancePredictor(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
        out = model(torch.zeros(3, 64, 5))
        self.assertEqual(tuple(out.shape), (3, 1))

    def test_softplus_output_transform_constrains_predictions(self):
        model = LSTMVariancePredictor(
            input_size=5,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            output_transform="softplus",
        )
        out = model(torch.zeros(3, 64, 5))
        self.assertTrue(torch.all(out >= 0.0))

    def test_unknown_output_transform_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "output_transform"):
            LSTMVariancePredictor(output_transform="relu")

    def test_fused_loss_reports_all_components_and_extreme_lambdas(self):
        pred = torch.tensor([[1.0], [3.0]])
        y_gt = torch.tensor([[2.0], [2.0]])
        y_garch = torch.tensor([[1.0], [1.0]])
        gt_only = garch_fused_loss(pred, y_gt, y_garch, lambda_garch=0.0)
        garch_only = garch_fused_loss(pred, y_gt, y_garch, lambda_garch=1.0)
        self.assertAlmostEqual(gt_only["total"].item(), 1.0)
        self.assertAlmostEqual(gt_only["gt_mse"].item(), 1.0)
        self.assertAlmostEqual(garch_only["total"].item(), 2.0)
        self.assertAlmostEqual(garch_only["garch_mse"].item(), 2.0)

    def test_train_loader_drops_singleton_last_batch(self):
        X = np.zeros((5, 64, 5), dtype=np.float32)
        y = np.ones((5, 1), dtype=np.float32)
        dataset = GinnTensorDataset(X, y, y)
        loader = make_train_loader(dataset, batch_size=4, seed=0)
        batch_sizes = [batch[0].shape[0] for batch in loader]
        self.assertEqual(batch_sizes, [4])

    def test_training_epoch_updates_parameters_and_returns_finite_components(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, 64, 5)).astype(np.float32)
        y_gt = np.abs(rng.normal(0.2, 0.02, size=(8, 1))).astype(np.float32)
        y_garch = np.abs(rng.normal(0.18, 0.02, size=(8, 1))).astype(np.float32)
        model = LSTMVariancePredictor(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
        before = [p.detach().clone() for p in model.parameters()]
        loader = make_train_loader(GinnTensorDataset(X, y_gt, y_garch), batch_size=4, seed=1)
        losses = train_one_epoch(
            model,
            loader,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            device=torch.device("cpu"),
            lambda_garch=0.3,
        )
        self.assertEqual(set(losses), {"total", "gt_mse", "garch_mse"})
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_evaluate_model_returns_metrics_predictions_and_undefined_correlation(self):
        X = np.zeros((3, 64, 5), dtype=np.float32)
        targets = np.full((3, 1), 0.5, dtype=np.float32)
        garch_targets = np.full((3, 1), 0.4, dtype=np.float32)
        ids = np.array([0, 0, 1], dtype=np.int32)
        model = LSTMVariancePredictor(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        result = evaluate_model(model, X, targets, garch_targets, ids, torch.device("cpu"), batch_size=2)
        self.assertEqual(result["predictions"].shape, (3,))
        self.assertEqual(result["targets"].shape, (3,))
        self.assertAlmostEqual(result["metrics"]["mse"], 0.25)
        self.assertIsNone(result["metrics"]["pearson_corr"])
        self.assertEqual(result["metrics"]["negative_prediction_count"], 0)
        self.assertIn("per_contract", result["metrics"])


if __name__ == "__main__":
    unittest.main()
