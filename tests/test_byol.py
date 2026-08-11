from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.byol import BYOLEncoder, byol_loss, byol_prediction_loss
from training.train_byol import train_byol_epoch


class BYOLModelTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        model = BYOLEncoder(input_dim=5, hidden_dim=16, projection_dim=12, predictor_hidden_dim=10)
        x1 = torch.randn(4, 8, 5)
        x2 = torch.randn(4, 8, 5)

        h1, h2, p1, p2, z1_target, z2_target = model(x1, x2)

        self.assertEqual(h1.shape, (4, 16))
        self.assertEqual(h2.shape, (4, 16))
        self.assertEqual(p1.shape, (4, 12))
        self.assertEqual(p2.shape, (4, 12))
        self.assertEqual(z1_target.shape, (4, 12))
        self.assertEqual(z2_target.shape, (4, 12))
        self.assertEqual(model.encode(x1).shape, (4, 16))

    def test_target_parameters_are_frozen(self) -> None:
        model = BYOLEncoder(input_dim=5, hidden_dim=16, projection_dim=12, predictor_hidden_dim=10)

        self.assertTrue(all(parameter.requires_grad for parameter in model.online_backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.online_projector.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.online_predictor.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.target_backbone.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.target_projector.parameters()))

    def test_ema_update_changes_target_parameters(self) -> None:
        model = BYOLEncoder(input_dim=5, hidden_dim=16, projection_dim=12, predictor_hidden_dim=10)
        target_before = [parameter.detach().clone() for parameter in model.target_backbone.parameters()]
        with torch.no_grad():
            for parameter in model.online_backbone.parameters():
                parameter.add_(0.1)

        model.update_target(tau=0.9)

        self.assertTrue(
            any(
                not torch.allclose(before, after)
                for before, after in zip(target_before, model.target_backbone.parameters())
            )
        )

    def test_byol_loss_is_finite(self) -> None:
        p1 = torch.randn(4, 12)
        p2 = torch.randn(4, 12)
        z1 = torch.randn(4, 12)
        z2 = torch.randn(4, 12)

        loss = byol_loss(p1, z2, p2, z1)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_prediction_loss_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            byol_prediction_loss(torch.randn(4, 12), torch.randn(4, 11))

    def test_train_byol_epoch_returns_diagnostics(self) -> None:
        torch.manual_seed(0)
        model = BYOLEncoder(input_dim=5, hidden_dim=16, projection_dim=12, predictor_hidden_dim=10)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        loader = DataLoader(
            TensorDataset(torch.randn(8, 8, 5)),
            batch_size=4,
            shuffle=False,
            drop_last=True,
        )

        metrics = train_byol_epoch(model, loader, optimizer, target_decay=0.9)

        self.assertEqual(set(metrics), {"loss", "view_cosine", "embedding_std", "embedding_norm"})
        for value in metrics.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)).item())


if __name__ == "__main__":
    unittest.main()
