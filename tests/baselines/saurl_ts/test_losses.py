from __future__ import annotations

import unittest

import torch

from baselines.saurl_ts import (
    five_kernel_mmd,
    symmetric_byol_loss,
    temporal_total_variation,
)


class SaURLLossTests(unittest.TestCase):
    def test_mmd_is_symmetric_and_zero_for_identical_inputs(self) -> None:
        torch.manual_seed(20)
        x = torch.randn(5, 7, 3)
        y = torch.randn(5, 7, 3)
        torch.testing.assert_close(five_kernel_mmd(x, y), five_kernel_mmd(y, x))
        self.assertAlmostEqual(float(five_kernel_mmd(x, x)), 0.0, places=6)

    def test_mmd_is_finite_for_repeated_constant_samples(self) -> None:
        x = torch.ones(4, 8, 5, requires_grad=True)
        y = torch.full((4, 8, 5), 2.0, requires_grad=True)
        value = five_kernel_mmd(x, y)
        self.assertTrue(torch.isfinite(value))
        self.assertGreaterEqual(float(value.detach()), 0.0)
        value.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(y.grad).all())

    def test_diversity_objective_has_expected_non_positive_sign(self) -> None:
        x = torch.randn(4, 8, 5)
        y = x + 2.0
        self.assertLessEqual(float(-five_kernel_mmd(x, y)), 1e-6)

    def test_temporal_total_variation_is_zero_for_constant_mask(self) -> None:
        mask = torch.ones(3, 64, 1)
        self.assertEqual(float(temporal_total_variation(mask)), 0.0)

    def test_symmetric_byol_is_zero_for_matching_directions(self) -> None:
        vector = torch.randn(4, 16)
        loss = symmetric_byol_loss(vector, vector, vector, vector)
        self.assertAlmostEqual(float(loss), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
