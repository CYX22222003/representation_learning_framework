from __future__ import annotations

import unittest

import torch

from baselines.lwa import (
    mean_sample_l1,
    negative_pair_mask,
    positive_pair_indices,
    symmetric_nt_xent,
    two_way_mapping_loss,
)


class LWALossTests(unittest.TestCase):
    def test_positive_indices_and_negative_mask(self) -> None:
        positives = positive_pair_indices(3, "cpu")
        torch.testing.assert_close(positives, torch.tensor([3, 4, 5, 0, 1, 2]))
        mask = negative_pair_mask(3, "cpu")
        self.assertEqual(mask.shape, (6, 6))
        self.assertTrue(torch.equal(mask.sum(dim=1), torch.full((6,), 4)))
        rows = torch.arange(6)
        self.assertFalse(mask[rows, rows].any())
        self.assertFalse(mask[rows, positives].any())

    def test_symmetric_nt_xent_is_dynamic_and_has_finite_gradients(self) -> None:
        for batch_size in (2, 5):
            left = torch.randn(batch_size, 8, requires_grad=True)
            right = torch.randn(batch_size, 8, requires_grad=True)
            loss = symmetric_nt_xent(left, right, temperature=0.15)
            self.assertEqual(loss.shape, ())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(left.grad).all())
            self.assertTrue(torch.isfinite(right.grad).all())

    def test_mean_sample_l1_matches_hand_computation(self) -> None:
        prediction = torch.tensor([[1.0, 3.0], [2.0, -1.0]])
        target = torch.tensor([[0.0, 1.0], [5.0, 1.0]])
        # Per-sample sums are 3 and 5, whose mean is 4.
        self.assertEqual(float(mean_sample_l1(prediction, target)), 4.0)

    def test_two_way_mapping_loss_is_unweighted_sum(self) -> None:
        zeros = torch.zeros(2, 3)
        ones = torch.ones(2, 3)
        twos = torch.full((2, 3), 2.0)
        fourier, wavelet, total = two_way_mapping_loss(
            zeros, ones, zeros, twos
        )
        self.assertEqual(float(fourier), 3.0)
        self.assertEqual(float(wavelet), 6.0)
        self.assertEqual(float(total), 9.0)


if __name__ == "__main__":
    unittest.main()
