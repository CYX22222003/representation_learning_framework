from __future__ import annotations

import unittest

import torch

from baselines.saurl_ts import DilatedCNNEncoder, RepresentationWiseAttention


class SaURLEncoderAttentionTests(unittest.TestCase):
    def test_selected_encoder_shapes(self) -> None:
        encoder = DilatedCNNEncoder()
        batch = torch.randn(4, 64, 5)
        self.assertEqual(encoder.forward_states(batch).shape, (4, 64, 128))
        self.assertEqual(encoder(batch).shape, (4, 128))
        self.assertEqual(len(encoder.blocks), 6)
        self.assertEqual(
            tuple(block.conv1.dilation[0] for block in encoder.blocks),
            (1, 2, 4, 8, 16, 32),
        )

    def test_attention_shapes_bounds_and_manual_sum(self) -> None:
        attention = RepresentationWiseAttention()
        branches = torch.randn(4, 3, 128)
        result = attention(branches)
        self.assertEqual(result.regional_weights.shape, (4, 3, 8))
        self.assertEqual(result.weights.shape, (4, 3, 128))
        self.assertEqual(result.weighted_branches.shape, (4, 3, 128))
        self.assertEqual(result.fused.shape, (4, 128))
        self.assertTrue(torch.all((result.weights >= 0) & (result.weights <= 1)))
        torch.testing.assert_close(result.fused, result.weighted_branches.sum(dim=1))

    def test_attention_rejects_concatenated_representation(self) -> None:
        attention = RepresentationWiseAttention()
        with self.assertRaises(ValueError):
            attention(torch.randn(2, 384))


if __name__ == "__main__":
    unittest.main()
