from __future__ import annotations

import unittest

import torch

from baselines.lwa import (
    ConvolutionalMapper,
    FourierDomainEncoder,
    LWAConfig,
    ResidualBlock1D,
    TimeDomainEncoder,
    WaveletDomainEncoder,
    orthonormal_rfft,
)


class LWAEncoderMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.config = LWAConfig()

    def test_time_encoder_exact_shapes_and_schedule(self) -> None:
        encoder = TimeDomainEncoder(self.config).eval()
        batch = torch.randn(2, 64, 5)
        self.assertEqual(encoder.forward_features(batch).shape, (2, 64, 4))
        self.assertEqual(encoder(batch).shape, (2, 128))
        self.assertEqual(len(encoder.blocks), 8)
        self.assertEqual(
            tuple(block.stride for block in encoder.blocks),
            (1, 2, 1, 2, 1, 2, 1, 2),
        )
        self.assertEqual(
            tuple(block.out_channels for block in encoder.blocks),
            (32, 32, 32, 32, 64, 64, 64, 64),
        )

    def test_residual_block_preserves_or_downsamples_length(self) -> None:
        same = ResidualBlock1D(16, 16, kernel_size=5).eval()
        down = ResidualBlock1D(16, 32, kernel_size=5, stride=2).eval()
        batch = torch.randn(3, 16, 33)
        self.assertEqual(same(batch).shape, (3, 16, 33))
        self.assertEqual(down(batch).shape, (3, 32, 17))

    def test_fourier_encoder_exact_shapes(self) -> None:
        encoder = FourierDomainEncoder(self.config).eval()
        transformed = orthonormal_rfft(torch.randn(2, 64, 5), self.config)
        magnitude, phase = encoder.forward_branch_features(transformed)
        self.assertEqual(magnitude.shape, (2, 64, 9))
        self.assertEqual(phase.shape, (2, 64, 9))
        self.assertEqual(encoder(transformed).shape, (2, 128))

    def test_wavelet_encoder_exact_shapes(self) -> None:
        encoder = WaveletDomainEncoder(self.config).eval()
        transformed = torch.rand(2, 5, 48, 64)
        self.assertEqual(encoder.forward_features(transformed).shape, (2, 128, 6, 8))
        self.assertEqual(encoder(transformed).shape, (2, 128))

    def test_mapper_preserves_latent_length(self) -> None:
        mapper = ConvolutionalMapper(128, 64, 3)
        batch = torch.randn(4, 128)
        self.assertEqual(mapper(batch).shape, batch.shape)
        self.assertEqual(sum(p.numel() for p in mapper.parameters()), 449)


if __name__ == "__main__":
    unittest.main()
