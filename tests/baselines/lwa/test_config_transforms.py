from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import torch

from baselines.lwa import LWAConfig, MorletCWT, orthonormal_rfft


class LWAConfigTransformTests(unittest.TestCase):
    def test_frozen_length_64_dimensions(self) -> None:
        config = LWAConfig()
        self.assertEqual(config.frequency_bins, 33)
        self.assertEqual(config.fourier_reduced_length, 9)
        self.assertEqual(config.wavelet_reduced_shape, (6, 8))
        self.assertEqual(config.inference_dim, 384)
        self.assertEqual(len(config.wavelet_scales), 48)
        self.assertAlmostEqual(config.wavelet_scales[0], 1.0)
        self.assertAlmostEqual(config.wavelet_scales[-1], 128.0)

    def test_unapproved_input_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LWAConfig(sequence_length=128)

    def test_rfft_uses_time_axis_and_orthonormal_normalization(self) -> None:
        config = LWAConfig()
        batch = torch.arange(2 * 64 * 5, dtype=torch.float32).reshape(2, 64, 5)
        actual = orthonormal_rfft(batch, config)
        expected = torch.fft.rfft(
            batch.transpose(1, 2), dim=-1, norm="ortho"
        )
        self.assertEqual(actual.shape, (2, 5, 33))
        torch.testing.assert_close(actual, expected)
        wrong_axis = torch.fft.rfft(batch, dim=-1, norm="ortho")
        self.assertNotEqual(actual.shape, wrong_axis.shape)

    @unittest.skipUnless(
        importlib.util.find_spec("pywt") is not None,
        "PyWavelets is installed in the admitted remote runtime, not this venv",
    )
    def test_cwt_is_channelwise_float32_magnitude(self) -> None:
        config = LWAConfig()
        time = torch.linspace(0.0, 2.0 * torch.pi, 64)
        batch = torch.stack(
            tuple(
                torch.stack(
                    tuple(
                        torch.sin((channel + 1) * time + sample)
                        for channel in range(5)
                    ),
                    dim=1,
                )
                for sample in range(2)
            )
        )
        actual = MorletCWT(config)(batch)
        self.assertEqual(actual.shape, (2, 5, 48, 64))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device.type, "cpu")
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(np.all(actual.numpy() >= 0.0))


if __name__ == "__main__":
    unittest.main()
