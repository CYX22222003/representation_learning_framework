from __future__ import annotations

import unittest

import torch

from baselines.saurl_ts import (
    SaURLConfig,
    SelfAdaptiveDataAugmentation,
    build_saurl,
    reconstruct_with_phase,
)


class SaURLAugmentationTests(unittest.TestCase):
    def test_default_dual_domain_shapes_and_finiteness(self) -> None:
        torch.manual_seed(7)
        model = build_saurl()
        for batch_size in (2, 4):
            batch = torch.randn(batch_size, 64, 5)
            result = model.make_views(batch)
            self.assertEqual(result.temporal.source.shape, (batch_size, 64, 5))
            self.assertEqual(result.temporal.view1.mask.shape, (batch_size, 64, 1))
            self.assertEqual(result.frequency.source.shape, (batch_size, 33, 5))
            self.assertEqual(result.frequency.view1.mask.shape, (batch_size, 33, 1))
            for view in (
                result.views.temporal_view1,
                result.views.temporal_view2,
                result.views.frequency_view1,
                result.views.frequency_view2,
            ):
                self.assertEqual(view.shape, (batch_size, 64, 5))
                self.assertFalse(view.is_complex())
                self.assertTrue(torch.isfinite(view).all())

    def test_identity_spectrum_round_trip(self) -> None:
        torch.manual_seed(8)
        batch = torch.randn(3, 64, 5)
        spectrum = torch.fft.rfft(batch, dim=1)
        reconstructed = reconstruct_with_phase(
            spectrum.abs(), torch.angle(spectrum), sequence_length=64
        )
        torch.testing.assert_close(reconstructed, batch, rtol=1e-5, atol=1e-6)

    def test_frequency_view_uses_its_transformed_magnitude_and_source_phase(self) -> None:
        torch.manual_seed(9)
        model = build_saurl()
        result = model.make_views(torch.randn(2, 64, 5))
        expected = reconstruct_with_phase(
            result.frequency.view1.view,
            result.frequency_phase,
            sequence_length=64,
        )
        torch.testing.assert_close(result.views.frequency_view1, expected)

    def test_deterministic_threshold_mask_has_straight_through_gradients(self) -> None:
        torch.manual_seed(10)
        config = SaURLConfig(dropout=0.0)
        module = SelfAdaptiveDataAugmentation(config)
        result = module(torch.randn(4, 64, 5))
        torch.testing.assert_close(result.view1.soft_mask, result.view2.soft_mask)
        torch.testing.assert_close(result.view1.mask, result.view2.mask)
        expected = (torch.sigmoid(result.factor_logits) > 0.5).to(
            result.view1.mask.dtype
        )
        torch.testing.assert_close(result.view1.mask.detach(), expected)
        self.assertTrue(
            torch.all(
                (result.view1.mask.detach() == 0)
                | (result.view1.mask.detach() == 1)
            )
        )
        self.assertFalse(torch.equal(result.view1.view, result.view2.view))
        loss = result.view1.mask.mean() + result.view2.mask.mean()
        loss.backward()
        gradient = module.factor_head.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_paper_threshold_is_strict_at_one_half(self) -> None:
        module = SelfAdaptiveDataAugmentation(SaURLConfig(dropout=0.0))
        with torch.no_grad():
            module.factor_head.weight.zero_()
            module.factor_head.bias.zero_()
        result = module(torch.randn(2, 64, 5))
        torch.testing.assert_close(
            result.view1.soft_mask, torch.full_like(result.view1.soft_mask, 0.5)
        )
        self.assertEqual(int(result.view1.mask.detach().count_nonzero()), 0)

    def test_deterministic_view_generation_replays(self) -> None:
        model = build_saurl(SaURLConfig(dropout=0.0)).eval()
        batch = torch.randn(2, 64, 5)
        first = model.make_views(batch).views
        second = model.make_views(batch).views
        torch.testing.assert_close(first.temporal_view1, second.temporal_view1)
        torch.testing.assert_close(first.frequency_view2, second.frequency_view2)


if __name__ == "__main__":
    unittest.main()
