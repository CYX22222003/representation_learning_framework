from __future__ import annotations

import unittest

import torch

from baselines.lwa import (
    LWAConfig,
    build_lwa_inference_encoder,
    build_lwa_joint,
    build_lwa_mapper_stage,
    orthonormal_rfft,
)


def _views(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time = torch.randn(batch_size, 64, 5)
    fourier = orthonormal_rfft(time)
    wavelet = torch.rand(batch_size, 5, 48, 64)
    return time, fourier, wavelet


class LWAModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.config = LWAConfig()

    def test_joint_model_finite_forward_and_backward(self) -> None:
        model = build_lwa_joint(self.config)
        output = model(*_views())
        self.assertEqual(output.representations.time.shape, (2, 128))
        self.assertEqual(output.representations.fourier.shape, (2, 128))
        self.assertEqual(output.representations.wavelet.shape, (2, 128))
        self.assertEqual(output.projections.time.shape, (2, 128))
        self.assertTrue(torch.isfinite(output.total_loss))
        output.total_loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_mapper_stage_freezes_encoders_and_uses_raw_representations(self) -> None:
        joint = build_lwa_joint(self.config).eval()
        mapper_stage = build_lwa_mapper_stage(joint).train()
        self.assertFalse(mapper_stage.time_encoder.training)
        self.assertFalse(mapper_stage.fourier_encoder.training)
        self.assertFalse(mapper_stage.wavelet_encoder.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in (
                    mapper_stage.time_encoder,
                    mapper_stage.fourier_encoder,
                    mapper_stage.wavelet_encoder,
                )
                for parameter in module.parameters()
            )
        )
        output = mapper_stage(*_views())
        self.assertEqual(output.mapped_fourier.shape, (2, 128))
        self.assertEqual(output.mapped_wavelet.shape, (2, 128))
        self.assertFalse(output.representations.time.requires_grad)
        output.total_loss.backward()
        mapper_gradients = [
            parameter.grad for parameter in mapper_stage.mapper_parameters()
        ]
        self.assertTrue(all(gradient is not None for gradient in mapper_gradients))
        self.assertTrue(
            all(
                parameter.grad is None
                for module in (
                    mapper_stage.time_encoder,
                    mapper_stage.fourier_encoder,
                    mapper_stage.wavelet_encoder,
                )
                for parameter in module.parameters()
            )
        )

    def test_inference_width_slice_order_and_auxiliary_exclusion(self) -> None:
        mapper_stage = build_lwa_mapper_stage(build_lwa_joint(self.config))
        inference = build_lwa_inference_encoder(mapper_stage)
        time, _, _ = _views(3)
        parts = inference.encode_parts(time)
        self.assertEqual(parts.concatenated.shape, (3, 384))
        torch.testing.assert_close(parts.concatenated[:, :128], parts.time)
        torch.testing.assert_close(
            parts.concatenated[:, 128:256], parts.mapped_fourier
        )
        torch.testing.assert_close(
            parts.concatenated[:, 256:384], parts.mapped_wavelet
        )
        module_names = set(dict(inference.named_modules()))
        self.assertFalse(any("fourier_encoder" in name for name in module_names))
        self.assertFalse(any("wavelet_encoder" in name for name in module_names))
        self.assertFalse(any("projector" in name for name in module_names))
        self.assertFalse(any("embedding_mapper" in name for name in module_names))

    def test_inference_is_deterministic_and_permanently_evaluation_mode(self) -> None:
        inference = build_lwa_inference_encoder(
            build_lwa_mapper_stage(build_lwa_joint(self.config))
        )
        fixed = torch.randn(2, 64, 5)
        inference.train()
        self.assertFalse(inference.training)
        first = inference(fixed)
        second = inference(fixed)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertFalse(first.requires_grad)


if __name__ == "__main__":
    unittest.main()
