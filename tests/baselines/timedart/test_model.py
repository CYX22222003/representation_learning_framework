from __future__ import annotations

import io
import unittest

import torch

from baselines.timedart import PatchDiffusion, TimeDARTConfig, TimeDARTEncoder, TimeDARTPretrainer
from baselines.timedart.attention import causal_mask, self_only_mask
from baselines.timedart.patching import channelwise_patches, unpatch_channels


class TimeDARTModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.config = TimeDARTConfig(
            input_len=8,
            channels=5,
            patch_len=2,
            stride=2,
            d_model=8,
            n_heads=2,
            d_ff=16,
            encoder_layers=2,
            decoder_layers=1,
            dropout=0.0,
            projector_dropout=0.0,
            diffusion_steps=20,
        )

    def test_patches_cover_each_channel_without_overlap(self) -> None:
        x = torch.arange(2 * 8 * 5, dtype=torch.float32).reshape(2, 8, 5)
        patches = channelwise_patches(x, self.config)
        self.assertEqual(tuple(patches.shape), (10, 4, 2))
        torch.testing.assert_close(patches[0], x[0, :, 0].reshape(4, 2))
        torch.testing.assert_close(patches[1], x[0, :, 1].reshape(4, 2))
        torch.testing.assert_close(unpatch_channels(patches, self.config), x)

    def test_attention_masks_and_direct_clean_history_boundary(self) -> None:
        expected_causal = torch.tensor(
            [[False, True, True, True],
             [False, False, True, True],
             [False, False, False, True],
             [False, False, False, False]]
        )
        torch.testing.assert_close(causal_mask(4), expected_causal)
        torch.testing.assert_close(self_only_mask(4), ~torch.eye(4, dtype=torch.bool))

        model = TimeDARTPretrainer(self.config).eval()
        patches = torch.randn(5, 4, 2)
        changed = patches.clone()
        changed[:, 2] += 50
        with torch.no_grad():
            before = model._clean_history(patches)
            after = model._clean_history(changed)
        # The target patch at position 2 enters the shifted clean path only
        # from output position 3 onward. Instance normalization is deliberately
        # excluded here because it uses whole-window statistics.
        torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)
        self.assertGreater((before[:, 3:] - after[:, 3:]).abs().max().item(), 1e-4)

    def test_full_window_instance_normalization_limits_strict_ar_claim(self) -> None:
        model = TimeDARTPretrainer(self.config).eval()
        x = torch.randn(1, 8, 5)
        changed = x.clone()
        changed[:, -2:, 0] += 3
        with torch.no_grad():
            before = model.clean_history_states(x)
            after = model.clean_history_states(changed)
        # Even the first state can move: its SOS-only input is fixed, but
        # later states use earlier patches normalized with full-window stats.
        # Position 1 (zero based) sees the first patch and exposes the effect.
        self.assertGreater((before[0, 1] - after[0, 1]).abs().max().item(), 1e-5)

    def test_diffusion_independent_steps_buffers_and_noise_formula(self) -> None:
        diffusion = PatchDiffusion(steps=20)
        clean = torch.ones(2, 4, 2)
        steps = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        noise = torch.zeros_like(clean)
        corrupted, returned_noise, returned_steps = diffusion(
            clean, timesteps=steps, noise=noise
        )
        expected = diffusion.gamma[steps].sqrt().unsqueeze(-1).expand_as(clean)
        torch.testing.assert_close(corrupted, expected)
        torch.testing.assert_close(returned_noise, noise)
        torch.testing.assert_close(returned_steps, steps)
        self.assertIn("gamma", diffusion.state_dict())
        self.assertIn("beta", diffusion.state_dict())
        self.assertEqual(len(torch.unique(steps)), 8)

    def test_patchwise_projection_cannot_mix_reconstruction_positions(self) -> None:
        model = TimeDARTPretrainer(self.config).eval()
        states = torch.randn(5, 4, 8)
        changed = states.clone()
        changed[:, 2] += 10
        with torch.no_grad():
            original = model.patch_projector(states)
            modified = model.patch_projector(changed)
        torch.testing.assert_close(original[:, :2], modified[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(original[:, 3:], modified[:, 3:], rtol=0, atol=0)
        self.assertGreater((original[:, 2] - modified[:, 2]).abs().max().item(), 1e-4)

    def test_self_only_decoder_excludes_other_patch_queries_and_history(self) -> None:
        model = TimeDARTPretrainer(self.config).eval()
        queries = torch.randn(5, 4, 8)
        history = torch.randn(5, 4, 8)
        changed_queries = queries.clone()
        changed_history = history.clone()
        changed_queries[:, 3] += 20
        changed_history[:, 3] -= 20
        with torch.no_grad():
            before = model.denoiser(queries, history)
            after = model.denoiser(changed_queries, changed_history)
        torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)
        self.assertGreater((before[:, 3] - after[:, 3]).abs().max().item(), 1e-4)

    def test_pretraining_backward_and_decoder_free_extraction(self) -> None:
        model = TimeDARTPretrainer(self.config)
        x = torch.randn(2, 8, 5)
        result = model(x)
        self.assertEqual(tuple(result.reconstruction.shape), (2, 8, 5))
        self.assertEqual(tuple(result.timesteps.shape), (10, 4))
        self.assertTrue(torch.isfinite(result.loss).item())
        result.loss.backward()
        self.assertIsNotNone(model.encoder.patch_embedding.weight.grad)
        self.assertIsNotNone(model.denoiser.blocks[0].cross_attention.in_proj_weight.grad)
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ))

        frozen = model.frozen_encoder()
        self.assertIsInstance(frozen, TimeDARTEncoder)
        self.assertFalse(any(parameter.requires_grad for parameter in frozen.parameters()))
        self.assertFalse(any("denoiser" in name or "diffusion" in name for name in frozen.state_dict()))
        with torch.no_grad():
            features = frozen(x)
        self.assertEqual(tuple(features.shape), (2, 5 * (8 + 2)))
        torch.testing.assert_close(features[:, 40:45], x.mean(dim=1))
        torch.testing.assert_close(
            features[:, 45:50], (x.var(dim=1, unbiased=False) + 1e-5).sqrt()
        )

    def test_frozen_encoder_state_and_batch_replay(self) -> None:
        model = TimeDARTPretrainer(self.config).eval().frozen_encoder()
        x = torch.randn(3, 8, 5)
        with torch.no_grad():
            together = model(x)
            separate = torch.cat([model(x[i : i + 1]) for i in range(3)])
        torch.testing.assert_close(together, separate, rtol=1e-6, atol=1e-6)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        reloaded = TimeDARTEncoder(self.config).eval()
        reloaded.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
        with torch.no_grad():
            replayed = reloaded(x)
        torch.testing.assert_close(together, replayed, rtol=0, atol=0)

    def test_complete_pretrainer_state_replays_fixed_corruption(self) -> None:
        model = TimeDARTPretrainer(self.config).eval()
        x = torch.randn(2, 8, 5)
        timesteps = torch.arange(4).repeat(10, 1) % self.config.diffusion_steps
        noise = torch.randn(10, 4, 2)
        with torch.no_grad():
            original = model(x, timesteps=timesteps, noise=noise)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = TimeDARTPretrainer(self.config).eval()
        restored.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
        with torch.no_grad():
            replayed = restored(x, timesteps=timesteps, noise=noise)
        torch.testing.assert_close(original.reconstruction, replayed.reconstruction, rtol=0, atol=0)
        torch.testing.assert_close(original.loss, replayed.loss, rtol=0, atol=0)

    def test_invalid_patching_and_nonfinite_input_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TimeDARTConfig(input_len=8, stride=1, patch_len=2)
        encoder = TimeDARTEncoder(self.config)
        with self.assertRaises(ValueError):
            encoder(torch.zeros(2, 7, 5))
        invalid = torch.zeros(2, 8, 5)
        invalid[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            encoder(invalid)


if __name__ == "__main__":
    unittest.main()
