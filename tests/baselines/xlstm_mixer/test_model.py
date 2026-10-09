from __future__ import annotations

import copy
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn

from baselines.xlstm_mixer import NonAffineRevIN, XLSTMMixer, XLSTMMixerConfig


class RecordingStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(x.detach().clone())
        return x * self.scale


class XLSTMMixerModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.config = XLSTMMixerConfig()
        self.stack = RecordingStack()
        self.builder = patch(
            "baselines.xlstm_mixer.model.build_xlstm_stack",
            return_value=self.stack,
        )
        self.builder.start()
        self.addCleanup(self.builder.stop)

    def build_model(self) -> XLSTMMixer:
        return XLSTMMixer(self.config, backend="vanilla")

    def test_frozen_default_contract_and_invalid_alternatives(self) -> None:
        canonical = XLSTMMixerConfig()
        self.assertEqual(canonical.channel_order, ("open", "high", "low", "close", "volume"))
        self.assertEqual((canonical.context_length, canonical.horizon), (64, 8))
        self.assertEqual((canonical.embedding_dim, canonical.num_heads), (128, 8))
        self.assertEqual(canonical.num_initial_tokens, 1)
        self.assertEqual(canonical.reverse_view, "latent_feature_axis")
        self.assertFalse(canonical.revin_affine)
        self.assertEqual(canonical.to_dict(), XLSTMMixerConfig.from_dict(canonical.to_dict()).to_dict())
        self.assertEqual(len(canonical.sha256), 64)

        with self.assertRaises(ValueError):
            XLSTMMixerConfig(num_initial_tokens=0)
        with self.assertRaises(ValueError):
            XLSTMMixerConfig(reverse_view="variate_axis")
        with self.assertRaises(ValueError):
            XLSTMMixerConfig(embedding_dim=10, num_heads=4)
        with self.assertRaises(ValueError):
            XLSTMMixerConfig(context_length=32)

    def test_revin_matches_source_math_and_has_no_parameters(self) -> None:
        revin = NonAffineRevIN(num_variates=2, epsilon=1e-5)
        x = torch.tensor([[[1.0, 3.0], [3.0, 7.0], [5.0, 11.0]]], requires_grad=True)
        normalized, statistics = revin.normalize(x)
        expected_mean = x.detach().mean(dim=1, keepdim=True)
        expected_std = (x.detach().var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        torch.testing.assert_close(statistics.mean, expected_mean)
        torch.testing.assert_close(statistics.standard_deviation, expected_std)
        torch.testing.assert_close(normalized, (x - expected_mean) / expected_std)
        torch.testing.assert_close(revin.denormalize(normalized, statistics), x)
        self.assertFalse(statistics.mean.requires_grad)
        self.assertFalse(statistics.standard_deviation.requires_grad)
        self.assertEqual(sum(parameter.numel() for parameter in revin.parameters()), 0)

    def test_complete_dataflow_shapes_and_shared_projections(self) -> None:
        model = self.build_model().eval()
        x = torch.randn(2, 64, 5)
        with torch.no_grad():
            trace = model.forecast_with_trace(x)
        self.assertEqual(tuple(trace.normalized_input.shape), (2, 64, 5))
        self.assertEqual(tuple(trace.preliminary_forecast.shape), (2, 8, 5))
        self.assertEqual(tuple(trace.embedded_variates.shape), (2, 5, 128))
        self.assertEqual(tuple(trace.tokens_with_initial.shape), (2, 6, 128))
        self.assertEqual(tuple(trace.forward_view.shape), (2, 6, 128))
        self.assertEqual(tuple(trace.reverse_view.shape), (2, 6, 128))
        self.assertEqual(tuple(trace.mixed_variates.shape), (2, 5, 256))
        self.assertEqual(tuple(trace.normalized_forecast.shape), (2, 8, 5))
        self.assertEqual(tuple(trace.forecast.shape), (2, 8, 5))

        # Each projection is one module applied with variates as a batch-like
        # axis; no channel-specific ModuleList or parameter bank exists.
        self.assertEqual(tuple(model.time_projection.weight.shape), (8, 64))
        self.assertEqual(tuple(model.up_projection.weight.shape), (128, 8))
        self.assertEqual(tuple(model.output_projection.weight.shape), (8, 256))

    def test_reversal_uses_latent_axis_and_preserves_variate_order(self) -> None:
        tokens = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]
        )
        expected = torch.tensor(
            [[[3.0, 2.0, 1.0], [6.0, 5.0, 4.0], [9.0, 8.0, 7.0]]]
        )
        torch.testing.assert_close(XLSTMMixer.reverse_latent_features(tokens), expected)

    def test_reverse_view_runs_first_and_token_is_removed_without_displacement(self) -> None:
        model = self.build_model().eval()
        x = torch.randn(2, 64, 5)
        with torch.no_grad():
            trace = model.forecast_with_trace(x)
        self.assertEqual(len(self.stack.calls), 2)
        torch.testing.assert_close(self.stack.calls[0], trace.reversed_tokens)
        torch.testing.assert_close(self.stack.calls[1], trace.tokens_with_initial)
        torch.testing.assert_close(trace.tokens_with_initial[:, 1:], trace.embedded_variates)
        expected_mixed = torch.cat(
            (trace.embedded_variates, torch.flip(trace.embedded_variates, dims=(-1,))),
            dim=-1,
        )
        torch.testing.assert_close(trace.mixed_variates, expected_mixed)

    def test_headline_extraction_is_horizon_last_and_close_index_three(self) -> None:
        model = self.build_model()
        full_path = torch.arange(2 * 8 * 5, dtype=torch.float32).reshape(2, 8, 5)
        torch.testing.assert_close(model.headline_close(full_path), full_path[:, 7, 3])

    def test_full_path_loss_uses_all_outputs_and_targets_never_enter_forward(self) -> None:
        model = self.build_model().eval()
        x = torch.randn(2, 64, 5)
        target_a = torch.zeros(2, 8, 5)
        target_b = torch.ones(2, 8, 5)
        with torch.no_grad():
            before = model(x)
            loss_a = model.full_path_l1_loss(before, target_a)
            after = model(x)
            loss_b = model.full_path_l1_loss(after, target_b)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(loss_a, (before - target_a).abs().mean())
        torch.testing.assert_close(loss_b, (after - target_b).abs().mean())

    def test_eval_is_batch_size_invariant(self) -> None:
        model = self.build_model().eval()
        x = torch.randn(4, 64, 5)
        with torch.no_grad():
            together = model(x)
            separate = torch.cat([model(x[index : index + 1]) for index in range(4)])
        torch.testing.assert_close(together, separate, rtol=1e-6, atol=1e-6)

    def test_finite_backward_reaches_all_trainable_components(self) -> None:
        model = self.build_model().train()
        x = torch.randn(3, 64, 5)
        target = torch.randn(3, 8, 5)
        loss = model.full_path_l1_loss(model(x), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_state_dict_round_trip_replays_and_rejects_wrong_contracts(self) -> None:
        model = self.build_model().eval()
        x = torch.randn(2, 64, 5)
        with torch.no_grad():
            expected = model(x)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)

        replacement_stack = RecordingStack()
        with patch(
            "baselines.xlstm_mixer.model.build_xlstm_stack",
            return_value=replacement_stack,
        ):
            replay = XLSTMMixer(self.config, backend="cuda").eval()
        replay.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
        with torch.no_grad():
            actual = replay(x)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

        state = copy.deepcopy(model.state_dict())
        state["_extra_state"]["architecture_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "architecture hash"):
            model.load_state_dict(state, strict=True)

        state = copy.deepcopy(model.state_dict())
        state["_extra_state"]["source_contract"]["official_source_commit"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "source contract"):
            model.load_state_dict(state, strict=True)

    def test_invalid_inputs_outputs_and_targets_are_rejected(self) -> None:
        model = self.build_model()
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 63, 5))
        with self.assertRaises(TypeError):
            model(torch.zeros(2, 64, 5, dtype=torch.int64))
        invalid = torch.zeros(2, 64, 5)
        invalid[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model(invalid)
        with self.assertRaises(ValueError):
            model.headline_close(torch.zeros(2, 7, 5))
        with self.assertRaises(ValueError):
            model.full_path_l1_loss(torch.zeros(2, 8, 5), torch.zeros(2, 8, 4))


if __name__ == "__main__":
    unittest.main()
