from __future__ import annotations

import io
import unittest

import numpy as np
import torch
import torch.nn as nn

from baselines.sgn import SGNClassifier, SGNConfig
from baselines.sgn.model import PeriodMerge, ShiftedWindowBlock


INITIAL = np.asarray([[0.0, 1.0]] * 4 + [[1.0, 0.0]], dtype=np.float32)


class _Double(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return 2.0 * values


class ModelTests(unittest.TestCase):
    def test_config_freezes_complete_hierarchy(self) -> None:
        config = SGNConfig()
        self.assertEqual(config.pooled_dim, 128)
        self.assertEqual(config.kernel_sizes, (1, 3, 5, 7, 9, 11, 13))
        with self.assertRaises(ValueError):
            SGNConfig(period=32)

    def test_forward_loss_and_gradients_are_finite(self) -> None:
        torch.manual_seed(0)
        model = SGNClassifier(INITIAL)
        values = torch.randn(3, 64, 5)
        logits, diagnostics = model(values, return_diagnostics=True)
        self.assertEqual(tuple(logits.shape), (3, 3))
        loss = logits.square().mean() + 0.1 * diagnostics["similarity_regularizer"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_eval_assignment_and_output_are_deterministic(self) -> None:
        model = SGNClassifier(INITIAL).eval()
        values = torch.randn(2, 64, 5)
        first = model(values)
        second = model(values)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        assignment = model.assignment.assignment()
        self.assertTrue(torch.all((assignment == 0) | (assignment == 1)))
        torch.testing.assert_close(assignment.sum(dim=1), torch.ones(5))

    def test_temperature_and_checkpoint_round_trip(self) -> None:
        model = SGNClassifier(INITIAL)
        model.advance_temperature()
        self.assertEqual(int(model.assignment.training_step), 1)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        replay = SGNClassifier(INITIAL)
        replay.load_state_dict(torch.load(stream, weights_only=True))
        self.assertEqual(int(replay.assignment.training_step), 1)
        self.assertEqual(float(replay.assignment.temperature), float(model.assignment.temperature))

    def test_shift_restores_first_half_window(self) -> None:
        values = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
        output = ShiftedWindowBlock(_Double(), shifted=True)(values)
        flat_in, flat_out = values.reshape(1, 2, -1), output.reshape(1, 2, -1)
        torch.testing.assert_close(flat_out[:, :, :2], flat_in[:, :, :2])
        torch.testing.assert_close(flat_out[:, :, 2:], 2.0 * flat_in[:, :, 2:])

    def test_merge_shapes_and_single_window_guard(self) -> None:
        merge = PeriodMerge(SGNConfig())
        for windows, expected in ((6, 3), (5, 3), (4, 3), (3, 2), (2, 1)):
            output = merge(torch.randn(2, 128, windows, 16))
            self.assertEqual(tuple(output.shape), (2, 128, expected, 16))
        with self.assertRaises(ValueError):
            merge(torch.randn(2, 128, 1, 16))

    def test_wrong_input_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SGNClassifier(INITIAL)(torch.randn(2, 63, 5))


if __name__ == "__main__":
    unittest.main()
