from __future__ import annotations

import unittest

import torch

from training.phase5_baselines import (
    BASELINES,
    CPU_REPLAY_TOLERANCE,
    TASKS,
    Phase5BaselineConfig,
    Phase5RawBaseline,
    RawOHLCVLSTM,
    model_spec,
    smoke_test_baselines,
)


class Phase5BaselineTests(unittest.TestCase):
    def test_frozen_matrix_contract(self) -> None:
        self.assertEqual(BASELINES, ("raw_ohlcv_mlp", "raw_ohlcv_lstm"))
        self.assertEqual(
            TASKS, ("regression_h2", "classification_h2", "absolute_price_h8")
        )
        with self.assertRaises(ValueError):
            Phase5BaselineConfig("raw_ohlcv_mlp", "regression_h2", 1, epochs=49)
        with self.assertRaises(ValueError):
            Phase5BaselineConfig("raw_ohlcv_lstm", "classification_h2", 1, seed=1)

    def test_models_accept_five_channel_sequences(self) -> None:
        values = torch.randn(3, 64, 5)
        for baseline in BASELINES:
            for task in TASKS:
                config = Phase5BaselineConfig(baseline, task, 1, device="cpu")
                output = Phase5RawBaseline(config)(values)
                self.assertEqual(tuple(output.shape), (3, 3) if task == "classification_h2" else (3, 1))
                self.assertTrue(torch.isfinite(output).all())
                if task == "absolute_price_h8":
                    self.assertTrue(torch.all((output >= 0.0) & (output <= 1.0)))

    def test_lstm_is_multivariate_and_three_layer(self) -> None:
        model = RawOHLCVLSTM()
        self.assertEqual(model.lstm1.input_size, 5)
        self.assertEqual(
            [model.lstm1.hidden_size, model.lstm2.hidden_size, model.lstm3.hidden_size],
            [50, 30, 20],
        )

    def test_architecture_declares_no_explicit_price_skip(self) -> None:
        config = Phase5BaselineConfig("raw_ohlcv_mlp", "absolute_price_h8", 1, device="cpu")
        self.assertFalse(model_spec(config)["explicit_current_price_skip"])

    def test_cpu_replay_tolerance_is_architecture_specific(self) -> None:
        self.assertEqual(CPU_REPLAY_TOLERANCE["raw_ohlcv_mlp"], 2e-5)
        self.assertEqual(CPU_REPLAY_TOLERANCE["raw_ohlcv_lstm"], 2e-3)
        self.assertGreater(
            CPU_REPLAY_TOLERANCE["raw_ohlcv_lstm"],
            CPU_REPLAY_TOLERANCE["raw_ohlcv_mlp"],
        )

    def test_cpu_smoke_matrix(self) -> None:
        result = smoke_test_baselines()
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["models"]), 6)


if __name__ == "__main__":
    unittest.main()
