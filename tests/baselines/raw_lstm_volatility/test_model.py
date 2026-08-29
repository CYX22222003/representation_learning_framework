from __future__ import annotations

import unittest

import numpy as np
import torch

from baselines.raw_lstm_volatility.model import (
    RawLSTMVolatility,
    VolatilitySequenceDataset,
    count_parameters,
    make_loader,
    set_seed,
    train_one_epoch,
)


class RawLSTMModelTest(unittest.TestCase):
    def test_output_shape_and_non_negative(self) -> None:
        model = RawLSTMVolatility()
        for batch in (1, 3):
            out = model(torch.randn(batch, 64, 5))
            self.assertEqual(tuple(out.shape), (batch, 1))
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue((out >= 0).all())

    def test_dataset_uses_row_indices(self) -> None:
        seq = np.arange(6 * 4 * 5, dtype=np.float32).reshape(6, 4, 5)
        labels = np.array([0.2, 0.4], dtype=np.float32)
        ds = VolatilitySequenceDataset(seq, labels, np.array([1, 4], dtype=np.int64))
        np.testing.assert_array_equal(ds[0][0].numpy(), seq[1])
        self.assertAlmostEqual(float(ds[1][1]), 0.4, places=6)

    def test_training_step_changes_parameters(self) -> None:
        set_seed(0)
        model = RawLSTMVolatility(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
        seq = np.random.default_rng(0).normal(size=(8, 6, 5)).astype(np.float32)
        labels = np.linspace(0.01, 0.08, 8, dtype=np.float32)
        ds = VolatilitySequenceDataset(seq, labels, np.arange(8, dtype=np.int64))
        loader = make_loader(ds, batch_size=4, seed=0)
        before = [p.detach().clone() for p in model.parameters()]
        loss = train_one_epoch(model, loader, torch.optim.Adam(model.parameters(), lr=1e-3), torch.device("cpu"))
        self.assertGreater(loss, 0.0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_seeded_initialization_and_loader_order(self) -> None:
        set_seed(123)
        a = RawLSTMVolatility(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
        set_seed(123)
        b = RawLSTMVolatility(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
        self.assertTrue(all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters())))
        self.assertGreater(count_parameters(a), 0)


if __name__ == "__main__":
    unittest.main()
