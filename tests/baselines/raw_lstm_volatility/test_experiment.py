from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baselines.raw_lstm_volatility.run_experiment import ExperimentConfig, run_experiment
from tasks.volatility_labels import build_aligned_volatility_labels, save_volatility_label_bundle, sha256_file


class RawLSTMExperimentTest(unittest.TestCase):
    def test_synthetic_run_artifacts(self) -> None:
        rng = np.random.default_rng(1)
        train = rng.normal(size=(12, 6, 5)).astype(np.float32)
        test = rng.normal(size=(8, 6, 5)).astype(np.float32)
        train[:, :, 3] = np.clip(np.abs(train[:, :, 3]), 0.01, None)
        test[:, :, 3] = np.clip(np.abs(test[:, :, 3]), 0.01, None)
        bundle = build_aligned_volatility_labels([np.concatenate([train[:6], test[:4]]), np.concatenate([train[6:], test[4:]])], train_ratio=0.6)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed.npz"
            labels = root / "labels.npz"
            np.savez_compressed(processed, train=train, test=test)
            manifest = {
                "label_mode": "next_stride_one_window_realized_volatility",
                "horizon": 1,
                "processed_npz_sha256": sha256_file(processed),
                "processed_shapes": {"train": list(train.shape), "test": list(test.shape)},
            }
            save_volatility_label_bundle(bundle, manifest, labels)
            run_root = root / "run"
            sweep = run_experiment(
                processed,
                labels,
                run_root,
                ExperimentConfig(epoch_budgets=(1, 2), batch_size=4, hidden_size=8, num_layers=1, dropout=0.0, device="cpu"),
            )
            self.assertEqual([row["epoch"] for row in sweep], [1, 2])
            self.assertTrue((run_root / "sweep_metrics.json").exists())
            for epoch in (1, 2):
                budget = run_root / f"e{epoch}"
                self.assertTrue((budget / "checkpoint.pth").exists())
                self.assertTrue((budget / "history.npz").exists())
                self.assertTrue((budget / "predictions.npz").exists())
                self.assertTrue((budget / "metrics.json").exists())
                self.assertTrue((budget / "per_contract_metrics.json").exists())
                with np.load(budget / "predictions.npz") as pred:
                    np.testing.assert_array_equal(pred["processed_row_indices"], bundle["test_row_indices"])
            saved = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved), 2)


if __name__ == "__main__":
    unittest.main()
