from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baselines.garch_lstm_stacking.meta import StackingMetaModel, clipped_predictions
from baselines.garch_lstm_stacking.run_experiment import StackConfig, _load_raw_lstm_run, verify_run


class ExperimentTests(unittest.TestCase):
    def test_raw_lstm_missing_budget_predictions_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            (raw / "config.json").write_text(json.dumps({"epoch_budgets": [15], "seed": 0}), encoding="utf-8")
            processed = root / "processed.npz"
            labels = root / "labels.npz"
            np.savez(processed, train=np.zeros((2, 4, 5), dtype=np.float32), test=np.zeros((2, 4, 5), dtype=np.float32))
            labels.write_bytes(b"placeholder")
            with self.assertRaisesRegex(FileNotFoundError, "predictions"):
                _load_raw_lstm_run(raw, processed, labels, StackConfig(epoch_budgets=(15,)))

    def test_verify_run_replays_saved_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof = root / "oof"
            e15 = root / "e15"
            oof.mkdir()
            e15.mkdir()
            g_oof = np.linspace(0.01, 0.04, 8, dtype=np.float32)
            l_oof = np.linspace(0.02, 0.05, 8, dtype=np.float32)
            y_oof = 0.4 * g_oof + 0.6 * l_oof
            meta = StackingMetaModel.fit(g_oof, l_oof, y_oof)
            g_test = np.array([0.02, 0.03], dtype=np.float32)
            l_test = np.array([0.04, 0.05], dtype=np.float32)
            raw = meta.predict_raw(g_test, l_test)
            nonnegative = clipped_predictions(raw)
            np.savez(oof / "garch_predictions.npz", prediction_guarded=g_oof, targets=y_oof)
            np.savez(oof / "lstm_predictions_e15.npz", predictions=l_oof)
            (e15 / "meta_scaler.json").write_text(json.dumps(meta.scaler_dict()), encoding="utf-8")
            (e15 / "meta_model.json").write_text(json.dumps(meta.model_dict()), encoding="utf-8")
            np.savez(
                e15 / "predictions.npz",
                garch_prediction_guarded=g_test,
                lstm_prediction=l_test,
                stack_prediction_raw=raw.astype(np.float32),
                stack_prediction_nonnegative=nonnegative.astype(np.float32),
            )
            self.assertTrue(verify_run(root))


if __name__ == "__main__":
    unittest.main()
