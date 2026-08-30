from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baselines.garch_lstm_stacking.plot_experiment import plot_all


class PlotTests(unittest.TestCase):
    def test_plots_from_saved_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "oof").mkdir()
            for epoch in (15, 50, 100):
                (root / f"e{epoch}").mkdir()
            n = 20
            targets = np.linspace(0.01, 0.05, n, dtype=np.float32)
            cids = np.repeat([0, 1], n // 2).astype(np.int32)
            rows = np.arange(n, dtype=np.int64)
            starts = np.arange(n, dtype=np.int64)
            np.savez(root / "oof" / "garch_predictions.npz", prediction_guarded=targets * 0.9, targets=targets)
            np.savez(root / "oof" / "lstm_predictions_e15.npz", predictions=targets * 1.1)
            comparisons = []
            for epoch in (15, 50, 100):
                stack = targets * (1.0 + epoch / 1000.0)
                np.savez(
                    root / f"e{epoch}" / "predictions.npz",
                    targets=targets,
                    lstm_prediction=targets * 1.1,
                    stack_prediction_nonnegative=stack,
                    contract_ids=cids,
                    processed_row_indices=rows,
                    window_starts=starts,
                )
                (root / f"e{epoch}" / "meta_model.json").write_text(
                    json.dumps({"coefficients": {"garch": 0.1, "lstm": 0.8, "interaction": 0.0}}),
                    encoding="utf-8",
                )
                (root / f"e{epoch}" / "per_contract_metrics.json").write_text(
                    json.dumps([
                        {"contract_id": 0, "mse": 0.1 / epoch},
                        {"contract_id": 1, "mse": 0.2 / epoch},
                    ]),
                    encoding="utf-8",
                )
                comparisons.append(
                    {
                        "epoch": epoch,
                        "raw_lstm_mse": 0.02,
                        "stack_mse": 0.01,
                    }
                )
            (root / "comparison_with_raw_lstm.json").write_text(json.dumps(comparisons), encoding="utf-8")
            (root / "garch_diagnostics.json").write_text(
                json.dumps({"test": [{"contract_id": 0, "fallback_forecast_count": 1, "capped_forecast_count": 0}]}),
                encoding="utf-8",
            )
            paths = plot_all(root)
            self.assertEqual(len(paths), 6)
            for path in paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
