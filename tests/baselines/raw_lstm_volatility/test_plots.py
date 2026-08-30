from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baselines.raw_lstm_volatility.plot_experiment import plot_run


class RawLSTMPlotTest(unittest.TestCase):
    def test_plots_from_saved_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep = []
            for epoch in (1, 2):
                budget = root / f"e{epoch}"
                budget.mkdir(parents=True)
                np.savez(budget / "history.npz", epochs=np.arange(1, epoch + 1), train_loss=np.linspace(1.0, 0.5, epoch))
                targets = np.linspace(0.01, 0.2, 20, dtype=np.float32)
                preds = targets + 0.01
                np.savez(
                    budget / "predictions.npz",
                    predictions=preds,
                    targets=targets,
                    contract_ids=np.repeat([0, 1], 10),
                    processed_row_indices=np.arange(20),
                    window_starts=np.arange(20),
                )
                metrics = {"epoch": epoch, "mse": 0.0001, "rmse": 0.01, "mae": 0.01, "corr": 1.0}
                (budget / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                (budget / "per_contract_metrics.json").write_text(
                    json.dumps([{"contract_id": 0, "mse": 0.1}, {"contract_id": 1, "mse": 0.2}]),
                    encoding="utf-8",
                )
                sweep.append(metrics)
            (root / "sweep_metrics.json").write_text(json.dumps(sweep), encoding="utf-8")
            outputs = plot_run(root)
            self.assertTrue(outputs)
            for path in outputs:
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
