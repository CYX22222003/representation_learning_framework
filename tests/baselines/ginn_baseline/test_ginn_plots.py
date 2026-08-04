from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from baselines.ginn_baseline.ginn_model import LSTMVariancePredictor
from baselines.ginn_baseline.plot_experiment import plot_budget, plot_epoch_sweep
from tests.baselines.ginn_baseline.helpers import make_two_epoch_synthetic_run


class PlotTests(unittest.TestCase):
    def test_saved_artifacts_generate_non_empty_plots_without_model_or_raw_data(self):
        with tempfile.TemporaryDirectory() as raw:
            run_root = make_two_epoch_synthetic_run(Path(raw))
            with patch.object(pd, "read_feather", side_effect=AssertionError("raw read")), patch.object(
                LSTMVariancePredictor, "forward", side_effect=AssertionError("model call")
            ):
                for epoch in (1, 2):
                    plot_budget(run_root / f"e{epoch}")
                plot_epoch_sweep(run_root)
            expected = [
                run_root / "e1" / "images" / "training_curve.png",
                run_root / "e1" / "images" / "pred_vs_actual.png",
                run_root / "e1" / "images" / "error_distribution.png",
                run_root / "e2" / "images" / "training_curve.png",
                run_root / "e2" / "images" / "pred_vs_actual.png",
                run_root / "e2" / "images" / "error_distribution.png",
                run_root / "images" / "epoch_sweep.png",
            ]
            self.assertTrue(all(path.exists() for path in expected))
            self.assertTrue(all(path.stat().st_size > 1000 for path in expected))


if __name__ == "__main__":
    unittest.main()
