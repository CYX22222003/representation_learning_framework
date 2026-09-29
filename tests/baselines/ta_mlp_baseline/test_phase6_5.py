from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.ta_mlp_baseline.phase6_5 import (
    Phase65TAConfig,
    build_phase65_ta_model,
    compute_walk_ta_features,
    run_phase65_ta_training,
    smoke_test_phase65_ta_models,
    strict_matrix_entries,
)


def candle_frame(rows: int = 180) -> pd.DataFrame:
    date = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    close = 0.5 + 0.02 * np.sin(np.arange(rows) / 9.0)
    return pd.DataFrame(
        {
            "condition_id": ["condition"] * rows,
            "date": date,
            "open": close - 0.001,
            "high": close + 0.002,
            "low": close - 0.002,
            "close": close,
            "volume": 100.0 + np.arange(rows),
        }
    )


class Phase65TAMLPTests(unittest.TestCase):
    def test_fixed_volume_fit_is_invariant_to_evaluation_volume_changes(self) -> None:
        frame = candle_frame()
        cutoff = pd.Timestamp("2026-01-06T00:00:00Z")
        first, first_manifest = compute_walk_ta_features(
            frame,
            train_start=pd.Timestamp("2026-01-01T00:00:00Z"),
            cutoff=cutoff,
        )
        modified = frame.copy()
        modified.loc[modified["date"] + pd.Timedelta(hours=1) >= cutoff, "volume"] *= 1000.0
        second, second_manifest = compute_walk_ta_features(
            modified,
            train_start=pd.Timestamp("2026-01-01T00:00:00Z"),
            cutoff=cutoff,
        )
        self.assertEqual(first_manifest["volume_fit"], second_manifest["volume_fit"])
        before = first["decision_date_ns"] + 3_600_000_000_000 < int(cutoff.value)
        np.testing.assert_allclose(
            first.loc[before, "zsVol"].to_numpy(),
            second.loc[before, "zsVol"].to_numpy(),
        )

    def test_matrix_and_architecture_are_frozen(self) -> None:
        self.assertEqual(len(strict_matrix_entries()), 5)
        self.assertEqual(sum(protocol == "P2" for _, protocol in strict_matrix_entries()), 4)
        config = Phase65TAConfig("ta_mlp", "P2", walk=1, device="cpu")
        model = build_phase65_ta_model(config)
        linear_widths = [module.out_features for module in model.modules() if hasattr(module, "out_features")]
        self.assertEqual(linear_widths, [128, 64, 32, 3])
        with self.assertRaises(ValueError):
            Phase65TAConfig("h0", "P1U", walk=1)

    def test_smoke_covers_five_per_walk_configurations(self) -> None:
        result = smoke_test_phase65_ta_models()
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["models"]), 5)
        self.assertEqual(result["models"]["ta_mlp/P2"]["batch_size"], 64)

    def test_occupied_run_is_refused_before_store_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                run_phase65_ta_training(
                    root / "missing.npz",
                    root,
                    Phase65TAConfig("ta_mlp", "P2", walk=1, device="cpu"),
                )


if __name__ == "__main__":
    unittest.main()
