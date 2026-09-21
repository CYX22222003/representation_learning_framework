from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_v2"))

from plot_findata_native_ohlcv import bounded_fill_ohlcv


def _frame(hours: list[int]) -> pd.DataFrame:
    close = np.asarray([0.10, 0.20, 0.30, 0.40], dtype=np.float64)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [f"2026-01-01 {hour:02d}:00:00+00:00" for hour in hours], utc=True
            ),
            "open": close - 0.01,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": [10.0, 20.0, 30.0, 40.0],
        }
    )


class FindataOhlcvPlotTests(unittest.TestCase):
    def test_fill_only_complete_isolated_gap_with_flat_zero_volume_bar(self) -> None:
        grid, counts = bounded_fill_ohlcv(
            _frame([0, 1, 3, 6]), interval_minutes=60, maximum_fill_bars=1
        )
        self.assertEqual(len(grid), 7)
        self.assertEqual(counts["internal_missing_rows"], 3)
        self.assertEqual(counts["imputed_rows"], 1)
        self.assertEqual(counts["unfilled_missing_rows"], 2)

        isolated = grid.loc[grid["date"] == pd.Timestamp("2026-01-01 02:00:00+00:00")].iloc[0]
        self.assertTrue(isolated["is_imputed"])
        np.testing.assert_allclose(
            isolated[["open", "high", "low", "close"]].to_numpy(dtype=float),
            np.full(4, 0.20),
        )
        self.assertEqual(float(isolated["volume"]), 0.0)

        long_gap = grid.loc[
            grid["date"].isin(
                pd.to_datetime(
                    ["2026-01-01 04:00:00+00:00", "2026-01-01 05:00:00+00:00"],
                    utc=True,
                )
            )
        ]
        self.assertTrue(long_gap[list(("open", "high", "low", "close", "volume"))].isna().all().all())
        self.assertFalse(long_gap["is_imputed"].any())

if __name__ == "__main__":
    unittest.main()
