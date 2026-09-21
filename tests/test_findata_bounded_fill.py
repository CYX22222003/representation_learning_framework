from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_v2"))

from build_findata_bounded_fill import build_bounded_fill


class FindataBoundedFillTests(unittest.TestCase):
    def test_only_isolated_internal_gap_is_materialized(self) -> None:
        dates = pd.to_datetime(
            [
                "2026-01-01 00:00:00+00:00",
                "2026-01-01 01:00:00+00:00",
                "2026-01-01 03:00:00+00:00",
                "2026-01-01 06:00:00+00:00",
            ],
            utc=True,
        )
        clean = pd.DataFrame(
            {
                "condition_id": ["a"] * 4,
                "interval_minutes": [60] * 4,
                "date": dates,
                "open": [0.10, 0.20, 0.30, 0.40],
                "high": [0.11, 0.21, 0.31, 0.41],
                "low": [0.09, 0.19, 0.29, 0.39],
                "close": [0.10, 0.20, 0.30, 0.40],
                "volume": [1.0, 2.0, 3.0, 4.0],
                "trades": [1.0, 2.0, 3.0, 4.0],
            }
        )
        filled, counts = build_bounded_fill(clean, resolution="1h", interval_minutes=60)

        self.assertEqual(counts["inserted_rows"], 1)
        self.assertEqual(counts["long_gap_rows_preserved"], 2)
        self.assertNotIn(pd.Timestamp("2026-01-01 04:00:00+00:00"), set(filled["date"]))
        inserted = filled.loc[filled["is_imputed"]].iloc[0]
        self.assertEqual(inserted["date"], pd.Timestamp("2026-01-01 02:00:00+00:00"))
        self.assertTrue((inserted[["open", "high", "low", "close"]] == 0.20).all())
        self.assertEqual(inserted["volume"], 0.0)
        self.assertEqual(inserted["trades"], 0.0)
        self.assertEqual(inserted["original_gap_length_bars"], 1)
        self.assertEqual(inserted["time_since_last_observation"], 60)

        observed = filled.loc[filled["is_observed"], clean.columns].reset_index(drop=True)
        assert_frame_equal(observed, clean, check_exact=True)


if __name__ == "__main__":
    unittest.main()
