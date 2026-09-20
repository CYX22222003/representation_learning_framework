from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from data_processing.data_processing import build_processed_bundle, create_sequences, prepare_contract


def frame(rows: int = 20) -> pd.DataFrame:
    values = np.arange(rows, dtype=np.float64)
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=rows, freq="4h"),
        "open": values, "high": values + 0.1, "low": values - 0.1,
        "close": values + 0.05, "volume": values + 1.0,
    })


class SplitSafePipelineTests(unittest.TestCase):
    def test_boundary_precedes_windows_and_splits_share_no_raw_rows(self) -> None:
        prepared = prepare_contract(frame(), seq_len=4, train_ratio=0.75)
        self.assertEqual(prepared.metadata["raw_boundary_index"], 15)
        self.assertLess(prepared.train_window_ends.max(), 15)
        self.assertGreaterEqual(prepared.test_window_starts.min(), 15)
        self.assertEqual(prepared.train.shape, (12, 4, 5))
        self.assertEqual(prepared.test.shape, (2, 4, 5))

    def test_test_changes_do_not_change_train_windows_or_fitted_parameters(self) -> None:
        original = frame()
        changed = original.copy()
        changed.loc[15:, "volume"] = 1_000_000_000.0
        changed.loc[15:, "close"] = -999.0
        first = prepare_contract(original, seq_len=4, train_ratio=0.75)
        second = prepare_contract(changed, seq_len=4, train_ratio=0.75)
        np.testing.assert_array_equal(first.train, second.train)
        self.assertEqual(first.metadata["preprocessing"], second.metadata["preprocessing"])

    def test_imputation_is_causal_not_interpolated_from_future(self) -> None:
        data = frame()
        data.loc[3, "close"] = np.nan
        prepared = prepare_contract(data, seq_len=4, train_ratio=0.75)
        self.assertEqual(prepared.train[0, 3, 3], prepared.train[0, 2, 3])

    def test_windows_include_the_last_valid_start(self) -> None:
        values = np.arange(15, dtype=np.float32).reshape(5, 3)
        windows = create_sequences(values, 5)
        self.assertEqual(windows.shape, (1, 5, 3))

    def test_short_test_partition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "both sides"):
            prepare_contract(frame(10), seq_len=4, train_ratio=0.8)

    def test_rebuild_preserves_row_identity_hashes(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "contract.feather"
            source.write_bytes(b"stable-source-fixture")
            files = [(source.name, source.stat().st_size)]
            with patch("data_processing.data_processing.pd.read_feather", return_value=frame()):
                first = build_processed_bundle(files, 4, train_ratio=0.75, data_dir=temp_dir)
                second = build_processed_bundle(files, 4, train_ratio=0.75, data_dir=temp_dir)
        self.assertEqual(first.manifest["identity_hashes"], second.manifest["identity_hashes"])
        self.assertEqual(first.manifest["sequence_hashes"], second.manifest["sequence_hashes"])


if __name__ == "__main__":
    unittest.main()
