from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.volatility_labels import (
    OVERLAP_NOTE,
    build_aligned_volatility_labels,
    realized_volatility,
    save_volatility_label_bundle,
    load_volatility_label_bundle,
    validate_volatility_label_bundle,
)


def _contract(n: int, seq_len: int = 4, base: float = 1.0) -> np.ndarray:
    arr = np.zeros((n, seq_len, 5), dtype=np.float32)
    for i in range(n):
        arr[i, :, 3] = base * np.exp((i + 1) * 0.01 * np.arange(seq_len, dtype=np.float32))
        arr[i, :, :3] = arr[i, :, 3:4]
    return arr


class VolatilityLabelsTest(unittest.TestCase):
    def test_realized_volatility_constant_log_return(self) -> None:
        prices = np.exp(np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32))
        self.assertAlmostEqual(float(realized_volatility(prices)), 0.1, places=6)

    def test_contract_and_split_boundaries_are_not_crossed(self) -> None:
        bundle = build_aligned_volatility_labels([_contract(5, base=1.0), _contract(4, base=2.0)], train_ratio=0.6)
        np.testing.assert_array_equal(bundle["train_row_indices"], np.array([0, 1, 3], dtype=np.int64))
        np.testing.assert_array_equal(bundle["test_row_indices"], np.array([0, 2], dtype=np.int64))
        np.testing.assert_array_equal(bundle["train_contract_ids"], np.array([0, 0, 1], dtype=np.int32))
        np.testing.assert_array_equal(bundle["test_contract_ids"], np.array([0, 1], dtype=np.int32))
        self.assertEqual(bundle["train_labels"].shape[0], 3)
        self.assertEqual(bundle["test_labels"].shape[0], 2)

    def test_labels_use_next_sequence_same_contract(self) -> None:
        c0 = _contract(5)
        bundle = build_aligned_volatility_labels([c0], train_ratio=0.8)
        expected = realized_volatility(c0[1, :, 3])
        self.assertAlmostEqual(float(bundle["train_labels"][0]), float(expected), places=7)

    def test_validation_and_round_trip(self) -> None:
        bundle = build_aligned_volatility_labels([_contract(5)], train_ratio=0.8)
        result = validate_volatility_label_bundle(bundle, train_size=4, test_size=1)
        self.assertIn("63 of 64", result["overlap_note"])
        self.assertEqual(OVERLAP_NOTE, result["overlap_note"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.npz"
            save_volatility_label_bundle(bundle, {"label_mode": "next_stride_one_window_realized_volatility", "horizon": 1}, path)
            loaded, _ = load_volatility_label_bundle(path)
        np.testing.assert_array_equal(loaded["train_labels"], bundle["train_labels"])


if __name__ == "__main__":
    unittest.main()
