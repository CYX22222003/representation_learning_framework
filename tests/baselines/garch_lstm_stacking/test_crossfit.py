from __future__ import annotations

import unittest

import numpy as np

from baselines.garch_lstm_stacking.crossfit import assert_oof_integrity, assert_same_row_identity, make_expanding_folds


class CrossfitTests(unittest.TestCase):
    def test_expanding_folds_cover_each_oof_row_once(self) -> None:
        contracts = np.repeat([0, 1], [12, 13]).astype(np.int32)
        rows = np.r_[np.arange(12), np.arange(100, 113)].astype(np.int64)
        starts = np.r_[np.arange(12), np.arange(13)].astype(np.int64)
        plan = make_expanding_folds(contracts, rows, starts, n_folds=5)
        assert_oof_integrity(plan)
        self.assertEqual(len(plan.prediction_row_indices()), len(np.unique(plan.prediction_row_indices())))
        self.assertGreater(plan.burn_in_row_indices.size, 0)
        self.assertEqual(plan.skipped_contracts, ())

    def test_short_contract_is_skipped_with_reason(self) -> None:
        plan = make_expanding_folds(
            np.array([3, 3, 3], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
            n_folds=5,
        )
        self.assertEqual(len(plan.records), 0)
        self.assertEqual(plan.skipped_contracts[0]["contract_id"], 3)

    def test_row_identity_assertion_rejects_mismatch(self) -> None:
        ref = {
            "targets": np.array([1.0], dtype=np.float32),
            "contract_ids": np.array([1], dtype=np.int32),
            "processed_row_indices": np.array([5], dtype=np.int64),
            "window_starts": np.array([5], dtype=np.int64),
        }
        cand = {key: value.copy() for key, value in ref.items()}
        assert_same_row_identity(ref, cand)
        cand["window_starts"] = np.array([6], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "window_starts"):
            assert_same_row_identity(ref, cand)


if __name__ == "__main__":
    unittest.main()
