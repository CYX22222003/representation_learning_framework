"""Source-join tests using the repository's contract-safe Phase 5 fixture."""
import unittest
import numpy as np
import pandas as pd

from data_processing.phase5_walks import build_phase5_walk_bundle
from data_processing.phase6_9_xlstm_mixer import build_future_paths
from tests.data_processing.test_phase5_walks import candles, metadata, spec


class FuturePathTests(unittest.TestCase):
    def build(self, source):
        original = build_phase5_walk_bundle(source, metadata(), spec(), horizon=8,
                                             task_role="exploratory_raw_delta_h8")
        arrays, manifest = build_future_paths(source, original.arrays, original.manifest,
                                              enforce_frozen_counts=False)
        return original, arrays, manifest

    def test_excludes_intermediate_imputed_bars_preserving_original_order(self):
        source = candles()
        original, arrays, manifest = self.build(source)
        rows = arrays["train_original_row_indices"]
        self.assertTrue(np.all(np.diff(rows) > 0))
        np.testing.assert_array_equal(arrays["train_contexts"], original.arrays["train_sequences"][rows])
        np.testing.assert_array_equal(arrays["train_target_close"], original.arrays["train_target_close"][rows])
        self.assertEqual(arrays["train_target_close"].dtype, np.float64)
        times = arrays["train_target_time_ns"]
        self.assertFalse(np.any(times == source.loc[70, "date"].value))
        self.assertLess(len(rows), len(original.arrays["train_sequences"]))
        self.assertEqual(manifest["matched_comparator_reruns_required"], ["H0-D0", "Raw LSTM"])

    def test_reuses_upstream_volume_transform_for_context_and_targets(self):
        source = candles()
        original, arrays, manifest = self.build(source)
        volume = original.manifest["preprocessing"]["volume"]
        lookup = source.set_index("date")["volume"]
        # UTC nanoseconds are used directly, avoiding implicit timezone parsing.
        raw = lookup.reindex(pd.to_datetime(arrays["evaluation_target_time_ns"].ravel(), utc=True)).to_numpy().reshape(-1, 8)
        np.testing.assert_array_equal(arrays["evaluation_targets"][:, :, 4],
            ((raw - volume["mean"]) / volume["denominator"]).astype(np.float32))
        self.assertEqual(manifest["volume_transform"]["mean"], volume["mean"])

    def test_evaluation_changes_cannot_change_training_paths(self):
        source = candles()
        changed = source.copy()
        evaluation = changed["date"].add(pd.Timedelta(hours=1)).ge(spec().cutoff)
        changed.loc[evaluation, "volume"] *= 1000
        _, first, m1 = self.build(source)
        _, second, m2 = self.build(changed)
        self.assertEqual(m1["volume_transform"], m2["volume_transform"])
        for name in first:
            if name.startswith("train_"):
                np.testing.assert_array_equal(first[name], second[name])

    def test_source_context_tampering_is_rejected(self):
        source = candles()
        original = build_phase5_walk_bundle(source, metadata(), spec(), horizon=8,
                                             task_role="exploratory_raw_delta_h8")
        original.arrays["train_sequences"][0, 0, 4] += 1
        with self.assertRaisesRegex(ValueError, "scaled contexts"):
            build_future_paths(source, original.arrays, original.manifest, enforce_frozen_counts=False)
