from __future__ import annotations

import unittest

import numpy as np

from baselines.sgn.grouping import (
    brownian_distance_correlation,
    build_group_initialization,
    contract_stratified_indices,
    deterministic_kmeans,
    period_candidates,
)


class GroupingTests(unittest.TestCase):
    def test_contract_stratified_sampler_is_deterministic_and_complete(self) -> None:
        conditions = np.asarray(["a"] * 8 + ["b"] * 4 + ["c"] * 3)
        dates = np.arange(len(conditions), dtype=np.int64)
        first = contract_stratified_indices(conditions, dates, sample_size=9)
        second = contract_stratified_indices(conditions, dates, sample_size=9)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 9)
        self.assertEqual(set(conditions[first]), {"a", "b", "c"})

    def test_bdc_is_symmetric_finite_and_unit_diagonal(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(size=(24, 16, 5))
        values[:, :, 1] = values[:, :, 0] + 0.01 * rng.normal(size=(24, 16))
        matrix = brownian_distance_correlation(values)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(matrix), np.ones(5), atol=0.0)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertGreater(matrix[0, 1], matrix[0, 4])

    def test_kmeans_canonicalizes_smaller_group_first(self) -> None:
        points = np.asarray([[1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [-2.0, -2.0]])
        labels, centroids = deterministic_kmeans(points, 2)
        self.assertEqual(labels.tolist(), [1, 1, 1, 0])
        self.assertEqual(centroids.shape, (2, 2))

    def test_period_candidates_exclude_dc_and_full_period(self) -> None:
        time = np.arange(64, dtype=np.float64)
        signal = np.sin(2 * np.pi * time / 16.0)
        values = np.repeat(signal[None, :, None], 12, axis=0)
        values = np.repeat(values, 5, axis=2)
        bins, periods = period_candidates(values)
        self.assertEqual(int(bins[0]), 4)
        self.assertEqual(int(periods[0]), 16)
        self.assertNotIn(0, bins)
        self.assertNotIn(1, bins)

    def test_build_initialization_returns_approved_shapes(self) -> None:
        rng = np.random.default_rng(3)
        time = np.arange(64, dtype=np.float64)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(30, 1, 1))
        amplitudes = rng.uniform(0.5, 1.5, size=(30, 1, 1))
        base = amplitudes * np.sin(2.0 * np.pi * time[None, :, None] / 16.0 + phases)
        base += 0.05 * rng.normal(size=(30, 64, 1))
        ohlc = np.repeat(base, 4, axis=2) + 0.01 * rng.normal(size=(30, 64, 4))
        volume = rng.normal(size=(30, 64, 1))
        values = np.concatenate((ohlc, volume), axis=2).astype(np.float32)
        conditions = np.asarray(["a"] * 15 + ["b"] * 15)
        dates = np.arange(30, dtype=np.int64)
        result = build_group_initialization(values, conditions, dates, sample_size=20)
        self.assertEqual(result.initial_logits.shape, (5, 2))
        self.assertEqual(result.labels.tolist(), [1, 1, 1, 1, 0])
        self.assertIn(16, result.fft_periods)


if __name__ == "__main__":
    unittest.main()
