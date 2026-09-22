from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from features.phase5_features import (
    BRANCH_DIMS,
    _load_valid_chunk,
    array_sha256,
    extract_deterministic_features,
)


class Phase5FeatureTests(unittest.TestCase):
    def test_array_hash_includes_shape_and_dtype(self) -> None:
        values = np.arange(8, dtype=np.float32)
        self.assertNotEqual(array_sha256(values), array_sha256(values.reshape(2, 4)))
        self.assertNotEqual(array_sha256(values), array_sha256(values.astype(np.float64)))

    def test_deterministic_extraction_is_restartable(self) -> None:
        rng = np.random.default_rng(4)
        sequences = rng.normal(size=(3, 64, 5)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            first = extract_deterministic_features(
                sequences, Path(directory), "train", chunk_size=2, workers=1
            )
            second = extract_deterministic_features(
                sequences, Path(directory), "train", chunk_size=2, workers=1
            )
        self.assertEqual(first[0].shape, (3, BRANCH_DIMS["statistical"]))
        self.assertEqual(first[1].shape, (3, BRANCH_DIMS["transformed"]))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
