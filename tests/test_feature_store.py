from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from features.feature_store import FeatureBundle, NpzFeatureStore


class FeatureStoreTests(unittest.TestCase):
    def test_feature_store_round_trips_named_neural_branches(self) -> None:
        bundle = FeatureBundle(
            statistical=np.ones((3, 2), dtype=np.float32),
            transformed=np.ones((3, 4), dtype=np.float32) * 2,
            neural_branches={
                "vae": np.ones((3, 5), dtype=np.float32) * 3,
                "contrastive": np.ones((3, 6), dtype=np.float32) * 4,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.npz"
            NpzFeatureStore(str(path)).save(bundle)

            with np.load(path) as data:
                self.assertEqual(set(data.files), {"statistical", "transformed", "vae", "contrastive"})

            loaded = NpzFeatureStore(str(path)).load()
            self.assertEqual(set(loaded.neural_branches), {"vae", "contrastive"})
            np.testing.assert_array_equal(loaded.neural_branches["vae"], bundle.neural_branches["vae"])
            np.testing.assert_array_equal(
                loaded.neural_branches["contrastive"],
                bundle.neural_branches["contrastive"],
            )
            self.assertEqual(loaded.neural.shape, (3, 11))
            self.assertEqual(
                list(loaded.as_branch_dict()),
                ["statistical", "transformed", "vae", "contrastive"],
            )

    def test_feature_store_loads_legacy_empty_neural_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_empty.npz"
            np.savez_compressed(
                path,
                statistical=np.ones((2, 2), dtype=np.float32),
                transformed=np.ones((2, 3), dtype=np.float32),
                neural=np.array([], dtype=np.float32),
            )

            loaded = NpzFeatureStore(str(path)).load()
            self.assertEqual(loaded.neural_branches, {})
            self.assertIsNone(loaded.neural)

    def test_feature_store_loads_legacy_packed_neural_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_packed.npz"
            neural = np.ones((2, 5), dtype=np.float32)
            np.savez_compressed(
                path,
                statistical=np.ones((2, 2), dtype=np.float32),
                transformed=np.ones((2, 3), dtype=np.float32),
                neural=neural,
            )

            loaded = NpzFeatureStore(str(path)).load()
            self.assertEqual(set(loaded.neural_branches), {"neural"})
            np.testing.assert_array_equal(loaded.neural_branches["neural"], neural)

    def test_feature_bundle_accepts_legacy_positional_neural_matrix(self) -> None:
        neural = np.ones((2, 5), dtype=np.float32)
        bundle = FeatureBundle(
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            neural,
        )

        self.assertEqual(set(bundle.neural_branches), {"neural"})
        np.testing.assert_array_equal(bundle.neural_branches["neural"], neural)

    def test_feature_bundle_rejects_neural_branch_row_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same row count"):
            FeatureBundle(
                statistical=np.ones((3, 2), dtype=np.float32),
                transformed=np.ones((3, 4), dtype=np.float32),
                neural_branches={"vae": np.ones((2, 5), dtype=np.float32)},
            )


if __name__ == "__main__":
    unittest.main()
