from __future__ import annotations

import unittest

import numpy as np

from features.phase5_features import IDENTITY_FIELDS
from features.phase6_volatility_features import identity_mapping


def _identities(rows: list[tuple[str, int, int, int]], split: str) -> dict[str, np.ndarray]:
    columns = list(zip(*rows, strict=True))
    return {
        f"{split}_{field}": np.asarray(columns[index])
        for index, field in enumerate(IDENTITY_FIELDS)
    }


class Phase6VolatilityFeatureAlignmentTests(unittest.TestCase):
    def test_identity_mapping_preserves_target_order(self) -> None:
        source = _identities(
            [("a", 1, 2, 3), ("b", 4, 5, 6), ("c", 7, 8, 9)], "train"
        )
        target = _identities([("c", 7, 8, 9), ("a", 1, 2, 3)], "train")
        np.testing.assert_array_equal(identity_mapping(source, target, "train"), [2, 0])

    def test_identity_mapping_rejects_missing_target(self) -> None:
        source = _identities([("a", 1, 2, 3)], "test")
        target = _identities([("b", 4, 5, 6)], "test")
        with self.assertRaisesRegex(ValueError, "absent"):
            identity_mapping(source, target, "test")

    def test_identity_mapping_rejects_duplicate_source(self) -> None:
        source = _identities([("a", 1, 2, 3), ("a", 1, 2, 3)], "train")
        target = _identities([("a", 1, 2, 3)], "train")
        with self.assertRaisesRegex(ValueError, "not unique"):
            identity_mapping(source, target, "train")


if __name__ == "__main__":
    unittest.main()
