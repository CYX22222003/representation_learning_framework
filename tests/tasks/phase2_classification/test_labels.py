from __future__ import annotations

import unittest

import numpy as np

from tasks.phase2_classification.labels import build_movement_label_bundle, movement_classes, validate_label_bundle


def contract(closes: list[float], seq_len: int = 3) -> np.ndarray:
    result = np.zeros((len(closes), seq_len, 5), dtype=np.float32)
    result[:, :, 3] = np.asarray(closes, dtype=np.float32)[:, None]
    return result


class MovementLabelTests(unittest.TestCase):
    def test_exact_thresholds_are_stable(self) -> None:
        labels = movement_classes(np.asarray([-0.006, -0.005, 0.0, 0.005, 0.006]), 0.005)
        np.testing.assert_array_equal(labels, np.asarray([0, 1, 1, 1, 2]))

    def test_horizon_never_crosses_split_or_contract(self) -> None:
        bundle = build_movement_label_bundle(
            [contract([0.10, 0.20, 0.205, 0.40, 0.50, 0.60]), contract([0.70, 0.60, 0.59, 0.40, 0.30])],
            horizon=1, threshold=0.05, train_ratio=0.6,
        )
        np.testing.assert_array_equal(bundle["train_indices"], np.asarray([0, 1, 3, 4]))
        np.testing.assert_array_equal(bundle["test_indices"], np.asarray([0, 1, 3]))
        np.testing.assert_array_equal(bundle["train_contract_ids"], np.asarray([0, 0, 1, 1]))
        self.assertEqual(bundle["test_contract_ids"].tolist(), [0, 0, 1])
        validation = validate_label_bundle(bundle, train_size=6, test_size=5)
        self.assertEqual(validation["train_count"], 4)

    def test_non_finite_current_or_future_is_dropped(self) -> None:
        bundle = build_movement_label_bundle(
            [contract([0.1, 0.2, np.nan, 0.4, 0.5, 0.7, 0.9, 0.8])],
            horizon=1, threshold=0.05, train_ratio=0.75,
        )
        self.assertEqual(bundle["train_indices"].tolist(), [0, 3, 4])

    def test_validator_rejects_label_that_disagrees_with_delta(self) -> None:
        bundle = build_movement_label_bundle(
            [contract([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.8])],
            horizon=1, threshold=0.05, train_ratio=0.75,
        )
        bundle["train_labels"][0] = 1
        with self.assertRaisesRegex(ValueError, "do not match delta"):
            validate_label_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
