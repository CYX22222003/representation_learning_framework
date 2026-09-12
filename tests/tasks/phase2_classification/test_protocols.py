from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from tasks.phase2_classification.protocols import LogitAdjustedCrossEntropy, class_priors, protocol_sample_indices


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.asarray([0] * 3 + [1] * 8 + [2] * 2, dtype=np.int64)

    def test_majority_undersampling_is_deterministic_and_majority_only(self) -> None:
        first, audit = protocol_sample_indices(self.labels, "P1U", seed=7)
        second, _ = protocol_sample_indices(self.labels, "P1U", seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(audit["draw_counts"], [3, 2, 2])
        self.assertEqual(audit["duplicate_count"], 0)

    def test_oversampling_balances_to_largest_class(self) -> None:
        indices, audit = protocol_sample_indices(self.labels, "P1O", seed=3)
        self.assertEqual(audit["draw_counts"], [8, 8, 8])
        self.assertGreater(audit["duplicate_count"], 0)
        self.assertEqual(len(indices), 24)

    def test_logit_adjustment_matches_declared_equation(self) -> None:
        logits = torch.tensor([[0.2, -0.1, 0.4], [1.0, 0.5, -0.2]])
        targets = torch.tensor([2, 0])
        priors = class_priors(self.labels)
        actual = LogitAdjustedCrossEntropy(priors)(logits, targets)
        expected = F.cross_entropy(logits + torch.log(torch.tensor(priors)), targets)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
