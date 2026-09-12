from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.phase2_classification.data import alignment_positions


class AlignmentTests(unittest.TestCase):
    def test_alignment_rejects_contract_identity_mismatch(self) -> None:
        labels = {}
        for split in ("train", "test"):
            labels[f"{split}_labels"] = np.asarray([0, 1], dtype=np.int64)
            labels[f"{split}_indices"] = np.asarray([3, 4], dtype=np.int64)
            labels[f"{split}_contract_ids"] = np.asarray([1, 1], dtype=np.int32)
            labels[f"{split}_window_starts"] = np.asarray([7, 8], dtype=np.int64)
            labels[f"{split}_timestamps_ns"] = np.asarray([9, 10], dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alignment.npz"
            payload = {}
            for split in ("train", "test"):
                payload[f"{split}_label_positions"] = np.asarray([0, 1], dtype=np.int64)
                for name in ("indices", "contract_ids", "window_starts", "timestamps_ns", "labels"):
                    payload[f"{split}_{name}"] = labels[f"{split}_{name}"].copy()
            payload["test_contract_ids"][0] = 2
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "test_contract_ids"):
                alignment_positions(labels, path)


if __name__ == "__main__":
    unittest.main()
