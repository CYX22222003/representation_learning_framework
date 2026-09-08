from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch.nn as nn

from tasks.phase2_classification.runner import RunConfig, run_classification_experiment


class RunnerTests(unittest.TestCase):
    def test_one_epoch_smoke_writes_replayable_artifacts(self) -> None:
        rng = np.random.default_rng(4)
        X_train = rng.normal(size=(12, 4)).astype(np.float32)
        y_train = np.asarray([0, 1, 2] * 4, dtype=np.int64)
        X_test = rng.normal(size=(6, 4)).astype(np.float32)
        y_test = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        identities = {}
        for split, count in (("train", 12), ("test", 6)):
            identities[f"{split}_indices"] = np.arange(count, dtype=np.int64)
            identities[f"{split}_contract_ids"] = np.zeros(count, dtype=np.int32)
            identities[f"{split}_window_starts"] = np.arange(count, dtype=np.int64)
            identities[f"{split}_timestamps_ns"] = np.arange(count, dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = run_classification_experiment(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                identities=identities, model_factory=lambda: nn.Linear(4, 3),
                model_spec={"architecture": "linear-test"}, run_root=root,
                config=RunConfig(
                    model_id="test", protocol_id="P2", epoch_budgets=(1,), seed=2,
                    batch_size=4, learning_rate=1e-3, device="cpu",
                ),
                dataset_manifest={"test_fixture": True},
            )
            self.assertEqual(len(rows), 1)
            self.assertTrue((root / "e1" / "checkpoint.pth").exists())
            self.assertTrue((root / "e1" / "predictions.npz").exists())
            manifest = json.loads((root / "dataset_manifest.json").read_text())
            self.assertTrue(manifest["test_distribution_untouched"])
            with np.load(root / "sampling_indices.npz") as sampling:
                np.testing.assert_array_equal(sampling["positions"], np.arange(12))


if __name__ == "__main__":
    unittest.main()
