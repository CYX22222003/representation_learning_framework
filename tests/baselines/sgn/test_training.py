from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from training.phase6_9_sgn import (
    SGNTrainingConfig,
    _batches,
    _classification_evaluation,
    run_root,
    run_sgn_training,
    smoke_test_sgn,
)


class TrainingTests(unittest.TestCase):
    def test_config_is_frozen(self) -> None:
        self.assertEqual(SGNTrainingConfig(1).batch_size, 256)
        with self.assertRaises(ValueError):
            SGNTrainingConfig(1, learning_rate=1e-4)

    def test_shuffled_batches_cover_every_row_and_remainder(self) -> None:
        batches, remainder = _batches(515, 256, torch.Generator().manual_seed(0))
        values = torch.cat(batches).numpy()
        self.assertEqual(remainder, 3)
        np.testing.assert_array_equal(np.sort(values), np.arange(515))

    def test_classification_evaluation_preserves_metadata(self) -> None:
        data = {
            "y_train": np.asarray([0, 1, 1, 2]),
            "y_test": np.asarray([0, 1, 2]),
            "test_metadata": {"condition_ids": np.asarray(["a", "b", "c"])},
        }
        payload, arrays = _classification_evaluation(np.eye(3, dtype=np.float32), data)
        self.assertEqual(payload["sgn_c"]["macro_f1"], 1.0)
        np.testing.assert_array_equal(arrays["condition_ids"], data["test_metadata"]["condition_ids"])

    def test_cpu_smoke_covers_update_remainder_and_replay(self) -> None:
        result = smoke_test_sgn(batch_size=2)
        self.assertTrue(result["valid"])
        self.assertTrue(result["one_row_remainder_passed"])
        self.assertTrue(result["same_backend_checkpoint_replay"])
        self.assertGreater(result["parameter_count"], 0)

    def test_interrupted_resume_matches_uninterrupted_fixture(self) -> None:
        rng = np.random.default_rng(9)
        train = rng.normal(size=(4, 64, 5)).astype(np.float32)
        evaluation = rng.normal(size=(3, 64, 5)).astype(np.float32)
        data = {
            "X_train": train, "y_train": np.asarray([0, 1, 1, 2], dtype=np.int64),
            "X_test": evaluation, "y_test": np.asarray([0, 1, 2], dtype=np.int64),
            "train_metadata": {}, "test_metadata": {"condition_ids": np.asarray(["a", "b", "c"])},
        }
        data_manifest = {"walk": 1, "dataset_sha256": "fixture"}
        init = np.asarray([[0.0, 1.0]] * 4 + [[1.0, 0.0]], dtype=np.float32)
        init_manifest = {"walk": 1, "initialization_npz_sha256": "fixture"}
        config = SGNTrainingConfig(1, device="cpu")
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as resumed_dir:
            mocks = (
                patch("training.phase6_9_sgn.load_sgn_data", return_value=(data, data_manifest)),
                patch("training.phase6_9_sgn.load_initial_logits", return_value=(init, init_manifest)),
                patch("training.phase6_9_sgn.implementation_fingerprint", return_value="fixture"),
            )
            with mocks[0], mocks[1], mocks[2]:
                run_sgn_training(Path(first_dir), config, stop_after_epoch=2, _fixture=True)
                run_sgn_training(Path(resumed_dir), config, stop_after_epoch=1, _fixture=True)
                run_sgn_training(Path(resumed_dir), config, stop_after_epoch=2, _fixture=True)
            first = torch.load(run_root(Path(first_dir), 1) / "resume.pth", map_location="cpu", weights_only=True)
            resumed = torch.load(run_root(Path(resumed_dir), 1) / "resume.pth", map_location="cpu", weights_only=True)
            self.assertEqual(first["completed_epoch"], resumed["completed_epoch"])
            for left, right in zip(first["history"], resumed["history"]):
                self.assertEqual({k: v for k, v in left.items() if k != "seconds"},
                                 {k: v for k, v in right.items() if k != "seconds"})
            for name, value in first["model_state_dict"].items():
                torch.testing.assert_close(value, resumed["model_state_dict"][name], rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
