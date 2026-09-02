from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tasks.volatility_labels import save_volatility_label_bundle
from models.byol import BYOLEncoder


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase1_framework_runner", ROOT / "scripts" / "train_framework.py")
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import failure is fatal
    raise RuntimeError("Unable to load scripts/train_framework.py")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)

FEATURE_SPEC = importlib.util.spec_from_file_location("phase1_feature_builder", ROOT / "scripts" / "prepare_framework_features.py")
if FEATURE_SPEC is None or FEATURE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("Unable to load scripts/prepare_framework_features.py")
FEATURE_BUILDER = importlib.util.module_from_spec(FEATURE_SPEC)
sys.modules[FEATURE_SPEC.name] = FEATURE_BUILDER
FEATURE_SPEC.loader.exec_module(FEATURE_BUILDER)


class Phase1FrameworkTests(unittest.TestCase):
    def test_volatility_uses_shared_bundle_row_indices(self) -> None:
        train_branches = {"statistical": np.arange(30, dtype=np.float32).reshape(5, 6)}
        test_branches = {"statistical": np.arange(24, dtype=np.float32).reshape(4, 6)}
        bundle = {
            "train_labels": np.array([0.1, 0.2], dtype=np.float32),
            "test_labels": np.array([0.3, 0.4], dtype=np.float32),
            "train_row_indices": np.array([1, 4], dtype=np.int64),
            "test_row_indices": np.array([0, 2], dtype=np.int64),
            "train_contract_ids": np.array([0, 1], dtype=np.int32),
            "test_contract_ids": np.array([0, 1], dtype=np.int32),
            "train_window_starts": np.array([1, 4], dtype=np.int64),
            "test_window_starts": np.array([0, 2], dtype=np.int64),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            labels = Path(tmpdir) / "volatility.npz"
            save_volatility_label_bundle(bundle, {"label_mode": "next_stride_one_window_realized_volatility", "horizon": 1}, labels)
            config = RUNNER.FrameworkConfig(task="volatility_prediction", labels_npz=str(labels))
            x_train, y_train, x_test, y_test, _ = RUNNER.build_volatility_data(
                train_branches, test_branches, {"train_size": 5, "test_size": 4}, config
            )

        np.testing.assert_array_equal(x_train["statistical"], train_branches["statistical"][[1, 4]])
        np.testing.assert_array_equal(x_test["statistical"], test_branches["statistical"][[0, 2]])
        np.testing.assert_array_equal(y_train, bundle["train_labels"])
        np.testing.assert_array_equal(y_test, bundle["test_labels"])

    def test_volatility_model_has_regression_output(self) -> None:
        config = RUNNER.FrameworkConfig(task="volatility_prediction", labels_npz="labels.npz")
        model = RUNNER.make_model({"byol": 128}, config)
        output = model({"byol": torch.randn(3, 128)})
        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertEqual(RUNNER.target_kind("volatility_prediction"), "regression")

    def test_byol_extractor_uses_online_backbone_embedding(self) -> None:
        model = BYOLEncoder(input_dim=5, hidden_dim=16, projection_dim=12, predictor_hidden_dim=10)
        checkpoint = {
            "model_state_dict": model.state_dict(), "seq_len": 8, "input_dim": 5,
            "hidden_dim": 16, "projection_dim": 12, "predictor_hidden_dim": 10,
            "completed_epoch": 7,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "byol.pth"
            torch.save(checkpoint, checkpoint_path)
            embeddings, manifest = FEATURE_BUILDER._extract_byol_embeddings(
                np.random.default_rng(0).normal(size=(3, 8, 5)).astype(np.float32),
                checkpoint_path, batch_size=2, device=torch.device("cpu"),
            )
        self.assertEqual(embeddings.shape, (3, 16))
        self.assertEqual(embeddings.dtype, np.float32)
        self.assertEqual(manifest["embedding_source"], "BYOLEncoder.encode(...)[online_backbone]")


if __name__ == "__main__":
    unittest.main()
