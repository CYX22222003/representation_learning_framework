from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from baselines.timedart import TimeDARTConfig, TimeDARTPretrainer
from data_processing.phase6_7_timedart_data import (
    load_timedart_encoder_population,
    load_timedart_task_contexts,
)
from features.phase5_features import IDENTITY_FIELDS
from features.phase6_7_timedart_features import (
    COORDINATE_SLICES,
    extract_timedart_embeddings,
)
from training.phase6_7_downstream import Phase67DownstreamConfig
from training.phase6_7_timedart import (
    FROZEN_ENCODER_SCHEMA_VERSION,
    _replay_diagnostics,
    drop_last_batch_indices,
)


def _identities(prefix: str, rows: int) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_condition_ids": np.asarray([f"c-{index}" for index in range(rows)]),
        f"{prefix}_window_start_ns": np.arange(rows, dtype=np.int64),
        f"{prefix}_decision_date_ns": np.arange(rows, dtype=np.int64) + 10,
        f"{prefix}_decision_availability_ns": np.arange(rows, dtype=np.int64) + 20,
    }


class TimeDARTPipelineTests(unittest.TestCase):
    def test_data_loaders_open_only_declared_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            arrays: dict[str, np.ndarray] = {
                "encoder_train_sequences": np.ones((5, 64, 5), dtype=np.float32),
                "train_sequences": np.ones((3, 64, 5), dtype=np.float32),
                "test_sequences": np.ones((2, 64, 5), dtype=np.float32),
                # This array cannot be loaded with allow_pickle=False. The
                # TimeDART preparation path must leave it untouched.
                "train_classification_labels": np.asarray([object()], dtype=object),
            }
            arrays.update(_identities("encoder_train", 5))
            arrays.update(_identities("train", 3))
            arrays.update(_identities("test", 2))
            np.savez(path, **arrays)
            encoder, encoder_record = load_timedart_encoder_population(path, walk=1)
            contexts, task_record = load_timedart_task_contexts(path)
            self.assertEqual(encoder.shape, (5, 64, 5))
            self.assertEqual(contexts["train_sequences"].shape, (3, 64, 5))
            self.assertFalse(encoder_record["targets_loaded"])
            self.assertFalse(task_record["targets_loaded"])
            self.assertNotIn("train_classification_labels", task_record["arrays_loaded"])

    def test_drop_last_sampler_contract(self) -> None:
        batches, dropped = drop_last_batch_indices(
            39_070, 16, torch.Generator().manual_seed(0)
        )
        self.assertEqual(dropped, 14)
        self.assertEqual(len(batches), 2_441)
        self.assertTrue(all(len(batch) == 16 for batch in batches))
        flattened = torch.cat(batches)
        self.assertEqual(len(torch.unique(flattened)), len(flattened))

    def test_epoch50_checkpoint_extracts_170_coordinates(self) -> None:
        torch.manual_seed(3)
        model = TimeDARTPretrainer().eval()
        values = np.random.default_rng(4).normal(size=(3, 64, 5)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            torch.save(
                {
                    "schema_version": FROZEN_ENCODER_SCHEMA_VERSION,
                    "method": "timedart_frozen",
                    "walk": 1,
                    "completed_epoch": 50,
                    "model_config": TimeDARTConfig().to_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                },
                checkpoint,
            )
            features = extract_timedart_embeddings(
                values, checkpoint, walk=1, device="cpu", batch_size=2
            )
        self.assertEqual(features.shape, (3, 170))
        np.testing.assert_allclose(features[:, 160:165], values.mean(axis=1), rtol=1e-5)
        expected_std = np.sqrt(values.var(axis=1) + 1e-5)
        np.testing.assert_allclose(features[:, 165:170], expected_std, rtol=1e-5)
        self.assertEqual(COORDINATE_SLICES["volume_pooled"], [128, 160])

    def test_replay_tolerance_distinguishes_warning_and_material_drift(self) -> None:
        expected = torch.linspace(1.0, 2.0, 100)
        small = expected * (1.0 + 1e-5)
        large = expected * 1.01
        self.assertTrue(_replay_diagnostics(expected, expected)["strict"])
        self.assertTrue(_replay_diagnostics(small, expected)["accepted"])
        self.assertFalse(_replay_diagnostics(large, expected)["accepted"])

    def test_common_probe_registers_native_width(self) -> None:
        config = Phase67DownstreamConfig(
            "classification_h2",
            "timedart_frozen",
            walk=1,
            input_dim=170,
            device="cpu",
        )
        self.assertEqual(config.input_dim, 170)
        with self.assertRaises(ValueError):
            Phase67DownstreamConfig(
                "classification_h2",
                "timedart_frozen",
                walk=1,
                input_dim=128,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
