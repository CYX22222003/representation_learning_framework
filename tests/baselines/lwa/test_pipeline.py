from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import training.phase6_7_lwa as pipeline
from training.phase6_7_downstream import Phase67DownstreamConfig, build_external_head
from training.phase6_7_lwa import (
    LWAInputScaler,
    Phase67LWATrainingConfig,
    build_lwa_view_cache,
    drop_last_batch_indices,
    validate_lwa_view_cache,
)


class _FastCWT:
    def __init__(self, config) -> None:
        self.config = config

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        values = batch.transpose(1, 2).unsqueeze(2)
        return values.expand(-1, -1, 48, -1).abs().contiguous()


class LWAPipelineTests(unittest.TestCase):
    def test_replay_validation_record_preserves_warning_and_thresholds(self) -> None:
        diagnostics = [
            {
                "representation": "wavelet",
                "stage": "joint",
                "epoch": 5,
                "severity": "warning",
                "reason": "accepted_cross_device_drift",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = pipeline._write_replay_validation(
                Path(directory), walk=1, diagnostics=diagnostics, valid=True
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(payload["valid"])
        self.assertEqual(payload["warning_count"], 1)
        self.assertEqual(payload["critical_count"], 0)
        self.assertEqual(
            payload["thresholds"]["cross_device_relative_l2_max"],
            pipeline.CROSS_DEVICE_RELATIVE_L2_TOLERANCE,
        )

    def test_probe_replay_records_small_cross_device_drift_as_warning(self) -> None:
        expected = {
            name: torch.linspace(-1000.0, 1000.0, 512)
            for name in ("time", "fourier", "wavelet")
        }
        replayed = {name: value.clone() for name, value in expected.items()}
        replayed["wavelet"] = expected["wavelet"] * (1.0 + 2e-4)

        diagnostics = pipeline._probe_replay_diagnostics(expected, replayed)

        self.assertEqual([row["severity"] for row in diagnostics], ["pass", "pass", "warning"])
        self.assertEqual(diagnostics[-1]["reason"], "accepted_cross_device_drift")
        self.assertLessEqual(
            diagnostics[-1]["relative_l2"],
            pipeline.CROSS_DEVICE_RELATIVE_L2_TOLERANCE,
        )

    def test_probe_replay_keeps_material_or_structural_mismatch_critical(self) -> None:
        expected = {
            name: torch.linspace(-10.0, 10.0, 512)
            for name in ("time", "fourier", "wavelet")
        }
        replayed = {name: value.clone() for name, value in expected.items()}
        replayed["fourier"] = replayed["fourier"][:-1]
        replayed["wavelet"] = expected["wavelet"] * 1.01

        diagnostics = pipeline._probe_replay_diagnostics(expected, replayed)

        self.assertEqual(diagnostics[1]["severity"], "critical")
        self.assertEqual(diagnostics[1]["reason"], "shape_mismatch")
        self.assertEqual(diagnostics[2]["severity"], "critical")
        self.assertEqual(diagnostics[2]["reason"], "material_numerical_mismatch")

    def test_scaler_uses_train_population_and_constant_fallback(self) -> None:
        values = np.zeros((3, 64, 5), dtype=np.float32)
        values[..., 0] = 7.0
        values[..., 1] = np.arange(64, dtype=np.float32)
        scaler = LWAInputScaler.fit(values)
        self.assertEqual(float(scaler.scale[0]), 1.0)
        transformed = scaler.transform_numpy(values)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.all(transformed[..., 0] == 0.0))

    def test_drop_last_sampler_is_replayable_and_uses_only_full_batches(self) -> None:
        first = drop_last_batch_indices(39070, 128, torch.Generator().manual_seed(11))
        second = drop_last_batch_indices(39070, 128, torch.Generator().manual_seed(11))
        self.assertEqual(len(first), 305)
        self.assertTrue(all(len(batch) == 128 for batch in first))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertEqual(len(torch.cat(first).unique()), 39040)

    def test_frozen_pretraining_and_native_downstream_width(self) -> None:
        config = Phase67LWATrainingConfig(walk=1, device="cpu")
        self.assertEqual(config.physical_batch_size, 128)
        self.assertEqual((config.joint_epochs, config.mapper_epochs), (50, 50))
        downstream = Phase67DownstreamConfig(
            "classification_h2", "lwa_frozen", 1, input_dim=384, device="cpu"
        )
        self.assertEqual(build_external_head(downstream)(torch.randn(2, 384)).shape, (2, 3))
        with self.assertRaises(ValueError):
            Phase67DownstreamConfig(
                "classification_h2", "lwa_frozen", 1, input_dim=128, device="cpu"
            )

    def test_cache_is_atomic_typed_and_hash_validated(self) -> None:
        rng = np.random.default_rng(4)
        sequences = rng.normal(size=(3, 64, 5)).astype(np.float32)
        source = {"bundle": {}, "identity_hash": "test-identity"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "source.npz"
            dataset.write_bytes(b"source")
            Path(f"{dataset}.manifest.json").write_text("{}", encoding="utf-8")
            cache = root / "cache"
            with (
                patch.object(pipeline, "_load_encoder_population", return_value=(sequences, source)),
                patch.object(pipeline, "MorletCWT", _FastCWT),
            ):
                result = build_lwa_view_cache(dataset, cache, walk=1, chunk_size=2)
                replay = validate_lwa_view_cache(dataset, cache, walk=1)
            self.assertTrue(result["valid"] and replay["valid"])
            self.assertEqual(np.load(cache / "time.npy", mmap_mode="r").dtype, np.float32)
            self.assertEqual(np.load(cache / "fourier.npy", mmap_mode="r").dtype, np.complex64)
            self.assertEqual(np.load(cache / "wavelet.npy", mmap_mode="r").shape, (3, 5, 48, 64))
            self.assertFalse(cache.with_name(".cache.work").exists())


if __name__ == "__main__":
    unittest.main()
