from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from data_processing.phase5_walks import sha256_arrays, sha256_file
from training.phase6_9_xlstm_mixer import (
    DATA_SCHEMA_VERSION,
    IDENTITY_FIELDS,
    METHOD,
    NATIVE_HOUR_NS,
    Phase69XLSTMMixerConfig,
    evaluate_xlstm_predictions,
    implementation_fingerprint,
    load_xlstm_mixer_data,
    shuffled_batches,
    smoke_test_xlstm_mixer_training,
    validate_xlstm_mixer_arrays,
)


class TinyStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale


def synthetic_bundle() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    generator = np.random.default_rng(8)

    def valid_ohlcv(rows: int, length: int) -> np.ndarray:
        open_ = generator.random((rows, length), dtype=np.float32)
        close = generator.random((rows, length), dtype=np.float32)
        lower = np.minimum(open_, close)
        upper = np.maximum(open_, close)
        low = lower * generator.random((rows, length), dtype=np.float32)
        high = upper + (1.0 - upper) * generator.random(
            (rows, length), dtype=np.float32
        )
        volume = generator.random((rows, length), dtype=np.float32)
        return np.stack((open_, high, low, close, volume), axis=-1)

    counts = {"train": 3, "evaluation": 2}
    arrays: dict[str, np.ndarray] = {}
    identities: dict[str, str] = {}
    for split, count in counts.items():
        contexts = valid_ohlcv(count, 64)
        targets = valid_ohlcv(count, 8)
        if split == "train":
            decision = np.arange(10, 10 + count, dtype=np.int64) * NATIVE_HOUR_NS
            prefix = "train"
        else:
            decision = np.arange(101, 101 + count, dtype=np.int64) * NATIVE_HOUR_NS
            prefix = "evaluation"
        target_time = decision[:, None] + NATIVE_HOUR_NS * np.arange(1, 9)[None, :]
        metadata = {
            "row_ids": np.asarray([f"{prefix}-{index}" for index in range(count)]),
            "condition_ids": np.asarray([f"condition-{index % 2}" for index in range(count)]),
            "segment_ids": np.asarray([f"segment-{index % 2}" for index in range(count)]),
            "decision_time_ns": decision,
            "decision_availability_ns": decision + NATIVE_HOUR_NS,
            "target_time_ns": target_time,
            "target_availability_ns": target_time + NATIVE_HOUR_NS,
            "target_observed": np.ones((count, 8), dtype=bool),
            "target_imputed": np.zeros((count, 8), dtype=bool),
            "current_close": contexts[:, -1, 3].copy(),
            "target_close": targets[:, -1, 3].copy(),
        }
        arrays[f"{split}_contexts"] = contexts
        arrays[f"{split}_targets"] = targets
        for name, values in metadata.items():
            arrays[f"{split}_{name}"] = values
        identities[split] = sha256_arrays(
            *(metadata[field] for field in IDENTITY_FIELDS)
        )
    manifest: dict[str, object] = {
        "schema_version": DATA_SCHEMA_VERSION,
        "method": METHOD,
        "method_id": "XM-MV8",
        "walk": 1,
        "validation_passed": True,
        "full_path_policy": {
            "horizon_hours": list(range(1, 9)),
            "same_condition": True,
            "same_segment": True,
            "all_target_bars_observed": True,
            "target_imputation_allowed": False,
            "availability_only_intersection": True,
        },
        "channel_order": ["open", "high", "low", "close", "volume"],
        "additional_xlstm_channel_scaler": False,
        "volume_transform": {"mean": 0.0, "denominator": 1.0, "sha256": "f" * 64},
        "row_counts": counts,
        "identity_hashes": identities,
        "training_cutoff_ns": 100 * NATIVE_HOUR_NS,
        "evaluation_start_ns": 100 * NATIVE_HOUR_NS,
        "evaluation_end_ns": 200 * NATIVE_HOUR_NS,
    }
    return arrays, manifest


class XLSTMMixerTrainingTests(unittest.TestCase):
    def test_training_config_is_fixed_except_admitted_batch(self) -> None:
        config = Phase69XLSTMMixerConfig(walk=1, batch_size=256)
        self.assertEqual(config.snapshot_epochs, (5, 15, 50))
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.gradient_clip_norm, 1.0)
        with self.assertRaises(ValueError):
            Phase69XLSTMMixerConfig(walk=1, epochs=49)
        self.assertEqual(Phase69XLSTMMixerConfig(walk=1, backend="vanilla").backend, "vanilla")
        with self.assertRaises(ValueError):
            Phase69XLSTMMixerConfig(walk=1, backend="unknown")
        with self.assertRaises(ValueError):
            Phase69XLSTMMixerConfig(walk=1, device="cpu")

    def test_shuffled_batches_keep_remainder_and_cover_every_row(self) -> None:
        generator = torch.Generator().manual_seed(0)
        batches, remainder = shuffled_batches(10, 4, generator)
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2])
        self.assertEqual(remainder, 2)
        combined = torch.cat(batches)
        self.assertEqual(sorted(combined.tolist()), list(range(10)))

    def test_data_contract_replays_identity_time_and_endpoint_rules(self) -> None:
        arrays, manifest = synthetic_bundle()
        result = validate_xlstm_mixer_arrays(
            arrays, manifest, enforce_frozen_counts=False
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["train"]["rows"], 3)
        self.assertEqual(result["evaluation"]["rows"], 2)

        invalid = dict(arrays)
        invalid["evaluation_target_imputed"] = arrays[
            "evaluation_target_imputed"
        ].copy()
        invalid["evaluation_target_imputed"][0, 0] = True
        with self.assertRaisesRegex(ValueError, "imputed target"):
            validate_xlstm_mixer_arrays(
                invalid, manifest, enforce_frozen_counts=False
            )

    def test_npz_loader_validates_hash_and_preserves_metadata(self) -> None:
        arrays, manifest = synthetic_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "walk1.npz"
            np.savez_compressed(path, **arrays)
            manifest["dataset_sha256"] = sha256_file(path)
            manifest_path = Path(f"{path}.manifest.json")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_xlstm_mixer_data(
                path, walk=1, enforce_frozen_counts=False
            )
        self.assertEqual(loaded.train_contexts.shape, (3, 64, 5))
        self.assertEqual(loaded.evaluation_targets.shape, (2, 8, 5))
        np.testing.assert_array_equal(
            loaded.evaluation_metadata["row_ids"],
            arrays["evaluation_row_ids"],
        )

    def test_evaluation_keeps_full_path_and_extracts_eighth_close(self) -> None:
        arrays, manifest = synthetic_bundle()
        validation = validate_xlstm_mixer_arrays(
            arrays, manifest, enforce_frozen_counts=False
        )
        from training.phase6_9_xlstm_mixer import XLSTMMixerDataset

        data = XLSTMMixerDataset(
            train_contexts=arrays["train_contexts"],
            train_targets=arrays["train_targets"],
            evaluation_contexts=arrays["evaluation_contexts"],
            evaluation_targets=arrays["evaluation_targets"],
            train_metadata={
                name: arrays[f"train_{name}"]
                for name in (*IDENTITY_FIELDS, "target_observed", "target_imputed", "current_close", "target_close")
            },
            evaluation_metadata={
                name: arrays[f"evaluation_{name}"]
                for name in (*IDENTITY_FIELDS, "target_observed", "target_imputed", "current_close", "target_close")
            },
            manifest=manifest,
            dataset_sha256="d" * 64,
            manifest_sha256="m" * 64,
            train_identity_sha256=validation["train"]["identity_sha256"],
            evaluation_identity_sha256=validation["evaluation"]["identity_sha256"],
        )
        payload, saved, summary = evaluate_xlstm_predictions(
            data.evaluation_targets.copy(), data
        )
        self.assertEqual(summary["mae"], 0.0)
        self.assertEqual(summary["full_path_mae"], 0.0)
        np.testing.assert_array_equal(
            saved["prediction_future_price"], arrays["evaluation_targets"][:, 7, 3]
        )
        self.assertIn("all five channels", payload["extra_supervision_disclosure"])
        # Headline metrics must not silently re-quantize original endpoint labels.
        from dataclasses import replace
        exact_metadata = dict(data.evaluation_metadata)
        exact_metadata["target_close"] = exact_metadata["target_close"].astype(np.float64) + 1e-9
        _, precise_saved, precise_summary = evaluate_xlstm_predictions(
            data.evaluation_targets.copy(), replace(data, evaluation_metadata=exact_metadata)
        )
        np.testing.assert_array_equal(precise_saved["target_future_price"], exact_metadata["target_close"])
        self.assertGreater(precise_summary["mae"], 0.0)

    def test_vanilla_training_smoke_and_implementation_hash(self) -> None:
        with patch(
            "baselines.xlstm_mixer.model.build_xlstm_stack",
            side_effect=lambda *_args, **_kwargs: TinyStack(),
        ):
            result = smoke_test_xlstm_mixer_training()
        self.assertTrue(result["valid"])
        self.assertEqual(result["prediction_shape"], [2, 8, 5])
        self.assertTrue(result["checkpoint_replay"])
        self.assertEqual(len(implementation_fingerprint()), 64)


if __name__ == "__main__":
    unittest.main()
