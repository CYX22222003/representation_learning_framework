from __future__ import annotations

import unittest
import copy

import numpy as np
import torch

from baselines.saurl_ts import SaURLConfig, build_saurl
from features.phase6_7_external_features import OUTPUT_DIM, TASKS
from training.phase6_7_downstream import (
    METHOD,
    Phase67DownstreamConfig,
    apply_external_standardizer,
    build_external_head,
    fit_external_standardizer,
)
from training.phase6_7_external_encoders import (
    Phase67ExternalEncoderConfig,
    cross_device_tensors_close,
    no_drop_batch_indices,
)
import training.phase6_7_external_encoders as encoder_runner
from training.phase5_encoder import set_seed


class NoDropBatchTests(unittest.TestCase):
    def test_singleton_remainder_is_merged_and_every_row_is_used_once(self) -> None:
        generator = torch.Generator().manual_seed(11)
        batches = no_drop_batch_indices(65, 32, generator)
        self.assertEqual([len(batch) for batch in batches], [32, 33])
        values = torch.cat(batches)
        self.assertTrue(torch.equal(values.sort().values, torch.arange(65)))

    def test_non_singleton_remainder_is_retained(self) -> None:
        generator = torch.Generator().manual_seed(11)
        batches = no_drop_batch_indices(66, 32, generator)
        self.assertEqual([len(batch) for batch in batches], [32, 32, 2])

    def test_sampler_is_replayable_from_generator_state(self) -> None:
        first = torch.Generator().manual_seed(7)
        state = first.get_state()
        expected = no_drop_batch_indices(97, 32, first)
        replay = torch.Generator()
        replay.set_state(state)
        actual = no_drop_batch_indices(97, 32, replay)
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            self.assertTrue(torch.equal(left, right))


class NativeWidthProbeTests(unittest.TestCase):
    def test_generic_standardizer_supports_native_width_and_constant_columns(self) -> None:
        train = np.arange(6 * OUTPUT_DIM, dtype=np.float32).reshape(6, OUTPUT_DIM)
        train[:, 3] = 4.0
        scaler = fit_external_standardizer(train)
        self.assertEqual(scaler["mean"].shape, (OUTPUT_DIM,))
        self.assertEqual(float(scaler["scale"][3]), 1.0)
        transformed = apply_external_standardizer(train, scaler)
        self.assertEqual(transformed.shape, train.shape)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.all(transformed[:, 3] == 0.0))

    def test_standardizer_clips_evaluation_values(self) -> None:
        scaler = fit_external_standardizer(np.asarray([[0.0], [1.0]], dtype=np.float32))
        transformed = apply_external_standardizer(
            np.asarray([[-1_000.0], [1_000.0]], dtype=np.float32), scaler
        )
        np.testing.assert_array_equal(transformed[:, 0], np.asarray([-10.0, 10.0]))

    def test_all_task_heads_use_native_width(self) -> None:
        for task in TASKS:
            config = Phase67DownstreamConfig(task, METHOD, walk=1, device="cpu")
            output = build_external_head(config)(torch.randn(3, OUTPUT_DIM))
            expected = (3, 3) if task == "classification_h2" else (3, 1)
            self.assertEqual(tuple(output.shape), expected)


class FrozenConfigTests(unittest.TestCase):
    def test_external_encoder_budget_is_frozen(self) -> None:
        with self.assertRaises(ValueError):
            Phase67ExternalEncoderConfig("saurl_frozen", walk=1, epochs=15)

    def test_downstream_rejects_non_native_width(self) -> None:
        with self.assertRaises(ValueError):
            Phase67DownstreamConfig(
                "classification_h2", METHOD, walk=1, input_dim=445
            )


class ExactResumeTests(unittest.TestCase):
    def test_rng_model_and_optimizer_restore_replay_next_step(self) -> None:
        model_config = SaURLConfig(
            sequence_length=8,
            input_dim=2,
            augmentation_hidden_dim=4,
            encoder_hidden_dim=4,
            representation_dim=8,
            projector_hidden_dim=8,
            projection_dim=8,
            predictor_hidden_dim=8,
            dilations=(1, 2),
            attention_regions=2,
            batch_size=4,
        )
        set_seed(17)
        model = build_saurl(model_config)
        optimizers = encoder_runner._optimizer_bundle(model, model_config)
        generator = torch.Generator().manual_seed(29)
        batch = torch.randn(4, 8, 2)
        encoder_runner._training_step(
            model,
            batch,
            global_step=0,
            optimizers=optimizers,
            model_config=model_config,
        )
        model_state = copy.deepcopy(model.state_dict())
        optimizer_states = {
            name: copy.deepcopy(optimizer.state_dict())
            for name, optimizer in optimizers.items()
        }
        rng_state = encoder_runner._capture_rng_state(generator)

        encoder_runner._training_step(
            model,
            batch,
            global_step=1,
            optimizers=optimizers,
            model_config=model_config,
        )
        expected = copy.deepcopy(model.state_dict())

        replay = build_saurl(model_config)
        replay.load_state_dict(model_state, strict=True)
        replay_optimizers = encoder_runner._optimizer_bundle(replay, model_config)
        for name, optimizer in replay_optimizers.items():
            optimizer.load_state_dict(optimizer_states[name])
        replay_generator = torch.Generator()
        encoder_runner._restore_rng_state(rng_state, replay_generator)
        encoder_runner._training_step(
            replay,
            batch,
            global_step=1,
            optimizers=replay_optimizers,
            model_config=model_config,
        )
        for name, value in replay.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]), name)


class CrossDeviceToleranceTests(unittest.TestCase):
    def test_accepts_small_scale_relative_drift(self) -> None:
        expected = torch.linspace(-1000.0, 1000.0, 1024)
        replayed = expected * (1.0 + 2e-4)
        self.assertTrue(cross_device_tensors_close(replayed, expected))

    def test_rejects_material_relative_drift(self) -> None:
        expected = torch.linspace(-1000.0, 1000.0, 1024)
        replayed = expected * 1.01
        self.assertFalse(cross_device_tensors_close(replayed, expected))


if __name__ == "__main__":
    unittest.main()
