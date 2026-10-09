"""Bounded training-engine tests; fixtures never use canonical experiment rows."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data_processing.phase5_walks import sha256_file
from tests.baselines.xlstm_mixer.test_training import synthetic_bundle
import training.phase6_9_xlstm_mixer as engine


class LifecycleTests(unittest.TestCase):
    def test_cuda_rng_states_are_restored_as_cpu_byte_tensors(self):
        generator = torch.Generator().manual_seed(3)
        with patch("torch.cuda.is_available", return_value=False):
            state = engine._capture_rng_state(generator)
        state["torch_cuda"] = [torch.zeros(8, dtype=torch.uint8)]
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.set_rng_state_all") as restore:
            engine._restore_rng_state(state, generator)
        self.assertEqual(restore.call_args.args[0][0].device.type, "cpu")

    def test_admission_rejects_dependency_and_runtime_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admission.json"
            payload = {"schema_version": engine.ADMISSION_SCHEMA_VERSION,
                "method": engine.METHOD, "admitted": True, "backend": "vanilla",
                "physical_batch_size": 512, "source_contract_sha256": engine.SOURCE_CONTRACT.sha256,
                "implementation_sha256": engine.implementation_fingerprint(),
                "dependency": {"version": "wrong"}, "environment": {"torch": "wrong"}}
            path.write_text(json.dumps(payload))
            config = engine.Phase69XLSTMMixerConfig(walk=1)
            with patch.object(engine, "xlstm_dependency_manifest", return_value={"version": "1.0.3"}):
                with self.assertRaisesRegex(ValueError, "dependency changed"):
                    engine.validate_runtime_admission(path, Path("unused"), config)
            payload["dependency"] = {"version": "1.0.3"}
            path.write_text(json.dumps(payload))
            with patch.object(engine, "xlstm_dependency_manifest", return_value=payload["dependency"]), \
                 patch.object(engine, "resolve_device", return_value=torch.device("cpu")), \
                 patch.object(engine, "runtime_environment_payload", return_value={"torch": "correct"}):
                with self.assertRaisesRegex(ValueError, "runtime changed"):
                    engine.validate_runtime_admission(path, Path("unused"), config)

    @unittest.skipUnless(torch.cuda.is_available(), "bounded real-vanilla integration needs CUDA")
    def test_real_vanilla_interrupt_resume_and_full_replay(self):
        # Two three-row fixture trajectories (not two full experiments), using
        # the real backend/optimizer, all 50 tiny epochs, and fixed snapshots.
        arrays, manifest = synthetic_bundle()
        original_load = engine.load_xlstm_mixer_data
        original_save = engine._atomic_torch_save
        with tempfile.TemporaryDirectory(prefix="xm-mv8-fixture-") as directory:
            root = Path(directory)
            dataset = root / "walk1.npz"
            np.savez_compressed(dataset, **arrays)
            manifest["dataset_sha256"] = sha256_file(dataset)
            Path(f"{dataset}.manifest.json").write_text(json.dumps(manifest))
            admission = root / "fixture_admission.json"
            admission.write_text("{}")
            config = engine.Phase69XLSTMMixerConfig(walk=1)
            def fixture_load(path, walk):
                return original_load(path, walk, enforce_frozen_counts=False)
            def interrupt_after_epoch3(path, payload):
                original_save(path, payload)
                if path.name == "resume.pth" and payload["completed_epoch"] == 3:
                    raise InterruptedError("fixture interruption")
            with patch.object(engine, "load_xlstm_mixer_data", side_effect=fixture_load), \
                 patch.object(engine, "validate_runtime_admission", return_value={"fixture": True}):
                reference = root / "reference"
                resumed = root / "resumed"
                engine.run_xlstm_mixer_training(dataset, reference, admission, config)
                with patch.object(engine, "_atomic_torch_save", side_effect=interrupt_after_epoch3):
                    with self.assertRaises(InterruptedError):
                        engine.run_xlstm_mixer_training(dataset, resumed, admission, config)
                self.assertFalse((resumed / "training_complete.json").exists())
                # Finish epochs but fail final validation: completion stays absent.
                real_validate = engine.validate_xlstm_mixer_training
                with patch.object(engine, "validate_xlstm_mixer_training", side_effect=ValueError("fixture replay failure")):
                    with self.assertRaisesRegex(ValueError, "fixture replay failure"):
                        engine.run_xlstm_mixer_training(dataset, resumed, admission, config)
                self.assertFalse((resumed / "training_complete.json").exists())
                result = engine.run_xlstm_mixer_training(dataset, resumed, admission, config)
                self.assertTrue(result["valid"])
                self.assertTrue(real_validate(dataset, resumed, admission)["valid"])
                for epoch in (5, 15, 50):
                    a = torch.load(reference / f"e{epoch}" / "checkpoint.pth", weights_only=True)
                    b = torch.load(resumed / f"e{epoch}" / "checkpoint.pth", weights_only=True)
                    for key, value in a["model_state_dict"].items():
                        if torch.is_tensor(value):
                            torch.testing.assert_close(value, b["model_state_dict"][key], rtol=0, atol=0)
                    self.assertEqual([r["train_loss"] for r in a["history"]], [r["train_loss"] for r in b["history"]])
