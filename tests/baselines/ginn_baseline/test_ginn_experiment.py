from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baselines.ginn_baseline.ginn_data import GinnDataConfig, load_and_validate_cache
from baselines.ginn_baseline.run_experiment import (
    ExperimentConfig,
    main,
    run_loaded_experiment,
    verify_run,
)
from tests.baselines.ginn_baseline.helpers import make_two_epoch_synthetic_run, write_synthetic_cache


class GinnExperimentTests(unittest.TestCase):
    def test_two_epoch_run_writes_ordered_artifacts_and_verifies(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache_path = root / "cache.npz"
            manifest_path = root / "manifest.json"
            write_synthetic_cache(cache_path, manifest_path)
            cache = load_and_validate_cache(cache_path, manifest_path, GinnDataConfig())
            run_root = root / "run"
            config = ExperimentConfig(
                epoch_budgets=(1, 2),
                seed=0,
                batch_size=4,
                learning_rate=1e-3,
                lambda_garch=0.3,
                hidden_size=8,
                num_layers=1,
                dropout=0.0,
                output_transform="softplus",
                device="cpu",
            )
            run_loaded_experiment(cache, run_root, config, cache_path, manifest_path)
            sweep = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual([entry["epoch"] for entry in sweep], [1, 2])
            self.assertEqual(sweep[0]["training_config"]["output_transform"], "softplus")
            self.assertEqual(sweep[0]["model_config"]["output_transform"], "softplus")
            for epoch in (1, 2):
                budget = run_root / f"e{epoch}"
                self.assertTrue((budget / "checkpoint.pth").exists())
                self.assertTrue((budget / "history.npz").exists())
                self.assertTrue((budget / "predictions.npz").exists())
                self.assertTrue((budget / "metrics.json").exists())
                self.assertTrue((budget / "summary.md").exists())
            verify_run(run_root, cache_path, manifest_path, device="cpu")

    def test_verify_run_rejects_prediction_tampering(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_root = make_two_epoch_synthetic_run(root)
            cache_path = root / "cache.npz"
            manifest_path = root / "manifest.json"
            pred_path = run_root / "e1" / "predictions.npz"
            with np.load(pred_path) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["preds"] = arrays["preds"] + 1.0
            np.savez(pred_path, **arrays)
            with self.assertRaisesRegex(ValueError, "prediction mismatch"):
                verify_run(run_root, cache_path, manifest_path, device="cpu")

    def test_main_refuses_manifest_mismatch_before_training(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache_path = root / "cache.npz"
            manifest_path = root / "manifest.json"
            write_synthetic_cache(cache_path, manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["config"]["seq_len"] = 32
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            rc = main(
                [
                    "--cache-path",
                    str(cache_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--run-root",
                    str(root / "run"),
                    "--epoch-budgets",
                    "1",
                    "--hidden-size",
                    "8",
                    "--num-layers",
                    "1",
                    "--dropout",
                    "0.0",
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(rc, 1)
            self.assertFalse((root / "run").exists())

    def test_main_refuses_non_empty_run_without_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache_path = root / "cache.npz"
            manifest_path = root / "manifest.json"
            run_root = root / "run"
            write_synthetic_cache(cache_path, manifest_path)
            run_root.mkdir()
            (run_root / "note.txt").write_text("keep", encoding="utf-8")
            rc = main(
                [
                    "--cache-path",
                    str(cache_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--run-root",
                    str(run_root),
                    "--epoch-budgets",
                    "1",
                    "--hidden-size",
                    "8",
                    "--num-layers",
                    "1",
                    "--dropout",
                    "0.0",
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(rc, 2)
            self.assertEqual((run_root / "note.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
