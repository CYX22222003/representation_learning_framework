from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ablation.run_ablation import FrameworkConfig, run_experiment
from features.feature_store import FeatureBundle, NpzFeatureStore


class AblationTrainingIntegrationTests(unittest.TestCase):
    def test_selected_branch_is_the_only_model_input_and_manifest_entry(self) -> None:
        rng = np.random.default_rng(7)
        train = rng.normal(size=(6, 3, 5)).astype(np.float32)
        test = rng.normal(size=(5, 3, 5)).astype(np.float32)
        rows = len(train) + len(test)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed.npz"
            features = root / "features.npz"
            run_root = root / "run"
            np.savez(processed, train=train, test=test)
            NpzFeatureStore(str(features)).save(
                FeatureBundle(
                    statistical=rng.normal(size=(rows, 2)).astype(np.float32),
                    transformed=rng.normal(size=(rows, 3)).astype(np.float32),
                    neural_branches={"vae": rng.normal(size=(rows, 4)).astype(np.float32)},
                )
            )
            np.savez(f"{features}.index.npz", train_size=len(train), test_size=len(test))
            config = FrameworkConfig(
                task="price_prediction", epoch_budgets=(1,), batch_size=2,
                head_hidden_dim=8, device="cpu",
            )
            run_experiment(
                processed, features, run_root, config, selected_branches=("vae",)
            )
            manifest = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["selected_branches"], ["vae"])
        self.assertEqual(manifest["branch_dims"], {"vae": 4})


if __name__ == "__main__":
    unittest.main()
