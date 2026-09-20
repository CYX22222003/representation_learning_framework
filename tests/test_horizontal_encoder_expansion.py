from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from features.feature_store import FeatureBundle, NpzFeatureStore


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUNDLE = _load_script("horizontal_bundle", "build_phase2_horizontal_feature_bundle.py")
CKA = _load_script("horizontal_cka", "compute_horizontal_linear_cka.py")
FRAMEWORK = _load_script("horizontal_framework", "train_framework.py")
BOOTSTRAP = _load_script("horizontal_bootstrap", "bootstrap_phase2_horizontal_encoders.py")


def _save_bundle(path: Path, bundle: FeatureBundle, train_size: int, test_size: int) -> None:
    NpzFeatureStore(str(path)).save(bundle)
    np.savez_compressed(f"{path}.index.npz", train_size=train_size, test_size=test_size)


class HorizontalFeatureTests(unittest.TestCase):
    def test_superset_aliases_and_linear_cka_duplicate_control(self) -> None:
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical = root / "canonical.npz"
            rows, train_size = 7, 5
            _save_bundle(
                canonical,
                FeatureBundle(
                    statistical=rng.normal(size=(rows, 70)),
                    transformed=rng.normal(size=(rows, 55)),
                    neural_branches={
                        "vae": rng.normal(size=(rows, 64)),
                        "contrastive": rng.normal(size=(rows, 128)),
                        "byol": rng.normal(size=(rows, 128)),
                    },
                ),
                train_size,
                rows - train_size,
            )
            candidate = root / "candidate.npz"
            candidate_values = rng.normal(size=(rows, 128)).astype(np.float32)
            np.savez_compressed(candidate, contrastive_lstm=candidate_values)
            np.savez_compressed(
                f"{candidate}.index.npz", train_size=train_size, test_size=rows - train_size
            )
            output = root / "superset" / "seed0.npz"
            BUNDLE.build_horizontal_bundle(canonical, [candidate], output)

            train, test, index = FRAMEWORK.load_split_feature_branches(
                output,
                "statistical,contrastive,contrastive_lstm",
                "contrastive_dup1=contrastive",
            )
            self.assertEqual(index, {"train_size": train_size, "test_size": 2})
            self.assertEqual(list(train), [
                "statistical", "contrastive", "contrastive_lstm", "contrastive_dup1"
            ])
            np.testing.assert_array_equal(train["contrastive"], train["contrastive_dup1"])
            np.testing.assert_array_equal(test["contrastive"], test["contrastive_dup1"])

            names, matrix = CKA.compute_bundle_cka(
                output,
                ["contrastive", "contrastive_lstm"],
                [("contrastive_dup1", "contrastive")],
            )
            self.assertEqual(names, ["contrastive", "contrastive_lstm", "contrastive_dup1"])
            self.assertAlmostEqual(matrix[0, 2], 1.0, places=12)
            np.testing.assert_allclose(matrix, matrix.T)


class HorizontalMatrixTests(unittest.TestCase):
    def test_complete_predeclared_matrix_and_command_counts(self) -> None:
        self.assertEqual(len(BOOTSTRAP.CONFIGS), 15)
        for required in ("HC-ALT", "HC-DD", "HB-ALT", "HB-DD"):
            self.assertIn(required, BOOTSTRAP.CONFIGS)
        args = argparse.Namespace(
            seed_values=[0], epoch_budgets="15,50,100", device="cuda"
        )
        commands = BOOTSTRAP.build_stage_commands(args, Path("horizontal-root"))
        self.assertEqual(
            {name: len(values) for name, values in commands.items()},
            {"pretrain": 2, "features": 4, "bundles": 2, "cka": 2, "probes": 45},
        )
        pretrain_text = "\n".join(" ".join(command) for command in commands["pretrain"])
        self.assertNotIn("contrastive_lstm", pretrain_text)
        self.assertNotIn("contrastive_transformer", pretrain_text)
        self.assertIn("byol_lstm", pretrain_text)
        self.assertIn("byol_transformer", pretrain_text)
        probe_text = "\n".join(" ".join(command) for command in commands["probes"])
        self.assertIn("contrastive_dup1=contrastive,contrastive_dup2=contrastive", probe_text)
        self.assertIn("byol_dup1=byol,byol_dup2=byol", probe_text)


if __name__ == "__main__":
    unittest.main()
