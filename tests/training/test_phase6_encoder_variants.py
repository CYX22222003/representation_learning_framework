from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.phase6_encoder_variant_reporting import (
    COMPARISON_PAIRS,
    normalize_metrics,
    paired_differences,
)
from evaluation.phase6_encoder_variants import centered_linear_cka
from features.phase6_encoder_variant_features import (
    CONFIG_BRANCHES,
    CONFIG_DIMS,
    SCHEMA_VERSION,
    validate_variant_feature_store,
)
from training.phase6_encoder_variant_downstream import (
    Phase6VariantDownstreamConfig,
    fit_standardizer,
    project_paths_equal,
    run_variant_downstream,
    smoke_test_variant_downstream,
)
from training.phase6_encoder_variants import (
    Phase6EncoderVariantConfig,
    run_encoder_variant,
    smoke_test_encoder_variants,
)


class Phase6EncoderVariantInfrastructureTests(unittest.TestCase):
    def test_encoder_and_head_cpu_smoke(self) -> None:
        encoder = smoke_test_encoder_variants()
        downstream = smoke_test_variant_downstream()
        self.assertTrue(encoder["valid"])
        self.assertEqual(len(encoder["models"]), 4)
        self.assertTrue(downstream["valid"])
        self.assertEqual(len(downstream["heads"]), 6)

    def test_frozen_configs_reject_recipe_drift(self) -> None:
        with self.assertRaises(ValueError):
            Phase6EncoderVariantConfig("contrastive_lstm", 1, epochs=49)
        with self.assertRaises(ValueError):
            Phase6VariantDownstreamConfig("classification_h2", "HC-SL", 1, batch_size=256)

    def test_all_eleven_branch_contracts_have_declared_widths(self) -> None:
        self.assertEqual(len(CONFIG_BRANCHES), 11)
        self.assertEqual(CONFIG_DIMS["H0"], 445)
        for name in ("HC-SL", "HC-ST", "HB-SL", "HB-ST"):
            self.assertEqual(len(CONFIG_BRANCHES[name]), 5)
            self.assertEqual(CONFIG_DIMS[name], 445)
        for name in ("HC-AL", "HC-AT", "HB-AL", "HB-AT", "HC-DC", "HB-DC"):
            self.assertEqual(len(CONFIG_BRANCHES[name]), 6)
            self.assertEqual(CONFIG_DIMS[name], 573)

    def test_occupied_paths_are_refused_before_any_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            occupied = Path(directory)
            with self.assertRaises(FileExistsError):
                run_encoder_variant(
                    occupied / "missing.npz",
                    occupied,
                    Phase6EncoderVariantConfig("contrastive_lstm", 1, device="cpu"),
                )
            with self.assertRaises(ValueError):
                run_variant_downstream(
                    occupied / "missing.npz",
                    occupied / "missing-features.npz",
                    occupied,
                    Phase6VariantDownstreamConfig(
                        "classification_h2", "H0", 1, device="cpu"
                    ),
                )

    def test_standardizer_is_fit_from_training_values_only(self) -> None:
        train = np.arange(4 * 445, dtype=np.float64).reshape(4, 445)
        first = fit_standardizer(train)
        _ = np.full((100, 445), 1e9, dtype=np.float64)
        second = fit_standardizer(train)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_project_path_comparison_is_case_insensitive(self) -> None:
        self.assertTrue(
            project_paths_equal(
                Path("/mnt/e/school-work/project/data.npz"),
                Path("/mnt/e/School-Work/PROJECT/data.npz"),
            )
        )
        self.assertFalse(
            project_paths_equal(
                Path("/mnt/e/school-work/project/data.npz"),
                Path("/mnt/e/school-work/project/other.npz"),
            )
        )

    def test_centered_linear_cka_identity(self) -> None:
        values = np.random.default_rng(7).normal(size=(32, 8))
        self.assertAlmostEqual(centered_linear_cka(values, values), 1.0, places=12)

    def test_feature_provenance_hash_failure_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature = Path(directory) / "features.npz"
            np.savez_compressed(feature, placeholder=np.asarray([1]))
            Path(f"{feature}.manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "configuration_branches": {
                            key: list(value) for key, value in CONFIG_BRANCHES.items()
                        },
                        "configuration_dims": CONFIG_DIMS,
                        "feature_store_sha256": "intentionally-wrong",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_variant_feature_store(feature)

    def test_reporting_normalizes_schemas_and_keeps_predeclared_pairs(self) -> None:
        payload = {
            "metrics": {
                "framework": {"macro_f1": 0.4, "balanced_accuracy": 0.5}
            }
        }
        self.assertEqual(
            normalize_metrics("classification_h2", payload),
            {"macro_f1": 0.4, "balanced_accuracy": 0.5},
        )
        rows = []
        for index, configuration in enumerate(CONFIG_BRANCHES):
            rows.append(
                {
                    "task": "classification_h2",
                    "walk": 1,
                    "configuration": configuration,
                    "macro_f1": index / 100.0,
                    "balanced_accuracy": index / 50.0,
                }
            )
        differences = paired_differences(rows, "classification_h2", 1)
        self.assertEqual(len(differences), len(COMPARISON_PAIRS) * 2)
        self.assertFalse(any(row["candidate"] == "HC-DC" for row in differences))


if __name__ == "__main__":
    unittest.main()
