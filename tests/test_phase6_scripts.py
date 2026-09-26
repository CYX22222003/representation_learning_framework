from __future__ import annotations

import ast
import unittest
from pathlib import Path

from data_processing.phase6_volatility import validate_horizon_freeze_manifest


ROOT = Path(__file__).resolve().parents[1]


class Phase6ScriptContractTests(unittest.TestCase):
    def test_phase6_scripts_are_present_under_scripts_v4(self) -> None:
        self.assertTrue((ROOT / "scripts_v4" / "audit_phase6_volatility.py").is_file())
        self.assertTrue((ROOT / "scripts_v4" / "validate_phase6_volatility_audit.py").is_file())
        self.assertTrue(
            (ROOT / "scripts_v4" / "validate_phase6_volatility_horizon_freeze.py").is_file()
        )
        self.assertTrue((ROOT / "scripts_v4" / "prepare_phase6_volatility_labels.py").is_file())
        self.assertTrue((ROOT / "scripts_v4" / "validate_phase6_volatility_labels.py").is_file())
        self.assertTrue((ROOT / "scripts_v4" / "prepare_phase6_volatility_features.py").is_file())
        self.assertTrue((ROOT / "scripts_v4" / "validate_phase6_volatility_features.py").is_file())
        self.assertTrue((ROOT / "scripts_v4" / "bootstrap_phase6_volatility.py").is_file())
        self.assertTrue((ROOT / "scripts_v4" / "validate_phase6_volatility_runs.py").is_file())
        encoder_variant_scripts = (
            "bootstrap_phase6_encoder_variants.py",
            "validate_phase6_encoder_variants.py",
            "prepare_phase6_encoder_variant_features.py",
            "validate_phase6_encoder_variant_features.py",
            "analyze_phase6_encoder_variant_cka.py",
            "bootstrap_phase6_encoder_variant_downstream.py",
            "validate_phase6_encoder_variant_downstream.py",
            "report_phase6_encoder_variants.py",
        )
        for name in encoder_variant_scripts:
            self.assertTrue((ROOT / "scripts_v4" / name).is_file(), name)

    def test_encoder_bootstraps_are_manifest_only_without_execute(self) -> None:
        for name in (
            "bootstrap_phase6_encoder_variants.py",
            "bootstrap_phase6_encoder_variant_downstream.py",
        ):
            source = (ROOT / "scripts_v4" / name).read_text(encoding="utf-8")
            self.assertIn('parser.add_argument("--execute", action="store_true")', source)
            self.assertIn("if args.execute:", source)

    def test_audit_entry_point_does_not_import_training_or_models(self) -> None:
        path = ROOT / "scripts_v4" / "audit_phase6_volatility.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertFalse(any(name.startswith("training") for name in imports))
        self.assertFalse(any(name.startswith("models") for name in imports))

    def test_default_horizons_are_predeclared_in_reusable_module(self) -> None:
        source = (ROOT / "src" / "data_processing" / "phase6_volatility.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("DEFAULT_CANDIDATE_HORIZONS = (2, 4, 8, 24)", source)
        self.assertIn('"primary_horizon_frozen": False', source)

    def test_h8_freeze_replays_without_authorizing_training(self) -> None:
        freeze = (
            ROOT
            / "experiments"
            / "phase6"
            / "volatility_prediction"
            / "manifests"
            / "horizon_freeze_h8.json"
        )
        result = validate_horizon_freeze_manifest(freeze, ROOT)
        self.assertEqual(result["primary_horizon_hours"], 8)
        self.assertTrue(result["label_bundle_authorized"])
        self.assertFalse(result["model_training_authorized"])


if __name__ == "__main__":
    unittest.main()
