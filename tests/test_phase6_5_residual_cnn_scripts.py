from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Phase65ResidualCNNScriptTests(unittest.TestCase):
    def test_scripts_exist_and_training_is_explicit(self) -> None:
        training_scripts = (
            "bootstrap_phase6_5_residual_cnn.py",
            "bootstrap_phase6_5_residual_cnn_downstream.py",
        )
        supporting_scripts = (
            "validate_phase6_5_residual_cnn.py",
            "prepare_phase6_5_residual_cnn_features.py",
            "validate_phase6_5_residual_cnn_features.py",
            "validate_phase6_5_residual_cnn_downstream.py",
            "analyze_phase6_5_residual_cnn_cka.py",
            "report_phase6_5_residual_cnn.py",
        )
        for name in (*training_scripts, *supporting_scripts):
            self.assertTrue((ROOT / "scripts_v5" / name).is_file(), name)
        for name in training_scripts:
            source = (ROOT / "scripts_v5" / name).read_text(encoding="utf-8")
            self.assertIn('parser.add_argument("--execute", action="store_true")', source)
            self.assertIn("if args.execute:", source)


if __name__ == "__main__":
    unittest.main()
