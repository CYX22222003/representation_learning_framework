from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Phase65BaselineScriptTests(unittest.TestCase):
    def test_phase65b_and_c_scripts_exist_and_training_is_explicit(self) -> None:
        training_scripts = (
            "bootstrap_phase6_5_garch_lstm.py",
            "bootstrap_phase6_5_ta_mlp.py",
        )
        supporting_scripts = (
            "prepare_phase6_5_ta_mlp_data.py",
            "validate_phase6_5_ta_mlp_data.py",
            "validate_phase6_5_garch_lstm.py",
            "validate_phase6_5_ta_mlp.py",
        )
        for name in (*training_scripts, *supporting_scripts):
            self.assertTrue((ROOT / "scripts_v5" / name).is_file(), name)
        for name in training_scripts:
            source = (ROOT / "scripts_v5" / name).read_text(encoding="utf-8")
            self.assertIn('parser.add_argument("--execute", action="store_true")', source)
            self.assertIn("if args.execute:", source)


if __name__ == "__main__":
    unittest.main()
