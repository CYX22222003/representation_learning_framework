from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_v3"))

from prepare_phase5_data import DEFAULT_OUTPUT_ROOT, canonical_walk_inputs


class Phase5ScriptLayoutTests(unittest.TestCase):
    def test_phase5_entry_points_and_outputs_use_new_roots(self) -> None:
        self.assertEqual(
            Path(DEFAULT_OUTPUT_ROOT).parts[-3:],
            ("experiments", "phase5", "data_preparation"),
        )
        inputs = canonical_walk_inputs()
        self.assertEqual(set(inputs), {1, 2})
        self.assertTrue(all("phase5_walk" in item.input_dir.name for item in inputs.values()))
        self.assertTrue((ROOT / "scripts_v3" / "prepare_phase5_data.py").is_file())
        self.assertTrue((ROOT / "scripts_v3" / "validate_phase5_data.py").is_file())


if __name__ == "__main__":
    unittest.main()
