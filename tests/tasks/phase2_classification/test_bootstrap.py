from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.bootstrap_phase2_classification import build_commands


class BootstrapTests(unittest.TestCase):
    def test_default_owned_matrix_is_complete_and_has_unique_run_roots(self) -> None:
        args = argparse.Namespace(
            processed_npz="processed.npz", features_npz="features.npz",
            labels_npz="labels.npz", ta_features_npz="ta.npz",
            protocols="P1U,P1O,P2", include_p0_reference=True,
            seeds="0,1,2", epoch_budgets="15,50,100", device="cuda", overwrite=False,
        )
        commands = build_commands(args, Path("matrix"))
        training = [command for command in commands if "--run-root" in command]
        roots = [command[command.index("--run-root") + 1] for command in training]
        self.assertEqual(len(training), 36)
        self.assertEqual(len(roots), len(set(roots)))
        self.assertEqual(
            {Path(root).parts[-3] for root in roots},
            {"C1_raw_ohlcv_mlp", "C2_framework", "C5_ta_mlp"},
        )
        self.assertFalse(any("_full" in root or "/A_" in root or "/B_" in root for root in roots))


if __name__ == "__main__":
    unittest.main()
