from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.bootstrap_phase2_decoders import build_commands, parser


class BootstrapTests(unittest.TestCase):
    def test_default_stage1_matrix_has_thirty_runs(self) -> None:
        args=parser().parse_args([])
        with tempfile.TemporaryDirectory() as tmp:
            commands=build_commands(args,Path(tmp))
        self.assertEqual(len(commands),32)
        self.assertEqual(sum("--run-root" in command for command in commands),30)

    def test_decoder_task_layout_matches_requested_storage_contract(self) -> None:
        args=parser().parse_args(["--layout","decoder-task"])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); commands=build_commands(args,root)
        run=next(command for command in commands if "--run-root" in command)
        run_root=Path(run[run.index("--run-root")+1])
        self.assertEqual(run_root.relative_to(root).parts[:3],("D0","price_prediction","seed0"))


if __name__ == "__main__": unittest.main()
