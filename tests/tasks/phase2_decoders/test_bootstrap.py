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


if __name__ == "__main__": unittest.main()
