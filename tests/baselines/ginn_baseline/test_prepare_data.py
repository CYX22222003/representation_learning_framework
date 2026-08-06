from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tests.baselines.ginn_baseline.helpers import synthetic_contract


class PrepareDataCliTests(unittest.TestCase):
    def test_prepare_cli_writes_and_validates_cache_from_resolved_top_k_sources(self):
        from baselines.ginn_baseline.prepare_data import main

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_dir = root / "data"
            data_dir.mkdir()
            cache_path = root / "cache.npz"
            manifest_path = root / "manifest.json"
            calls = []

            def fake_read_feather(path):
                calls.append(Path(path).name)
                return synthetic_contract(140 + len(calls), phase=float(len(calls)))

            with patch(
                "baselines.ginn_baseline.prepare_data.list_top_k",
                return_value=[("a-4h.feather", 10), ("b-4h.feather", 9)],
            ), patch.object(pd, "read_feather", side_effect=fake_read_feather):
                rc = main(
                    [
                        "--timeframe",
                        "4h",
                        "--top-k",
                        "2",
                        "--data-dir",
                        str(data_dir),
                        "--out-path",
                        str(cache_path),
                        "--manifest-path",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(calls, ["a-4h.feather", "b-4h.feather"])
            self.assertTrue(cache_path.exists())
            self.assertTrue(manifest_path.exists())

    def test_prepare_cli_refuses_existing_outputs_without_overwrite(self):
        from baselines.ginn_baseline.prepare_data import main

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            cache_path = root / "cache.npz"
            manifest_path = root / "manifest.json"
            cache_path.write_bytes(b"existing")
            manifest_path.write_text("{}", encoding="utf-8")
            rc = main(["--out-path", str(cache_path), "--manifest-path", str(manifest_path)])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
