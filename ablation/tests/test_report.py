from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ablation.report import build_report


class AblationReportTests(unittest.TestCase):
    def test_error_delta_is_positive_when_variant_beats_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "tasks": ["price_prediction"],
                "variants": [
                    {"name": "single_vae", "family": "single", "branches": ["vae"], "mode": "concat"},
                    {"name": "full_concat", "family": "control", "branches": ["vae", "byol"], "mode": "concat"},
                ],
            }
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            values = {"single_vae": (0.4, 0.5), "full_concat": (0.5, 0.6)}
            for name, (mae, rmse) in values.items():
                run = root / "runs" / "price_prediction" / name / "seed_0"
                run.mkdir(parents=True)
                (run / "sweep_metrics.json").write_text(
                    json.dumps([{
                        "epoch": 15, "mae": mae, "rmse": rmse,
                        "branch_dims": {"vae": 4}, "model_input_dim": 4,
                        "trainable_parameter_count": 25,
                    }]), encoding="utf-8"
                )
            json_path, markdown_path = build_report(root)
            rows = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(rows[0]["mae_delta_vs_full_concat"], 0.1)
            self.assertEqual(rows[0]["trainable_parameter_count"], 25)
            self.assertTrue(markdown_path.exists())


if __name__ == "__main__":
    unittest.main()
