from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_v2"))
sys.path.insert(0, str(ROOT / "src"))

from launch_phase3_encoder_pretraining import _commands
from phase3_encoder_common import ENCODER_VARIANTS
from prepare_phase3_encoder_data import _score_candidate, select_universe
from train_phase3_encoder import (
    EncoderConfig,
    _build_model,
    _train_byol_epoch,
    _train_contrastive_epoch,
    _train_vae_epoch,
)


def _frame(rows: int, prefix_volume: float, later_volume: float) -> pd.DataFrame:
    values = np.linspace(0.1, 0.9, rows, dtype=np.float64)
    volume = np.full(rows, later_volume, dtype=np.float64)
    volume[:256] = prefix_volume
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=rows, freq="4h"),
        "open": values,
        "high": values + 0.01,
        "low": values - 0.01,
        "close": values,
        "volume": volume,
    })


class Phase3UniverseSelectionTests(unittest.TestCase):
    def test_later_values_do_not_change_universe_score(self) -> None:
        with TemporaryDirectory() as temp:
            first = Path(temp) / "first-4h.feather"
            second = Path(temp) / "second-4h.feather"
            _frame(400, 2.0, 3.0).to_feather(first)
            _frame(400, 2.0, 3_000_000.0).to_feather(second)
            first_score = _score_candidate(first, 0.8, 64)
            second_score = _score_candidate(second, 0.8, 64)
        self.assertEqual(first_score["positive_volume_count"], second_score["positive_volume_count"])
        self.assertEqual(first_score["log_volume_score"], second_score["log_volume_score"])

    def test_selection_uses_prefix_activity_not_full_file_size(self) -> None:
        with TemporaryDirectory() as temp:
            data_dir = Path(temp)
            _frame(400, 5.0, 0.0).to_feather(data_dir / "active-4h.feather")
            _frame(600, 1.0, 1_000_000.0).to_feather(data_dir / "future-heavy-4h.feather")
            selected, records = select_universe(data_dir, "4h", 1, 0.8, 64)
        self.assertEqual(selected[0][0], "active-4h.feather")
        by_name = {record["filename"]: record for record in records}
        self.assertTrue(by_name["active-4h.feather"]["selected"])
        self.assertFalse(by_name["future-heavy-4h.feather"]["selected"])


class Phase3EncoderMatrixTests(unittest.TestCase):
    def test_all_frozen_variants_build_and_emit_expected_width(self) -> None:
        batch = torch.randn(2, 64, 5)
        for variant in ENCODER_VARIANTS:
            with self.subTest(variant=variant):
                model = _build_model(EncoderConfig(variant=variant, device="cpu"), 64, 5)
                model.eval()
                with torch.no_grad():
                    if variant == "vae_mlp":
                        embedding, _ = model.encode(batch)
                        expected = 64
                    elif variant.startswith("contrastive_"):
                        embedding = model(batch)[0]
                        expected = 128
                    else:
                        embedding = model.encode(batch)
                        expected = 128
                self.assertEqual(tuple(embedding.shape), (2, expected))

    def test_launcher_only_targets_scripts_v2(self) -> None:
        commands = _commands(ROOT / "data/phase3/processed/example.npz", "cuda")
        self.assertEqual(len(commands), len(ENCODER_VARIANTS))
        self.assertTrue(all(Path(command[1]).parent.name == "scripts_v2" for command in commands))
        self.assertTrue(all(command[command.index("--variant") + 1] in ENCODER_VARIANTS for command in commands))

    def test_every_variant_completes_one_training_step_with_finite_loss(self) -> None:
        torch.manual_seed(0)
        loader = DataLoader(TensorDataset(torch.randn(4, 64, 5)), batch_size=2, shuffle=False)
        for variant in ENCODER_VARIANTS:
            with self.subTest(variant=variant):
                config = EncoderConfig(variant=variant, batch_size=2, device="cpu")
                model = _build_model(config, 64, 5)
                optimizer = torch.optim.AdamW(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                if variant == "vae_mlp":
                    result = _train_vae_epoch(model, loader, optimizer, torch.device("cpu"), config)
                elif variant.startswith("contrastive_"):
                    result = _train_contrastive_epoch(
                        model, loader, optimizer, torch.device("cpu"), config
                    )
                else:
                    result = _train_byol_epoch(model, loader, optimizer, torch.device("cpu"), config)
                self.assertTrue(np.isfinite(result["loss"]))


if __name__ == "__main__":
    unittest.main()
