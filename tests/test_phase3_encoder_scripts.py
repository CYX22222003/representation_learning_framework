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
from features.statistical import batch_statistical_features
from features.transform import batch_transform_features
from prepare_phase3_price_labels import _split_labels
from phase3_price_common import PRICE_CONFIGS, source_branch
from extract_phase3_features import _validate_checkpoint_provenance
from phase3_encoder_common import sha256_file
from report_phase3_price import _replay_metrics
from tasks.price_prediction import PriceRegressor


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


class Phase3PricePipelineTests(unittest.TestCase):
    def test_feature_checkpoint_provenance_rejects_changed_source(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            records = {}
            for variant in ENCODER_VARIANTS:
                checkpoint = root / f"{variant}.pth"
                checkpoint.write_bytes(variant.encode())
                records[variant] = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
            _validate_checkpoint_provenance({"checkpoints": records})
            changed = Path(records[ENCODER_VARIANTS[0]]["path"])
            changed.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint source hash mismatch"):
                _validate_checkpoint_provenance({"checkpoints": records})

    def test_price_metric_replay_computes_all_published_metrics(self) -> None:
        metrics = _replay_metrics(
            np.asarray([-0.1, 0.4, 1.2], dtype=np.float32),
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        )
        self.assertEqual(set(metrics), {
            "mae", "rmse", "mse", "corr",
            "prediction_below_zero_fraction", "prediction_above_one_fraction",
        })
        self.assertAlmostEqual(metrics["prediction_below_zero_fraction"], 1 / 3)
        self.assertAlmostEqual(metrics["prediction_above_one_fraction"], 1 / 3)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _replay_metrics(np.asarray([np.nan]), np.asarray([0.0]))

    def test_deterministic_features_are_window_local(self) -> None:
        rng = np.random.default_rng(7)
        sequences = rng.normal(size=(2, 64, 5)).astype(np.float32)
        changed = sequences.copy()
        changed[1] = 1_000_000.0
        np.testing.assert_allclose(
            batch_statistical_features(sequences[:1]),
            batch_statistical_features(changed[:1]),
        )
        np.testing.assert_array_equal(
            batch_transform_features(sequences[:1]),
            batch_transform_features(changed[:1]),
        )

    def test_price_labels_drop_each_contract_terminal_row(self) -> None:
        sequences = np.zeros((5, 4, 5), dtype=np.float32)
        sequences[:, -1, 3] = np.asarray([0.1, 0.2, 0.9, 0.8, 0.7], dtype=np.float32)
        bundle = {
            "train": sequences,
            "train_contract_ids": np.asarray([0, 0, 1, 1, 1], dtype=np.int32),
            "train_window_starts": np.asarray([0, 1, 10, 11, 12], dtype=np.int64),
            "train_timestamps_ns": np.arange(5, dtype=np.int64),
        }
        labels = _split_labels(bundle, "train")
        np.testing.assert_array_equal(labels["train_row_indices"], [0, 2, 3])
        np.testing.assert_allclose(labels["train_labels"], [0.2, 0.8, 0.7])
        np.testing.assert_array_equal(labels["train_contract_ids"], [0, 1, 1])

    def test_price_matrix_has_fixed_width_controls(self) -> None:
        dims = {
            "statistical": 70, "transformed": 55, "vae_mlp": 64,
            "contrastive_cnn": 128, "contrastive_lstm": 128,
            "contrastive_transformer": 128, "byol_cnn": 128,
            "byol_lstm": 128, "byol_transformer": 128,
        }
        widths = {
            cid: sum(dims[source_branch(name)] for name in branches)
            for cid, branches in PRICE_CONFIGS.items()
        }
        self.assertEqual(widths["H0"], 445)
        self.assertEqual(widths["HC-SL"], 445)
        self.assertEqual(widths["HB-ST"], 445)
        self.assertEqual(widths["HC-AL"], widths["HC-DC"])
        self.assertEqual(widths["HB-ALT"], widths["HB-DD"])

    def test_price_head_is_shallow_scalar_regressor(self) -> None:
        model = PriceRegressor(445, 128)
        output = model(torch.randn(3, 445))
        self.assertEqual(tuple(output.shape), (3, 1))


if __name__ == "__main__":
    unittest.main()
