from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.phase6_5_lstm_capacity import (
    BACKBONE_CONFIG,
    Phase65LSTMCapacityConfig,
    build_lstm_capacity_model,
    run_lstm_capacity_encoder,
    smoke_test_lstm_capacity,
)
from features.phase6_5_lstm_capacity_features import CONFIG_BRANCHES, CONFIG_DIMS
from training.phase6_5_lstm_capacity_downstream import (
    Phase65LSTMCapacityDownstreamConfig,
    smoke_test_lstm_capacity_downstream,
)


class Phase65LSTMCapacityTests(unittest.TestCase):
    def test_cpu_smoke_covers_both_ssl_families(self) -> None:
        result = smoke_test_lstm_capacity()
        self.assertTrue(result["valid"])
        self.assertEqual(set(result["models"]), {"contrastive_lstm2", "byol_lstm2"})
        for row in result["models"].values():
            self.assertEqual(row["output_shape"], [3, 128])
            self.assertEqual(row["lstm_num_layers"], 2)
            self.assertEqual(row["lstm_dropout"], 0.1)

    def test_frozen_architecture_and_recipe(self) -> None:
        self.assertEqual(BACKBONE_CONFIG.hidden_dim, 128)
        self.assertEqual(BACKBONE_CONFIG.lstm_num_layers, 2)
        self.assertEqual(BACKBONE_CONFIG.lstm_dropout, 0.1)
        config = Phase65LSTMCapacityConfig("contrastive_lstm2", 1, device="cpu")
        model = build_lstm_capacity_model(config)
        self.assertEqual(model.backbone.lstm.num_layers, 2)
        self.assertEqual(model.backbone.lstm.dropout, 0.1)
        with self.assertRaises(ValueError):
            Phase65LSTMCapacityConfig("contrastive_lstm2", 1, epochs=49)
        with self.assertRaises(ValueError):
            Phase65LSTMCapacityConfig("byol_lstm2", 1, batch_size=128)
        with self.assertRaises(ValueError):
            Phase65LSTMCapacityConfig("contrastive_lstm", 1)

    def test_occupied_run_path_is_refused_before_data_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            occupied = Path(directory)
            with self.assertRaises(FileExistsError):
                run_lstm_capacity_encoder(
                    occupied / "missing.npz",
                    occupied,
                    Phase65LSTMCapacityConfig("contrastive_lstm2", 1, device="cpu"),
                )

    def test_feature_matrix_matches_frozen_substitution_and_addition_roles(self) -> None:
        self.assertEqual(set(CONFIG_BRANCHES), {"H0", "HC-SL2", "HB-SL2", "HC-AL2", "HB-AL2"})
        self.assertEqual(CONFIG_DIMS["H0"], 445)
        self.assertEqual(CONFIG_DIMS["HC-SL2"], 445)
        self.assertEqual(CONFIG_DIMS["HB-SL2"], 445)
        self.assertEqual(CONFIG_DIMS["HC-AL2"], 573)
        self.assertEqual(CONFIG_DIMS["HB-AL2"], 573)

    def test_downstream_smoke_and_recipe_guards(self) -> None:
        result = smoke_test_lstm_capacity_downstream()
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["heads"]), 6)
        with self.assertRaises(ValueError):
            Phase65LSTMCapacityDownstreamConfig(
                "classification_h2", "HC-SL2", 1, batch_size=256
            )
        with self.assertRaises(ValueError):
            Phase65LSTMCapacityDownstreamConfig(
                "classification_h2", "H0", 1
            )


if __name__ == "__main__":
    unittest.main()
