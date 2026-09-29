from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from features.phase6_5_residual_cnn_features import CONFIG_BRANCHES, CONFIG_DIMS
from models.temporal_backbones import SequenceResidualCNNBackbone
from training.phase6_5_residual_cnn import (
    BACKBONE_CONFIG,
    Phase65ResidualCNNConfig,
    build_residual_cnn_model,
    run_residual_cnn_encoder,
    smoke_test_residual_cnn,
)
from training.phase6_5_residual_cnn_downstream import (
    Phase65ResidualCNNDownstreamConfig,
    smoke_test_residual_cnn_downstream,
)


class Phase65ResidualCNNTests(unittest.TestCase):
    def test_cpu_smoke_covers_both_ssl_families_and_frozen_counts(self) -> None:
        result = smoke_test_residual_cnn()
        self.assertTrue(result["valid"])
        self.assertEqual(
            set(result["models"]), {"contrastive_rescnn", "byol_rescnn"}
        )
        self.assertEqual(
            result["models"]["contrastive_rescnn"]["trainable_parameter_count"],
            333_568,
        )
        self.assertEqual(
            result["models"]["byol_rescnn"]["trainable_parameter_count"],
            366_592,
        )
        for row in result["models"].values():
            self.assertEqual(row["output_shape"], [3, 128])
            self.assertEqual(row["residual_blocks"], 3)
            self.assertTrue(row["identity_skip_verified"])
            self.assertGreater(row["embedding_std"], 1e-3)
            self.assertFalse(row["collapse_warning"])
        self.assertTrue(result["models"]["byol_rescnn"]["byol_ema_updated"])

    def test_backbone_preserves_time_and_identity_skip(self) -> None:
        backbone = SequenceResidualCNNBackbone(5)
        values = torch.randn(2, 64, 5)
        self.assertEqual(tuple(backbone.forward_features(values).shape), (2, 128, 64))
        self.assertEqual(tuple(backbone(values).shape), (2, 128))
        block = backbone.blocks[1].eval()
        probe = torch.randn(2, 128, 64)
        with torch.no_grad():
            for convolution in (block.conv1, block.conv2):
                convolution.weight.zero_()
                convolution.bias.zero_()
        self.assertTrue(torch.equal(block(probe), probe))

    def test_frozen_architecture_and_recipe_guards(self) -> None:
        self.assertEqual(BACKBONE_CONFIG["hidden_dim"], 128)
        self.assertEqual(BACKBONE_CONFIG["num_blocks"], 3)
        self.assertEqual(BACKBONE_CONFIG["groups"], 8)
        self.assertEqual(BACKBONE_CONFIG["dropout"], 0.1)
        config = Phase65ResidualCNNConfig("contrastive_rescnn", 1, device="cpu")
        model = build_residual_cnn_model(config)
        self.assertEqual(len(model.backbone.blocks), 3)
        with self.assertRaises(ValueError):
            Phase65ResidualCNNConfig("contrastive_rescnn", 1, epochs=49)
        with self.assertRaises(ValueError):
            Phase65ResidualCNNConfig("byol_rescnn", 1, batch_size=128)
        with self.assertRaises(ValueError):
            Phase65ResidualCNNConfig("contrastive_lstm", 1)

    def test_occupied_run_path_is_refused_before_data_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            occupied = Path(directory)
            with self.assertRaises(FileExistsError):
                run_residual_cnn_encoder(
                    occupied / "missing.npz",
                    occupied,
                    Phase65ResidualCNNConfig("contrastive_rescnn", 1, device="cpu"),
                )

    def test_feature_matrix_matches_frozen_roles(self) -> None:
        self.assertEqual(
            set(CONFIG_BRANCHES), {"H0", "HC-SR", "HB-SR", "HC-AR", "HB-AR"}
        )
        self.assertEqual(CONFIG_DIMS["H0"], 445)
        self.assertEqual(CONFIG_DIMS["HC-SR"], 445)
        self.assertEqual(CONFIG_DIMS["HB-SR"], 445)
        self.assertEqual(CONFIG_DIMS["HC-AR"], 573)
        self.assertEqual(CONFIG_DIMS["HB-AR"], 573)

    def test_price_only_downstream_smoke_and_recipe_guards(self) -> None:
        result = smoke_test_residual_cnn_downstream()
        self.assertTrue(result["valid"])
        self.assertEqual(set(result["heads"]), {"HC-SR", "HC-AR"})
        with self.assertRaises(ValueError):
            Phase65ResidualCNNDownstreamConfig("HC-SR", 1, batch_size=256)
        with self.assertRaises(ValueError):
            Phase65ResidualCNNDownstreamConfig("H0", 1)
        with self.assertRaises(ValueError):
            Phase65ResidualCNNDownstreamConfig("HC-SR", 1, task="classification_h2")


if __name__ == "__main__":
    unittest.main()
