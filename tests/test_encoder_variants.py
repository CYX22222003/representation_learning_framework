from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.contrastive import nt_xent_loss
from models.encoder_variants import (
    TemporalBackboneConfig,
    build_contrastive_variant,
    trainable_parameter_count,
)
from training.train_encoder_variants import train_temporal_contrastive_epoch


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRAINER = _load_script("phase2_encoder_trainer", "train_phase2_contrastive_encoder.py")
EXTRACTOR = _load_script("phase2_encoder_extractor", "extract_phase2_encoder_features.py")


class EncoderVariantModelTests(unittest.TestCase):
    def test_primary_variants_preserve_embedding_contract(self) -> None:
        config = TemporalBackboneConfig(
            hidden_dim=128,
            transformer_num_layers=1,
            transformer_num_heads=4,
            transformer_feedforward_dim=64,
            max_seq_len=16,
        )
        batch = torch.randn(4, 8, 5)
        for variant in ("contrastive_lstm", "contrastive_transformer"):
            with self.subTest(variant=variant):
                model = build_contrastive_variant(variant, input_dim=5, backbone_config=config)
                hidden, projected = model(batch)
                self.assertEqual(hidden.shape, (4, 128))
                self.assertEqual(projected.shape, (4, 128))
                torch.testing.assert_close(projected.norm(dim=-1), torch.ones(4))
                self.assertGreater(trainable_parameter_count(model), 0)

    def test_variant_rejects_unknown_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown contrastive variant"):
            build_contrastive_variant("contrastive_gru", input_dim=5)

    def test_transformer_rejects_sequence_beyond_configured_maximum(self) -> None:
        config = TemporalBackboneConfig(max_seq_len=4)
        model = build_contrastive_variant("contrastive_transformer", input_dim=5, backbone_config=config)
        with self.assertRaisesRegex(ValueError, "exceeds configured maximum"):
            model(torch.randn(2, 5, 5))

    def test_training_epoch_returns_finite_health_diagnostics(self) -> None:
        torch.manual_seed(0)
        config = TemporalBackboneConfig(hidden_dim=16, max_seq_len=8)
        model = build_contrastive_variant(
            "contrastive_lstm", input_dim=5, embedding_dim=12, backbone_config=config
        )
        loader = DataLoader(TensorDataset(torch.randn(8, 8, 5)), batch_size=4, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = train_temporal_contrastive_epoch(model, loader, optimizer)
        self.assertEqual(set(metrics), {"loss", "embedding_std", "embedding_norm", "projected_std"})
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_temporal_variants_use_unchanged_nt_xent_loss(self) -> None:
        model = build_contrastive_variant(
            "contrastive_lstm",
            input_dim=5,
            embedding_dim=12,
            backbone_config=TemporalBackboneConfig(hidden_dim=16),
        )
        _, z1 = model(torch.randn(4, 8, 5))
        _, z2 = model(torch.randn(4, 8, 5))
        self.assertTrue(torch.isfinite(nt_xent_loss(z1, z2, temperature=0.2)))


class EncoderVariantArtifactTests(unittest.TestCase):
    def test_one_epoch_smoke_and_frozen_feature_extraction(self) -> None:
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed = root / "processed.npz"
            np.savez_compressed(
                processed,
                train=rng.normal(size=(8, 8, 5)).astype(np.float32),
                test=rng.normal(size=(3, 8, 5)).astype(np.float32),
            )
            run_root = root / "run"
            checkpoint = root / "checkpoint.pth"
            config = TRAINER.Phase2ContrastiveConfig(
                variant="contrastive_lstm",
                epoch_budgets=(1,),
                batch_size=4,
                hidden_dim=128,
                max_seq_len=8,
                device="cpu",
            )
            snapshots = TRAINER.run_experiment(processed, run_root, checkpoint, config)

            self.assertEqual(len(snapshots), 1)
            self.assertTrue(checkpoint.exists())
            architecture = json.loads((run_root / "architecture_manifest.json").read_text())
            self.assertEqual(architecture["variant"], "contrastive_lstm")
            self.assertEqual(architecture["downstream_embedding_dim"], 128)
            self.assertGreater(architecture["trainable_parameter_count"], 0)

            feature_path = root / "features.npz"
            EXTRACTOR.extract_feature_artifact(
                processed, checkpoint, feature_path, batch_size=4, device_name="cpu"
            )
            with np.load(feature_path) as artifact:
                self.assertEqual(artifact["contrastive_lstm"].shape, (11, 128))
                self.assertTrue(np.isfinite(artifact["contrastive_lstm"]).all())
            manifest = json.loads(Path(f"{feature_path}.manifest.json").read_text())
            self.assertEqual(manifest["branch_name"], "contrastive_lstm")
            self.assertEqual(manifest["train_shape"], [8, 128])
            self.assertEqual(manifest["test_shape"], [3, 128])


if __name__ == "__main__":
    unittest.main()
