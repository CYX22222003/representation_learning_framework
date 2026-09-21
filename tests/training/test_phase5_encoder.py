from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_v3"))

from launch_phase5_encoder_pretraining import commands
from training.phase5_encoder import (
    ENCODERS,
    Phase5EncoderConfig,
    build_loader,
    build_model,
    train_epoch,
)


class Phase5EncoderTests(unittest.TestCase):
    def test_six_run_matrix_uses_scripts_v3_and_separate_walk_roots(self) -> None:
        matrix = commands("cuda")
        self.assertEqual(len(matrix), 6)
        self.assertTrue(all(Path(command[1]).parent.name == "scripts_v3" for command in matrix))
        self.assertEqual(
            {(int(command[command.index("--walk") + 1]), command[command.index("--encoder") + 1]) for command in matrix},
            {(walk, encoder) for walk in (1, 2) for encoder in ENCODERS},
        )

    def test_all_canonical_neural_encoders_complete_one_cpu_step(self) -> None:
        torch.manual_seed(0)
        train = np.random.default_rng(0).normal(size=(4, 64, 5)).astype(np.float32)
        for encoder in ENCODERS:
            with self.subTest(encoder=encoder):
                config = Phase5EncoderConfig(
                    encoder=encoder, walk=1, batch_size=2, device="cpu"
                )
                model = build_model(config)
                loader = build_loader(train, config)
                optimizer = torch.optim.AdamW(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                diagnostics = train_epoch(
                    model, loader, optimizer, torch.device("cpu"), config
                )
                self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))

    def test_model_embedding_dimensions_match_canonical_branches(self) -> None:
        batch = torch.randn(3, 64, 5)
        expected = {"vae": 64, "contrastive": 128, "byol": 128}
        for encoder in ENCODERS:
            config = Phase5EncoderConfig(encoder=encoder, walk=1, device="cpu")
            model = build_model(config)
            model.eval()
            with torch.no_grad():
                if encoder == "vae":
                    embedding, _ = model.encode(batch)
                elif encoder == "contrastive":
                    embedding = model(batch)[0]
                else:
                    embedding = model.encode(batch)
            self.assertEqual(tuple(embedding.shape), (3, expected[encoder]))

    def test_fixed_training_contract_rejects_changed_seed_or_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed 0"):
            Phase5EncoderConfig(encoder="vae", walk=1, seed=1)
        with self.assertRaisesRegex(ValueError, "50-epoch"):
            Phase5EncoderConfig(encoder="vae", walk=1, epochs=15)


if __name__ == "__main__":
    unittest.main()
