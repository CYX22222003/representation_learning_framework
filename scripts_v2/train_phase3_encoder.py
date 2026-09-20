"""Train one frozen Phase 3 encoder configuration on training sequences only."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.byol import BYOLEncoder, byol_loss  # noqa: E402
from models.contrastive import ContrastiveEncoder, make_views, nt_xent_loss  # noqa: E402
from models.encoder_variants import (  # noqa: E402
    TemporalBackboneConfig,
    build_byol_variant,
    build_contrastive_variant,
    trainable_parameter_count,
)
from models.vae import SequenceVAE, vae_loss  # noqa: E402
from phase3_encoder_common import (  # noqa: E402
    DEFAULT_ENCODER_ROOT,
    DEFAULT_PROCESSED_NPZ,
    ENCODER_VARIANTS,
    EPOCH_BUDGETS,
    PHASE3_EXPERIMENT_ROOT,
    SEED,
    environment_manifest,
    read_json,
    refuse_occupied,
    require_phase3_path,
    resolve_device,
    set_seed,
    sha256_file,
    write_json,
)
from validate_phase3_encoder_data import validate_bundle  # noqa: E402


@dataclass(frozen=True)
class EncoderConfig:
    variant: str
    epoch_budgets: tuple[int, ...] = EPOCH_BUDGETS
    seed: int = SEED
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    embedding_dim: int = 128
    vae_hidden_dim: int = 256
    vae_latent_dim: int = 64
    vae_beta: float = 1.0
    temperature: float = 0.2
    target_decay: float = 0.99
    collapse_std_threshold: float = 1e-3
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.variant not in ENCODER_VARIANTS:
            raise ValueError(f"unknown Phase 3 encoder variant: {self.variant}")
        if self.epoch_budgets != EPOCH_BUDGETS or self.seed != SEED:
            raise ValueError("initial Phase 3 budgets and seed are frozen to 5/15/50 and seed 0")
        if self.batch_size <= 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["epoch_budgets"] = list(self.epoch_budgets)
        return payload


def _family_backbone(variant: str) -> tuple[str, str]:
    family, backbone = variant.split("_", 1)
    return family, backbone


def default_run_root(variant: str) -> Path:
    family, backbone = _family_backbone(variant)
    return DEFAULT_ENCODER_ROOT / family / backbone / "seed0"


def _load_train(path: Path) -> np.ndarray:
    validate_bundle(path)
    with np.load(path, allow_pickle=False) as bundle:
        train = np.asarray(bundle["train"], dtype=np.float32)
    if train.ndim != 3 or not np.isfinite(train).all():
        raise ValueError("training sequences must be finite [N,T,F]")
    return train


def _loader(train: np.ndarray, config: EncoderConfig) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    result = DataLoader(
        TensorDataset(torch.from_numpy(train)), batch_size=config.batch_size,
        shuffle=True, drop_last=True, generator=generator,
    )
    if len(result) == 0:
        raise ValueError("training split is too small for the fixed batch size")
    return result


def _backbone_config() -> TemporalBackboneConfig:
    return TemporalBackboneConfig(
        hidden_dim=128,
        lstm_num_layers=1,
        lstm_dropout=0.0,
        transformer_num_layers=2,
        transformer_num_heads=4,
        transformer_feedforward_dim=256,
        transformer_dropout=0.1,
        transformer_norm_first=False,
        max_seq_len=512,
    )


def _build_model(config: EncoderConfig, seq_len: int, input_dim: int) -> torch.nn.Module:
    if config.variant == "vae_mlp":
        return SequenceVAE(seq_len, input_dim, config.vae_latent_dim, config.vae_hidden_dim)
    if config.variant == "contrastive_cnn":
        return ContrastiveEncoder(input_dim, config.hidden_dim, config.embedding_dim)
    if config.variant.startswith("contrastive_"):
        return build_contrastive_variant(
            config.variant, input_dim, config.embedding_dim, _backbone_config()
        )
    if config.variant == "byol_cnn":
        return BYOLEncoder(input_dim, config.hidden_dim, config.embedding_dim, config.hidden_dim)
    return build_byol_variant(
        config.variant, input_dim, config.embedding_dim, config.hidden_dim, _backbone_config()
    )


def _check_gradients(model: torch.nn.Module) -> None:
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise FloatingPointError("non-finite encoder gradient")


def _train_vae_epoch(model, loader, optimizer, device, config) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "recon": 0.0, "kld": 0.0}
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstruction, mu, logvar, _ = model(batch)
        loss, recon, kld = vae_loss(reconstruction, batch, mu, logvar, beta=config.vae_beta)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite VAE loss")
        loss.backward()
        _check_gradients(model)
        optimizer.step()
        for name, value in (("loss", loss), ("recon", recon), ("kld", kld)):
            totals[name] += float(value.item())
    return {name: value / len(loader) for name, value in totals.items()}


def _train_contrastive_epoch(model, loader, optimizer, device, config) -> dict[str, float]:
    model.train()
    total = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        view1, view2 = make_views(batch)
        optimizer.zero_grad(set_to_none=True)
        _, projected1 = model(view1)
        _, projected2 = model(view2)
        loss = nt_xent_loss(projected1, projected2, temperature=config.temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite NT-Xent loss")
        loss.backward()
        _check_gradients(model)
        optimizer.step()
        total += float(loss.item())
    return {"loss": total / len(loader)}


def _train_byol_epoch(model, loader, optimizer, device, config) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "view_cosine": 0.0}
    for (batch,) in loader:
        batch = batch.to(device)
        view1, view2 = make_views(batch)
        optimizer.zero_grad(set_to_none=True)
        _, _, prediction1, prediction2, target1, target2 = model(view1, view2)
        loss = byol_loss(prediction1, target2, prediction2, target1)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite BYOL loss")
        loss.backward()
        _check_gradients(model)
        optimizer.step()
        model.update_target(tau=config.target_decay)
        with torch.no_grad():
            cosine = 0.5 * (
                F.cosine_similarity(F.normalize(prediction1, dim=-1), target2, dim=-1).mean()
                + F.cosine_similarity(F.normalize(prediction2, dim=-1), target1, dim=-1).mean()
            )
        totals["loss"] += float(loss.item())
        totals["view_cosine"] += float(cosine.item())
    return {name: value / len(loader) for name, value in totals.items()}


@torch.no_grad()
def _representation_health(model, variant: str, train: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    sample = torch.from_numpy(train[: min(len(train), 4096)]).to(device)
    chunks = []
    for batch in sample.split(512):
        if variant == "vae_mlp":
            embedding, _ = model.encode(batch)
        elif variant.startswith("contrastive_"):
            embedding = model(batch)[0]
        else:
            embedding = model.encode(batch)
        chunks.append(embedding.detach().cpu())
    values = torch.cat(chunks, dim=0)
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite frozen representation diagnostic")
    return {
        "embedding_std": float(values.std(dim=0, unbiased=False).mean().item()),
        "embedding_norm": float(values.norm(dim=-1).mean().item()),
        "embedding_abs_mean": float(values.abs().mean().item()),
    }


def _history_payload(history: list[dict[str, float]], epoch_seconds: list[float]) -> dict[str, np.ndarray]:
    names = sorted({name for row in history for name in row})
    payload = {
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        "epoch_seconds": np.asarray(epoch_seconds, dtype=np.float64),
    }
    for name in names:
        payload[name] = np.asarray([row.get(name, np.nan) for row in history], dtype=np.float64)
    return payload


def run_experiment(processed_npz: Path, run_root: Path, config: EncoderConfig) -> list[dict]:
    require_phase3_path(run_root, PHASE3_EXPERIMENT_ROOT / "encoder_pretraining")
    refuse_occupied((run_root,))
    train = _load_train(processed_npz)
    manifest = read_json(Path(f"{processed_npz}.manifest.json"))
    device = resolve_device(config.device)
    set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = _build_model(config, int(train.shape[1]), int(train.shape[2])).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = _loader(train, config)
    parameter_count = int(trainable_parameter_count(model))
    dataset_sha = sha256_file(processed_npz)

    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "config.json", config.to_dict())
    write_json(run_root / "environment.json", environment_manifest(device))
    write_json(run_root / "dataset_manifest.json", {
        "processed_npz": str(processed_npz.resolve()),
        "processed_npz_sha256": dataset_sha,
        "source_manifest_sha256": sha256_file(Path(f"{processed_npz}.manifest.json")),
        "train_shape": list(train.shape),
        "train_identity_hash": manifest["identity_hashes"]["train"],
        "test_identity_hash": manifest["identity_hashes"]["test"],
        "test_usage": "provenance validation only; values are excluded from optimization and diagnostics",
    })
    write_json(run_root / "architecture_manifest.json", {
        "variant": config.variant,
        "family": _family_backbone(config.variant)[0],
        "backbone": _family_backbone(config.variant)[1],
        "trainable_parameter_count": parameter_count,
        "downstream_embedding_dim": 64 if config.variant == "vae_mlp" else 128,
        "temporal_backbone_config": _backbone_config().to_dict() if config.variant.endswith(("lstm", "transformer")) else None,
    })

    history: list[dict[str, float]] = []
    epoch_seconds: list[float] = []
    snapshots: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, max(config.epoch_budgets) + 1):
        epoch_started = time.perf_counter()
        if config.variant == "vae_mlp":
            diagnostics = _train_vae_epoch(model, loader, optimizer, device, config)
        elif config.variant.startswith("contrastive_"):
            diagnostics = _train_contrastive_epoch(model, loader, optimizer, device, config)
        else:
            diagnostics = _train_byol_epoch(model, loader, optimizer, device, config)
        health = _representation_health(model, config.variant, train, device)
        diagnostics.update(health)
        if not all(math.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError("non-finite training diagnostic")
        diagnostics["collapse_warning"] = float(health["embedding_std"] < config.collapse_std_threshold)
        history.append(diagnostics)
        epoch_seconds.append(time.perf_counter() - epoch_started)
        print(
            f"{config.variant} epoch={epoch} loss={diagnostics['loss']:.8f} "
            f"embedding_std={health['embedding_std']:.8f} seconds={epoch_seconds[-1]:.2f}",
            flush=True,
        )
        if epoch not in config.epoch_budgets:
            continue
        budget_dir = run_root / f"e{epoch}"
        budget_dir.mkdir()
        checkpoint_path = budget_dir / "checkpoint.pth"
        checkpoint = {
            "phase": 3,
            "purpose": "encoder_pretraining",
            "variant": config.variant,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_epoch": epoch,
            "seq_len": int(train.shape[1]),
            "input_dim": int(train.shape[2]),
            "training_config": config.to_dict(),
            "processed_npz": str(processed_npz.resolve()),
            "processed_npz_sha256": dataset_sha,
            "train_identity_hash": manifest["identity_hashes"]["train"],
            "trainable_parameter_count": parameter_count,
            "backbone_config": _backbone_config().to_dict() if config.variant.endswith(("lstm", "transformer")) else None,
        }
        torch.save(checkpoint, checkpoint_path)
        np.savez_compressed(budget_dir / "history.npz", **_history_payload(history, epoch_seconds))
        row = {
            "variant": config.variant,
            "epoch": epoch,
            "seed": config.seed,
            "train_loss": diagnostics["loss"],
            "best_train_loss": min(item["loss"] for item in history),
            **health,
            "collapse_warning": bool(diagnostics["collapse_warning"]),
            "trainable_parameter_count": parameter_count,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        for name in ("recon", "kld", "view_cosine"):
            if name in diagnostics:
                row[name] = diagnostics[name]
        write_json(budget_dir / "metrics.json", row)
        snapshots.append(row)

    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    write_json(run_root / "training_complete.json", {
        "complete": True,
        "variant": config.variant,
        "budgets": list(config.epoch_budgets),
        "principal_downstream_checkpoint": str((run_root / "e50" / "checkpoint.pth").resolve()),
        "selection_rule": "epoch 50 was predeclared; no test metric was used",
    })
    return snapshots


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=ENCODER_VARIANTS)
    parser.add_argument("--processed-npz", type=Path, default=DEFAULT_PROCESSED_NPZ)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    try:
        config = EncoderConfig(variant=args.variant, device=args.device)
        run_root = args.run_root or default_run_root(args.variant)
        run_experiment(args.processed_npz, run_root, config)
        print(f"Phase 3 encoder pretraining complete: {run_root}")
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Phase 3 encoder pretraining failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
