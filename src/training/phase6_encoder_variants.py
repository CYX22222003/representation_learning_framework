"""Walk-specific Phase 6 temporal SSL encoder training and checkpoint replay."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_processing.phase5_walks import sha256_file, validate_phase5_bundle_files
from models.byol import byol_loss
from models.contrastive import make_views, nt_xent_loss
from models.encoder_variants import (
    BYOL_VARIANTS,
    CONTRASTIVE_VARIANTS,
    TemporalBackboneConfig,
    build_byol_variant,
    build_contrastive_variant,
)
from training.phase5_encoder import (
    environment_manifest,
    resolve_device,
    set_seed,
    write_json,
)


VARIANTS = (*CONTRASTIVE_VARIANTS, *BYOL_VARIANTS)
SNAPSHOT_EPOCHS = (5, 15, 50)


@dataclass(frozen=True)
class Phase6EncoderVariantConfig:
    variant: str
    walk: int
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.2
    target_decay: float = 0.99
    collapse_std_threshold: float = 1e-3
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("Phase 6 temporal encoders require 50 epochs and 5/15/50 snapshots")
        if (
            self.seed != 0
            or self.batch_size != 256
            or self.learning_rate != 1e-3
            or self.weight_decay != 1e-4
            or self.temperature != 0.2
            or self.target_decay != 0.99
        ):
            raise ValueError("Phase 6 temporal SSL recipe is frozen")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot_epochs"] = list(self.snapshot_epochs)
        return payload


def build_variant_model(config: Phase6EncoderVariantConfig) -> torch.nn.Module:
    backbone = TemporalBackboneConfig()
    if config.variant in CONTRASTIVE_VARIANTS:
        return build_contrastive_variant(config.variant, input_dim=5, backbone_config=backbone)
    return build_byol_variant(config.variant, input_dim=5, backbone_config=backbone)


def smoke_test_encoder_variants() -> dict[str, Any]:
    """Exercise forward, loss, backward, and BYOL EMA paths without artifacts."""

    set_seed(0)
    results: dict[str, Any] = {}
    for variant in VARIANTS:
        config = Phase6EncoderVariantConfig(variant=variant, walk=1, device="cpu")
        model = build_variant_model(config)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(4, 64, 5)),
            batch_size=2,
            shuffle=False,
            drop_last=True,
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        loss = _train_epoch(model, loader, optimizer, torch.device("cpu"), config)
        output = model.encode(torch.randn(3, 64, 5))
        if output.shape != (3, 128) or not torch.isfinite(output).all():
            raise RuntimeError(f"temporal encoder smoke failed: {variant}")
        results[variant] = {
            "loss_finite": bool(math.isfinite(loss)),
            "output_shape": list(output.shape),
            "trainable_parameter_count": int(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            ),
        }
    return {"valid": True, "models": results}


def _load_train(dataset_path: Path, walk: int) -> tuple[np.ndarray, dict[str, Any]]:
    result = validate_phase5_bundle_files(dataset_path)
    if int(result["walk"]) != walk:
        raise ValueError("temporal encoder dataset walk mismatch")
    manifest = json.loads(Path(f"{dataset_path}.manifest.json").read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=False) as stored:
        train = np.asarray(stored["encoder_train_sequences"], dtype=np.float32)
    if train.ndim != 3 or train.shape[1:] != (64, 5) or not np.isfinite(train).all():
        raise ValueError("encoder_train_sequences must be finite [N,64,5]")
    return train, manifest


def _loader(train: np.ndarray, config: Phase6EncoderVariantConfig) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train)),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(config.seed),
    )


def _train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Phase6EncoderVariantConfig,
) -> float:
    model.train()
    total = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        view1, view2 = make_views(batch)
        optimizer.zero_grad(set_to_none=True)
        if config.variant in CONTRASTIVE_VARIANTS:
            _, projected1 = model(view1)
            _, projected2 = model(view2)
            loss = nt_xent_loss(projected1, projected2, temperature=config.temperature)
        else:
            _, _, prediction1, prediction2, target1, target2 = model(view1, view2)
            loss = byol_loss(prediction1, target2, prediction2, target1)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite temporal encoder loss")
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("non-finite temporal encoder gradient")
        optimizer.step()
        if config.variant in BYOL_VARIANTS:
            model.update_target(tau=config.target_decay)
        total += float(loss.detach())
    if not len(loader):
        raise ValueError("empty temporal encoder loader")
    return total / len(loader)


@torch.no_grad()
def _health(model: torch.nn.Module, train: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    sample = torch.from_numpy(train[: min(len(train), 4096)]).to(device)
    values = []
    for batch in sample.split(512):
        values.append(model.encode(batch).cpu())
    embeddings = torch.cat(values)
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("non-finite temporal embedding")
    return {
        "embedding_std": float(embeddings.std(dim=0, unbiased=False).mean()),
        "embedding_norm": float(embeddings.norm(dim=-1).mean()),
        "embedding_abs_mean": float(embeddings.abs().mean()),
    }


def _atomic_history(path: Path, history: list[dict[str, float]]) -> None:
    arrays = {
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        **{
            name: np.asarray([row[name] for row in history], dtype=np.float64)
            for name in history[0]
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def run_encoder_variant(
    dataset_path: Path, run_root: Path, config: Phase6EncoderVariantConfig
) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite temporal encoder run: {run_root}")
    train, data_manifest = _load_train(dataset_path, config.walk)
    device = resolve_device(config.device)
    set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_variant_model(config).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = _loader(train, config)
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "config.json", config.to_dict())
    write_json(run_root / "environment.json", environment_manifest(device))
    write_json(
        run_root / "dataset_manifest.json",
        {
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_manifest_sha256": sha256_file(Path(f"{dataset_path}.manifest.json")),
            "encoder_train_identity_hash": data_manifest["identity_hashes"]["encoder_train"],
            "encoder_train_rows": len(train),
            "downstream_targets_loaded": False,
            "evaluation_values_loaded": False,
        },
    )
    parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    write_json(
        run_root / "architecture_manifest.json",
        {
            "variant": config.variant,
            "backbone": "lstm" if config.variant.endswith("lstm") else "transformer",
            "ssl_family": "contrastive" if config.variant.startswith("contrastive") else "byol",
            "backbone_config": TemporalBackboneConfig().to_dict(),
            "downstream_embedding_dim": 128,
            "trainable_parameter_count": parameter_count,
            "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "input_shape": [None, 64, 5],
            "target_interval_tokens_loaded": False,
        },
    )
    history: list[dict[str, float]] = []
    snapshots: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        loss = _train_epoch(model, loader, optimizer, device, config)
        health = _health(model, train, device)
        row = {
            "loss": loss,
            **health,
            "collapse_warning": float(health["embedding_std"] < config.collapse_std_threshold),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError("non-finite temporal encoder diagnostic")
        history.append(row)
        print(
            f"walk={config.walk} variant={config.variant} epoch={epoch} loss={loss:.8f} "
            f"embedding_std={health['embedding_std']:.8f} seconds={row['epoch_seconds']:.2f}",
            flush=True,
        )
        if epoch in config.snapshot_epochs:
            snapshot = run_root / f"e{epoch}"
            snapshot.mkdir()
            checkpoint = snapshot / "checkpoint.pth"
            torch.save(
                {
                    "phase": 6,
                    "purpose": "temporal_encoder_pretraining",
                    "walk": config.walk,
                    "variant": config.variant,
                    "completed_epoch": epoch,
                    "config": config.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dataset_sha256": sha256_file(dataset_path),
                    "parameter_count": parameter_count,
                },
                checkpoint,
            )
            _atomic_history(snapshot / "history.npz", history)
            metrics = {
                "epoch": epoch,
                "train_loss": loss,
                **health,
                "collapse_warning": bool(row["collapse_warning"]),
                "elapsed_training_seconds": time.perf_counter() - training_started,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                ),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
            write_json(snapshot / "metrics.json", metrics)
            snapshots.append(metrics)
    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    write_json(
        run_root / "training_complete.json",
        {
            "complete": True,
            "snapshots": list(config.snapshot_epochs),
            "principal_epoch": 50,
            "downstream_checkpoint_selected_from_results": False,
            "principal_checkpoint": str(run_root / "e50" / "checkpoint.pth"),
        },
    )
    return validate_encoder_variant(dataset_path, run_root)


def validate_encoder_variant(dataset_path: Path, run_root: Path) -> dict[str, Any]:
    required = {
        "config.json",
        "environment.json",
        "dataset_manifest.json",
        "architecture_manifest.json",
        "sweep_metrics.json",
        "training_complete.json",
    }
    missing = sorted(name for name in required if not (run_root / name).is_file())
    if missing:
        raise ValueError(f"temporal encoder run is missing artifacts: {missing}")
    config_payload = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    config_payload["snapshot_epochs"] = tuple(config_payload["snapshot_epochs"])
    config = Phase6EncoderVariantConfig(**{**config_payload, "device": "cpu"})
    train, manifest = _load_train(dataset_path, config.walk)
    source = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(dataset_path) != source["dataset_sha256"]:
        raise ValueError("temporal encoder dataset hash mismatch")
    if sha256_file(Path(f"{dataset_path}.manifest.json")) != source["dataset_manifest_sha256"]:
        raise ValueError("temporal encoder source-manifest hash mismatch")
    if source.get("encoder_train_identity_hash") != manifest["identity_hashes"]["encoder_train"]:
        raise ValueError("temporal encoder identity hash mismatch")
    if source.get("downstream_targets_loaded") or source.get("evaluation_values_loaded"):
        raise ValueError("temporal encoder target-independence invariant failed")
    architecture = json.loads(
        (run_root / "architecture_manifest.json").read_text(encoding="utf-8")
    )
    if (
        architecture.get("variant") != config.variant
        or architecture.get("backbone_config") != TemporalBackboneConfig().to_dict()
        or architecture.get("downstream_embedding_dim") != 128
        or architecture.get("input_shape") != [None, 64, 5]
        or architecture.get("target_interval_tokens_loaded") is not False
    ):
        raise ValueError("temporal encoder architecture manifest mismatch")
    rows = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))["snapshots"]
    if [int(row["epoch"]) for row in rows] != list(SNAPSHOT_EPOCHS):
        raise ValueError("temporal encoder snapshot matrix is incomplete")
    replayed = []
    for row in rows:
        epoch = int(row["epoch"])
        checkpoint_path = run_root / f"e{epoch}" / "checkpoint.pth"
        history_path = run_root / f"e{epoch}" / "history.npz"
        metrics_path = run_root / f"e{epoch}" / "metrics.json"
        if not all(path.is_file() for path in (checkpoint_path, history_path, metrics_path)):
            raise ValueError(f"temporal e{epoch} snapshot artifacts are incomplete")
        if sha256_file(checkpoint_path) != row["checkpoint_sha256"]:
            raise ValueError(f"temporal e{epoch} checkpoint hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("variant") != config.variant
            or checkpoint.get("walk") != config.walk
            or checkpoint.get("completed_epoch") != epoch
            or checkpoint.get("dataset_sha256") != sha256_file(dataset_path)
        ):
            raise ValueError(f"temporal e{epoch} checkpoint contract mismatch")
        model = build_variant_model(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        health = _health(model, train, torch.device("cpu"))
        if health["embedding_std"] < config.collapse_std_threshold:
            raise ValueError(f"temporal e{epoch} checkpoint is collapsed")
        replayed.append(epoch)
        with np.load(history_path, allow_pickle=False) as history:
            if set(history.files) != {
                "epochs",
                "loss",
                "embedding_std",
                "embedding_norm",
                "embedding_abs_mean",
                "collapse_warning",
                "epoch_seconds",
            }:
                raise ValueError(f"temporal e{epoch} history fields mismatch")
            if len(history["epochs"]) != epoch or int(history["epochs"][-1]) != epoch:
                raise ValueError(f"temporal e{epoch} history mismatch")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            metrics.get("checkpoint_sha256") != row["checkpoint_sha256"]
            or int(metrics.get("epoch", -1)) != epoch
            or bool(metrics.get("collapse_warning"))
        ):
            raise ValueError(f"temporal e{epoch} metrics contract mismatch")
    complete = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("principal_epoch") != 50:
        raise ValueError("temporal encoder completion marker is invalid")
    return {
        "valid": True,
        "walk": config.walk,
        "variant": config.variant,
        "snapshots": replayed,
        "encoder_train_identity_hash": manifest["identity_hashes"]["encoder_train"],
        "target_independent": True,
    }
