"""Reusable Phase 5 canonical encoder-pretraining implementation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_processing.phase5_walks import validate_phase5_bundle_files
from models.byol import BYOLEncoder, byol_loss
from models.contrastive import ContrastiveEncoder, make_views, nt_xent_loss
from models.vae import SequenceVAE, vae_loss


ENCODERS = ("vae", "contrastive", "byol")
SNAPSHOT_EPOCHS = (5, 15, 50)


@dataclass(frozen=True)
class Phase5EncoderConfig:
    encoder: str
    walk: int
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
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
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.encoder not in ENCODERS:
            raise ValueError(f"unknown Phase 5 encoder: {self.encoder}")
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("Phase 5 is frozen to one 50-epoch trajectory with 5/15/50 snapshots")
        if self.seed != 0:
            raise ValueError("Phase 5 primary encoder pretraining is frozen to seed 0")
        if self.batch_size <= 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["snapshot_epochs"] = list(self.snapshot_epochs)
        return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def environment_manifest(device: torch.device) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "python": os.sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": commit,
    }


def load_encoder_training_data(path: Path, walk: int) -> tuple[np.ndarray, dict[str, Any]]:
    validation = validate_phase5_bundle_files(path)
    if int(validation["walk"]) != walk:
        raise ValueError(f"dataset walk {validation['walk']} does not match requested walk {walk}")
    manifest_path = Path(f"{path}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as bundle:
        train = np.asarray(bundle["encoder_train_sequences"], dtype=np.float32)
    if train.ndim != 3 or train.shape[1:] != (64, 5) or not np.isfinite(train).all():
        raise ValueError("encoder_train_sequences must be finite float32 [N,64,5]")
    if len(train) != int(manifest["row_counts"]["encoder_train"]):
        raise ValueError("encoder train count disagrees with manifest")
    return train, manifest


def build_model(config: Phase5EncoderConfig, input_dim: int = 5) -> torch.nn.Module:
    if config.encoder == "vae":
        return SequenceVAE(64, input_dim, config.vae_latent_dim, config.vae_hidden_dim)
    if config.encoder == "contrastive":
        return ContrastiveEncoder(input_dim, config.hidden_dim, config.embedding_dim)
    return BYOLEncoder(input_dim, config.hidden_dim, config.embedding_dim, config.hidden_dim)


def build_loader(train: np.ndarray, config: Phase5EncoderConfig) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train)),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    if len(loader) == 0:
        raise ValueError("encoder population is smaller than the fixed batch size")
    return loader


def _check_gradients(model: torch.nn.Module) -> None:
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise FloatingPointError("non-finite encoder gradient")


def train_epoch(model, loader, optimizer, device, config: Phase5EncoderConfig) -> dict[str, float]:
    model.train()
    totals: dict[str, float]
    if config.encoder == "vae":
        totals = {"loss": 0.0, "recon": 0.0, "kld": 0.0}
    elif config.encoder == "byol":
        totals = {"loss": 0.0, "view_cosine": 0.0}
    else:
        totals = {"loss": 0.0}
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        if config.encoder == "vae":
            reconstruction, mu, logvar, _ = model(batch)
            loss, recon, kld = vae_loss(
                reconstruction, batch, mu, logvar, beta=config.vae_beta
            )
            values = {"loss": loss, "recon": recon, "kld": kld}
        elif config.encoder == "contrastive":
            view1, view2 = make_views(batch)
            _, projected1 = model(view1)
            _, projected2 = model(view2)
            loss = nt_xent_loss(projected1, projected2, temperature=config.temperature)
            values = {"loss": loss}
        else:
            view1, view2 = make_views(batch)
            _, _, prediction1, prediction2, target1, target2 = model(view1, view2)
            loss = byol_loss(prediction1, target2, prediction2, target1)
            cosine = 0.5 * (
                F.cosine_similarity(F.normalize(prediction1, dim=-1), target2, dim=-1).mean()
                + F.cosine_similarity(F.normalize(prediction2, dim=-1), target1, dim=-1).mean()
            )
            values = {"loss": loss, "view_cosine": cosine}
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {config.encoder} loss")
        loss.backward()
        _check_gradients(model)
        optimizer.step()
        if config.encoder == "byol":
            model.update_target(tau=config.target_decay)
        for name, value in values.items():
            totals[name] += float(value.detach().item())
    return {name: value / len(loader) for name, value in totals.items()}


@torch.no_grad()
def representation_health(
    model, encoder: str, train: np.ndarray, device: torch.device
) -> dict[str, float]:
    model.eval()
    chunks = []
    sample = torch.from_numpy(train[: min(len(train), 4096)]).to(device)
    for batch in sample.split(512):
        if encoder == "vae":
            embedding, _ = model.encode(batch)
        elif encoder == "contrastive":
            embedding = model(batch)[0]
        else:
            embedding = model.encode(batch)
        chunks.append(embedding.detach().cpu())
    values = torch.cat(chunks)
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite frozen representation diagnostic")
    return {
        "embedding_std": float(values.std(dim=0, unbiased=False).mean().item()),
        "embedding_norm": float(values.norm(dim=-1).mean().item()),
        "embedding_abs_mean": float(values.abs().mean().item()),
    }


def _history_payload(history: list[dict[str, float]], seconds: list[float]) -> dict[str, np.ndarray]:
    names = sorted({name for row in history for name in row})
    result = {
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        "epoch_seconds": np.asarray(seconds, dtype=np.float64),
    }
    for name in names:
        result[name] = np.asarray([row.get(name, np.nan) for row in history], dtype=np.float64)
    return result


def run_encoder_training(
    dataset_path: Path,
    run_root: Path,
    config: Phase5EncoderConfig,
) -> list[dict[str, Any]]:
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite encoder run: {run_root}")
    train, data_manifest = load_encoder_training_data(dataset_path, config.walk)
    device = resolve_device(config.device)
    set_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config, int(train.shape[2])).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = build_loader(train, config)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dataset_hash = sha256_file(dataset_path)
    manifest_hash = sha256_file(Path(f"{dataset_path}.manifest.json"))

    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "config.json", config.to_dict())
    write_json(run_root / "environment.json", environment_manifest(device))
    write_json(
        run_root / "dataset_manifest.json",
        {
            "walk": config.walk,
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_hash,
            "source_manifest_sha256": manifest_hash,
            "encoder_train_shape": list(train.shape),
            "encoder_train_identity_hash": data_manifest["identity_hashes"]["encoder_train"],
            "input_key": "encoder_train_sequences",
            "downstream_targets_loaded": False,
            "evaluation_values_loaded": False,
        },
    )
    write_json(
        run_root / "architecture_manifest.json",
        {
            "encoder": config.encoder,
            "backbone": "mlp" if config.encoder == "vae" else "cnn",
            "trainable_parameter_count": int(parameter_count),
            "downstream_embedding_dim": 64 if config.encoder == "vae" else 128,
            "input_channels": ["open", "high", "low", "close", "volume"],
            "explicit_imputation_mask_input": False,
            "augmentation": (
                "none" if config.encoder == "vae" else "existing scaling/jitter/time-mask on all OHLCV"
            ),
        },
    )

    history: list[dict[str, float]] = []
    epoch_seconds: list[float] = []
    snapshots: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        diagnostics = train_epoch(model, loader, optimizer, device, config)
        health = representation_health(model, config.encoder, train, device)
        diagnostics.update(health)
        if not all(math.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError("non-finite encoder diagnostic")
        diagnostics["collapse_warning"] = float(
            health["embedding_std"] < config.collapse_std_threshold
        )
        history.append(diagnostics)
        epoch_seconds.append(time.perf_counter() - epoch_started)
        print(
            f"walk={config.walk} encoder={config.encoder} epoch={epoch} "
            f"loss={diagnostics['loss']:.8f} embedding_std={health['embedding_std']:.8f} "
            f"seconds={epoch_seconds[-1]:.2f}",
            flush=True,
        )
        if epoch not in config.snapshot_epochs:
            continue
        snapshot_dir = run_root / f"e{epoch}"
        snapshot_dir.mkdir()
        checkpoint_path = snapshot_dir / "checkpoint.pth"
        torch.save(
            {
                "phase": 5,
                "purpose": "encoder_pretraining",
                "walk": config.walk,
                "encoder": config.encoder,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "completed_epoch": epoch,
                "seq_len": 64,
                "input_dim": 5,
                "training_config": config.to_dict(),
                "dataset_sha256": dataset_hash,
                "source_manifest_sha256": manifest_hash,
                "encoder_train_identity_hash": data_manifest["identity_hashes"]["encoder_train"],
                "trainable_parameter_count": int(parameter_count),
            },
            checkpoint_path,
        )
        np.savez_compressed(
            snapshot_dir / "history.npz", **_history_payload(history, epoch_seconds)
        )
        row: dict[str, Any] = {
            "walk": config.walk,
            "encoder": config.encoder,
            "epoch": epoch,
            "seed": config.seed,
            "train_loss": diagnostics["loss"],
            "best_train_loss": min(item["loss"] for item in history),
            **health,
            "collapse_warning": bool(diagnostics["collapse_warning"]),
            "trainable_parameter_count": int(parameter_count),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        for name in ("recon", "kld", "view_cosine"):
            if name in diagnostics:
                row[name] = diagnostics[name]
        write_json(snapshot_dir / "metrics.json", row)
        snapshots.append(row)

    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    write_json(
        run_root / "training_complete.json",
        {
            "complete": True,
            "phase": 5,
            "walk": config.walk,
            "encoder": config.encoder,
            "snapshots": list(config.snapshot_epochs),
            "principal_downstream_checkpoint": str(run_root / "e50" / "checkpoint.pth"),
            "selection_rule": "epoch 50 predeclared; no downstream or evaluation metric used",
        },
    )
    write_encoder_summary(run_root)
    return snapshots


def write_encoder_summary(run_root: Path) -> None:
    config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    snapshots = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))[
        "snapshots"
    ]
    lines = [
        f"# Phase 5 Walk {config['walk']} {config['encoder'].title()} Encoder",
        "",
        "One uninterrupted seed-0 trajectory; epoch 50 is predeclared for downstream use.",
        "",
        "| Epoch | Train loss | Embedding std | Collapse warning |",
        "|---:|---:|---:|:---:|",
    ]
    for row in snapshots:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.8f} | "
            f"{row['embedding_std']:.8f} | {str(row['collapse_warning']).lower()} |"
        )
    lines.extend(
        [
            "",
            "The loss and health diagnostics are unsupervised training evidence only; no evaluation or downstream metric selected a checkpoint.",
            "",
        ]
    )
    (run_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def validate_encoder_run(dataset_path: Path, run_root: Path) -> dict[str, Any]:
    """Replay one completed Phase 5 encoder run on CPU."""

    required_root = {
        "config.json",
        "environment.json",
        "dataset_manifest.json",
        "architecture_manifest.json",
        "sweep_metrics.json",
        "training_complete.json",
        "summary.md",
    }
    missing = sorted(name for name in required_root if not (run_root / name).is_file())
    if missing:
        raise ValueError(f"encoder run is missing root artifacts: {missing}")
    config_payload = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    config_payload["snapshot_epochs"] = tuple(config_payload["snapshot_epochs"])
    config = Phase5EncoderConfig(**config_payload)
    train, data_manifest = load_encoder_training_data(dataset_path, config.walk)
    dataset_hash = sha256_file(dataset_path)
    source_manifest_hash = sha256_file(Path(f"{dataset_path}.manifest.json"))
    recorded_data = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if recorded_data.get("dataset_sha256") != dataset_hash:
        raise ValueError("encoder run dataset hash mismatch")
    if recorded_data.get("source_manifest_sha256") != source_manifest_hash:
        raise ValueError("encoder run source-manifest hash mismatch")
    if recorded_data.get("encoder_train_identity_hash") != data_manifest["identity_hashes"]["encoder_train"]:
        raise ValueError("encoder run identity hash mismatch")
    if recorded_data.get("evaluation_values_loaded") is not False or recorded_data.get("downstream_targets_loaded") is not False:
        raise ValueError("encoder run must explicitly exclude evaluation values and targets")

    model = build_model(config)
    snapshots = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))["snapshots"]
    if [int(row["epoch"]) for row in snapshots] != list(SNAPSHOT_EPOCHS):
        raise ValueError("encoder snapshot matrix is incomplete")
    for row in snapshots:
        epoch = int(row["epoch"])
        snapshot_dir = run_root / f"e{epoch}"
        checkpoint_path = snapshot_dir / "checkpoint.pth"
        history_path = snapshot_dir / "history.npz"
        metrics_path = snapshot_dir / "metrics.json"
        if not checkpoint_path.is_file() or not history_path.is_file() or not metrics_path.is_file():
            raise ValueError(f"epoch {epoch} snapshot artifacts are incomplete")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if sha256_file(checkpoint_path) != metrics.get("checkpoint_sha256"):
            raise ValueError(f"epoch {epoch} checkpoint hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        expected = {
            "phase": 5,
            "purpose": "encoder_pretraining",
            "walk": config.walk,
            "encoder": config.encoder,
            "completed_epoch": epoch,
            "dataset_sha256": dataset_hash,
            "source_manifest_sha256": source_manifest_hash,
            "encoder_train_identity_hash": data_manifest["identity_hashes"]["encoder_train"],
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"epoch {epoch} checkpoint {key} mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        with np.load(history_path, allow_pickle=False) as history:
            if len(history["epochs"]) != epoch or int(history["epochs"][-1]) != epoch:
                raise ValueError(f"epoch {epoch} cumulative history mismatch")
        if not math.isfinite(float(metrics["train_loss"])):
            raise ValueError(f"epoch {epoch} training loss is non-finite")

    model.eval()
    sample = torch.from_numpy(train[:8])
    with torch.no_grad():
        if config.encoder == "vae":
            embedding, _ = model.encode(sample)
            expected_dim = 64
        elif config.encoder == "contrastive":
            embedding = model(sample)[0]
            expected_dim = 128
        else:
            embedding = model.encode(sample)
            expected_dim = 128
    if embedding.shape != (8, expected_dim) or not torch.isfinite(embedding).all():
        raise ValueError("epoch-50 checkpoint inference replay failed")
    complete = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("snapshots") != list(SNAPSHOT_EPOCHS):
        raise ValueError("encoder completion marker is invalid")
    return {
        "valid": True,
        "walk": config.walk,
        "encoder": config.encoder,
        "snapshots": list(SNAPSHOT_EPOCHS),
        "epoch50_checkpoint": str(run_root / "e50" / "checkpoint.pth"),
        "epoch50_checkpoint_sha256": snapshots[-1]["checkpoint_sha256"],
        "epoch50_train_loss": snapshots[-1]["train_loss"],
        "epoch50_embedding_std": snapshots[-1]["embedding_std"],
        "collapse_warning": snapshots[-1]["collapse_warning"],
    }
