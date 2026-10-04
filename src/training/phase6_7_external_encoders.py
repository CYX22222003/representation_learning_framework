"""Phase 6.7 walk-specific external representation pretraining.

The first registered method is the independently authored SaURL-TS adapter.
This module owns optimizer boundaries, alternating updates, checkpointing, and
resume; the model package intentionally exposes forward operations only.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from baselines.saurl_ts import BRANCH_NAMES, SaURLConfig, SaURLInputScaler, build_saurl
from data_processing.phase5_walks import sha256_arrays, sha256_file, validate_phase5_bundle_files
from features.phase5_features import array_sha256
from training.phase5_encoder import environment_manifest, resolve_device, set_seed, write_json


METHODS = ("saurl_frozen",)
SNAPSHOT_EPOCHS = (5, 15, 50)
SCHEMA_VERSION = "phase6-7-external-encoder-v1"
REPORTING_LABEL = "SaURL-TS-Frozen (paper-guided reimplementation)"


@dataclass(frozen=True)
class Phase67ExternalEncoderConfig:
    method: str
    walk: int
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    collapse_std_threshold: float = 1e-3
    device: str = "cuda"

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_epochs", tuple(self.snapshot_epochs))
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("Phase 6.7 encoders require 50 epochs and 5/15/50 snapshots")
        if self.seed != 0:
            raise ValueError("Phase 6.7 primary pretraining is frozen to seed 0")
        if self.collapse_std_threshold <= 0.0:
            raise ValueError("collapse_std_threshold must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot_epochs"] = list(self.snapshot_epochs)
        return payload


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _numpy_rng_state() -> dict[str, Any]:
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": name,
        "keys": torch.from_numpy(keys.copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": _numpy_rng_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler_generator": generator.get_state(),
    }


def _restore_rng_state(state: Mapping[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state(
        (
            str(numpy["name"]),
            np.asarray(numpy["keys"].cpu(), dtype=np.uint32),
            int(numpy["position"]),
            int(numpy["has_gauss"]),
            float(numpy["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = list(state.get("torch_cuda", []))
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)
    generator.set_state(state["sampler_generator"].cpu())


def no_drop_batch_indices(
    row_count: int, batch_size: int, generator: torch.Generator
) -> list[torch.Tensor]:
    """Return one shuffled full pass, merging a singleton remainder backward."""

    if row_count < 2:
        raise ValueError("SaURL MMD pretraining requires at least two rows")
    if batch_size < 2:
        raise ValueError("SaURL batch size must be at least two")
    permutation = torch.randperm(row_count, generator=generator)
    batches = list(permutation.split(batch_size))
    if len(batches[-1]) == 1:
        if len(batches) == 1:
            raise ValueError("cannot form a non-singleton SaURL batch")
        batches[-2] = torch.cat((batches[-2], batches[-1]))
        batches.pop()
    if sum(len(batch) for batch in batches) != row_count:
        raise AssertionError("no-drop sampler did not preserve every row exactly once")
    if any(len(batch) < 2 for batch in batches):
        raise AssertionError("SaURL no-drop sampler produced a singleton batch")
    return batches


def _load_encoder_population(
    dataset_path: Path, walk: int
) -> tuple[np.ndarray, dict[str, Any]]:
    validation = validate_phase5_bundle_files(dataset_path)
    if int(validation["walk"]) != walk:
        raise ValueError("SaURL encoder dataset walk mismatch")
    manifest_path = Path(f"{dataset_path}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=False) as stored:
        sequences = np.asarray(stored["encoder_train_sequences"], dtype=np.float32)
        identities = tuple(
            np.asarray(stored[f"encoder_train_{field}"])
            for field in (
                "condition_ids",
                "window_start_ns",
                "decision_date_ns",
                "decision_availability_ns",
            )
        )
    if sequences.ndim != 3 or sequences.shape[1:] != (64, 5):
        raise ValueError("encoder_train_sequences must have shape [N,64,5]")
    if not np.isfinite(sequences).all():
        raise ValueError("encoder_train_sequences must be finite")
    identity_hash = sha256_arrays(*identities)
    expected = manifest.get("identity_hashes", {}).get("encoder_train")
    if expected != identity_hash:
        raise ValueError("encoder training identity hash mismatch")
    return sequences, manifest


def _set_requires_grad(parameters: Iterable[torch.nn.Parameter], enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def _clear_gradients(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _assert_finite_gradients(parameters: Iterable[torch.nn.Parameter], label: str) -> None:
    seen = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        seen = True
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite {label} gradient")
    if not seen:
        raise RuntimeError(f"{label} update produced no gradients")


def _assert_no_gradients(parameters: Iterable[torch.nn.Parameter], label: str) -> None:
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError(f"{label} unexpectedly received gradients")


def _assert_finite_tensors(tensors: Iterable[torch.Tensor], label: str) -> None:
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise FloatingPointError(f"non-finite {label} tensor")


def _view_tensors(parts: Any) -> tuple[torch.Tensor, ...]:
    return (
        parts.views.temporal_view1,
        parts.views.temporal_view2,
        parts.views.frequency_view1,
        parts.views.frequency_view2,
        parts.temporal.view1.mask,
        parts.frequency.view1.mask,
    )


def _sada_values(losses: Any, parts: Any) -> dict[str, float]:
    values = {
        "sada_loss": float(losses.total.detach()),
        "sada_temporal_diversity": float(losses.temporal_diversity.detach()),
        "sada_frequency_diversity": float(losses.frequency_diversity.detach()),
        "temporal_mask_rate": float(parts.temporal.view1.mask.detach().mean()),
        "frequency_mask_rate": float(parts.frequency.view1.mask.detach().mean()),
    }
    for prefix, value in (
        ("temporal_view1", losses.temporal_view1),
        ("temporal_view2", losses.temporal_view2),
        ("frequency_view1", losses.frequency_view1),
        ("frequency_view2", losses.frequency_view2),
    ):
        values[f"{prefix}_cardinality"] = float(value.mask_cardinality.detach())
        values[f"{prefix}_preservation"] = float(value.preservation.detach())
        values[f"{prefix}_continuity"] = float(value.continuity.detach())
        values[f"{prefix}_dissimilarity"] = float(value.dissimilarity.detach())
    return values


def _sassl_values(outputs: Any) -> dict[str, float]:
    return {
        "sassl_loss": float(outputs.losses.total.detach()),
        "sassl_time": float(outputs.losses.time.detach()),
        "sassl_frequency": float(outputs.losses.frequency.detach()),
        "sassl_cross": float(outputs.losses.cross.detach()),
    }


def _mean_totals(totals: Mapping[str, float], counts: Mapping[str, int]) -> dict[str, float]:
    return {
        name: float(value / counts["sada" if name.startswith("sada_") or "mask" in name or "view" in name else "sassl"])
        for name, value in totals.items()
    }


@torch.no_grad()
def _representation_health(
    model: torch.nn.Module,
    scaler: SaURLInputScaler,
    sequences: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    fused: list[torch.Tensor] = []
    branches: dict[str, list[torch.Tensor]] = {name: [] for name in BRANCH_NAMES}
    sample = torch.from_numpy(sequences[: min(len(sequences), 4096)])
    for batch in sample.split(512):
        normalized = scaler.transform(batch.to(device))
        parts = model.encode_parts(normalized)
        fused.append(parts.fused.cpu())
        for index, name in enumerate(BRANCH_NAMES):
            branches[name].append(parts.branches[:, index].cpu())
    fused_values = torch.cat(fused)
    result = {
        "embedding_std": float(fused_values.std(dim=0, unbiased=False).mean()),
        "embedding_norm": float(fused_values.norm(dim=-1).mean()),
        "embedding_abs_mean": float(fused_values.abs().mean()),
    }
    for name in BRANCH_NAMES:
        values = torch.cat(branches[name])
        result[f"{name}_embedding_std"] = float(values.std(dim=0, unbiased=False).mean())
        result[f"{name}_embedding_norm"] = float(values.norm(dim=-1).mean())
    return result


@torch.no_grad()
def _probe_payload(
    model: torch.nn.Module, scaler: SaURLInputScaler, raw_probe: torch.Tensor
) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    normalized = scaler.transform(raw_probe.to(next(model.parameters()).device))
    parts = model.encode_parts(normalized)
    payload = {
        "raw_input": raw_probe.detach().cpu(),
        "normalized_input": normalized.detach().cpu(),
        "branches": parts.branches.detach().cpu(),
        "attention_weights": parts.attention_weights.detach().cpu(),
        "weighted_branches": parts.weighted_branches.detach().cpu(),
        "fused": parts.fused.detach().cpu(),
    }
    model.train(was_training)
    return payload


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    scaler: SaURLInputScaler,
    optimizers: Mapping[str, torch.optim.Optimizer],
    config: Phase67ExternalEncoderConfig,
    model_config: SaURLConfig,
    dataset_path: Path,
    data_manifest: Mapping[str, Any],
    completed_epoch: int,
    global_step: int,
    history: list[dict[str, float]],
    generator: torch.Generator,
    raw_probe: torch.Tensor,
    code_revision: str | None,
    device_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    history_hash = hashlib.sha256(
        json.dumps(history, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "6.7",
        "purpose": "external_encoder_pretraining",
        "method": config.method,
        "reporting_label": REPORTING_LABEL,
        "walk": config.walk,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "config": config.to_dict(),
        "model_config": model_config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_states": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "input_scaler": scaler.state_dict(),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_manifest_sha256": sha256_file(Path(f"{dataset_path}.manifest.json")),
        "encoder_train_identity_hash": data_manifest["identity_hashes"]["encoder_train"],
        "history": history,
        "loss_history_sha256": history_hash,
        "rng_states": _capture_rng_state(generator),
        "sampler_state": {"next_epoch": completed_epoch + 1, "batch_position": 0},
        "probe": _probe_payload(model, scaler, raw_probe),
        "code_revision": code_revision,
        "device_metadata": dict(device_metadata),
    }


def _config_from_payload(payload: Mapping[str, Any], device: str) -> Phase67ExternalEncoderConfig:
    values = dict(payload)
    values["snapshot_epochs"] = tuple(values["snapshot_epochs"])
    values["device"] = device
    return Phase67ExternalEncoderConfig(**values)


def _model_config_from_payload(payload: Mapping[str, Any]) -> SaURLConfig:
    values = dict(payload)
    values["dilations"] = tuple(values["dilations"])
    return SaURLConfig(**values)


def build_external_encoder(
    config: Phase67ExternalEncoderConfig, model_config: SaURLConfig | None = None
) -> torch.nn.Module:
    if config.method != "saurl_frozen":
        raise ValueError(f"unimplemented external method: {config.method}")
    return build_saurl(model_config or SaURLConfig())


def smoke_test_external_encoders() -> dict[str, Any]:
    """CPU synthetic forward/backward smoke; it writes no experiment artifacts."""

    set_seed(0)
    config = Phase67ExternalEncoderConfig("saurl_frozen", walk=1, device="cpu")
    model_config = SaURLConfig()
    model = build_external_encoder(config, model_config)
    values = torch.randn(4, 64, 5)
    scaler = SaURLInputScaler.fit(values)
    x = scaler.transform(values)
    opt_time = torch.optim.Adam(model.temporal_sada.parameters(), lr=model_config.sada_learning_rate)
    opt_frequency = torch.optim.Adam(model.frequency_sada.parameters(), lr=model_config.sada_learning_rate)
    opt_sassl = torch.optim.Adam(model.sassl_parameters(), lr=model_config.sassl_learning_rate)
    _training_step(
        model,
        x,
        global_step=0,
        optimizers={"sada_time": opt_time, "sada_frequency": opt_frequency, "sassl": opt_sassl},
        model_config=model_config,
    )
    model.eval()
    output = model.encode(x)
    if output.shape != (4, 128) or not torch.isfinite(output).all():
        raise RuntimeError("SaURL external encoder smoke failed")
    return {
        "valid": True,
        "method": config.method,
        "output_shape": list(output.shape),
        "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_representation_parameter_count": int(
            sum(parameter.numel() for parameter in model.sassl_parameters())
        ),
    }


def resource_smoke_external_encoder(
    dataset_path: Path,
    *,
    walk: int,
    device: str,
) -> dict[str, Any]:
    """Run one real training-only joint update for the pre-experiment gate."""

    resolved = resolve_device(device)
    train, manifest = _load_encoder_population(dataset_path, walk)
    set_seed(0)
    model_config = SaURLConfig()
    config = Phase67ExternalEncoderConfig(METHODS[0], walk, device=str(resolved))
    model = build_external_encoder(config, model_config).to(resolved)
    scaler = SaURLInputScaler.fit(torch.from_numpy(train))
    optimizers = _optimizer_bundle(model, model_config)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)
    started = time.perf_counter()
    batch = scaler.transform(torch.from_numpy(train[:4]).to(resolved)).float()
    metrics = _training_step(
        model,
        batch,
        global_step=0,
        optimizers=optimizers,
        model_config=model_config,
    )
    model.eval()
    with torch.no_grad():
        embedding = model.encode(batch)
    elapsed = time.perf_counter() - started
    if embedding.shape != (4, model_config.representation_dim):
        raise ValueError("real-batch resource smoke embedding shape mismatch")
    if not torch.isfinite(embedding).all() or not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("real-batch resource smoke produced non-finite values")
    return {
        "valid": True,
        "walk": walk,
        "device": str(resolved),
        "rows": 4,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(resolved)) if resolved.type == "cuda" else 0
        ),
        "total_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_manifest_sha256": sha256_file(Path(f"{dataset_path}.manifest.json")),
        "encoder_train_identity_hash": manifest["identity_hashes"]["encoder_train"],
        "embedding_hash": array_sha256(embedding.cpu().numpy()),
        "losses": metrics,
        "downstream_targets_loaded": False,
        "evaluation_values_loaded": False,
    }


def _training_step(
    model: Any,
    x: torch.Tensor,
    *,
    global_step: int,
    optimizers: Mapping[str, torch.optim.Optimizer],
    model_config: SaURLConfig,
) -> dict[str, float]:
    model.train()
    sada_parameters = list(model.sada_parameters())
    sassl_parameters = list(model.sassl_parameters())
    metrics: dict[str, float] = {}
    if global_step % model_config.ratio_step == 0:
        _clear_gradients(model)
        _set_requires_grad(sada_parameters, True)
        _set_requires_grad(sassl_parameters, False)
        optimizers["sada_time"].zero_grad(set_to_none=True)
        optimizers["sada_frequency"].zero_grad(set_to_none=True)
        parts = model.make_views(x)
        losses = model.sada_losses(parts)
        _assert_finite_tensors((*_view_tensors(parts), losses.total), "SaDA")
        losses.total.backward()
        _assert_finite_gradients(sada_parameters, "SaDA")
        _assert_no_gradients(sassl_parameters, "SaSSL during SaDA")
        optimizers["sada_time"].step()
        optimizers["sada_frequency"].step()
        metrics.update(_sada_values(losses, parts))

    _clear_gradients(model)
    _set_requires_grad(sada_parameters, False)
    _set_requires_grad(sassl_parameters, True)
    with torch.no_grad():
        views = model.make_views(x).detached_views()
    optimizers["sassl"].zero_grad(set_to_none=True)
    outputs = model.sassl_forward(views)
    _assert_finite_tensors(
        (
            outputs.losses.total,
            outputs.online_a.branches,
            outputs.online_b.branches,
            outputs.online_a.fused,
            outputs.online_b.fused,
        ),
        "SaSSL",
    )
    outputs.total_loss.backward()
    _assert_finite_gradients(sassl_parameters, "SaSSL")
    _assert_no_gradients(sada_parameters, "SaDA during SaSSL")
    optimizers["sassl"].step()
    model.update_targets(tau=model_config.ema_tau)
    metrics.update(_sassl_values(outputs))
    return metrics


def _optimizer_bundle(model: Any, config: SaURLConfig) -> dict[str, torch.optim.Optimizer]:
    return {
        "sada_time": torch.optim.Adam(
            model.temporal_sada.parameters(),
            lr=config.sada_learning_rate,
            weight_decay=config.weight_decay,
        ),
        "sada_frequency": torch.optim.Adam(
            model.frequency_sada.parameters(),
            lr=config.sada_learning_rate,
            weight_decay=config.weight_decay,
        ),
        "sassl": torch.optim.Adam(
            model.sassl_parameters(),
            lr=config.sassl_learning_rate,
            weight_decay=config.weight_decay,
        ),
    }


def _write_run_manifests(
    run_root: Path,
    dataset_path: Path,
    train: np.ndarray,
    data_manifest: Mapping[str, Any],
    config: Phase67ExternalEncoderConfig,
    model_config: SaURLConfig,
    model: torch.nn.Module,
    scaler: SaURLInputScaler,
    device: torch.device,
) -> None:
    run_root.mkdir(parents=True, exist_ok=False)
    write_json(run_root / "config.json", config.to_dict())
    write_json(run_root / "model_config.json", model_config.to_dict())
    write_json(run_root / "environment.json", environment_manifest(device))
    write_json(
        run_root / "dataset_manifest.json",
        {
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_manifest_sha256": sha256_file(Path(f"{dataset_path}.manifest.json")),
            "encoder_train_identity_hash": data_manifest["identity_hashes"]["encoder_train"],
            "encoder_train_sequence_hash": data_manifest["sequence_hashes"]["encoder_train"],
            "encoder_train_rows": len(train),
            "loaded_array": "encoder_train_sequences",
            "downstream_targets_loaded": False,
            "evaluation_values_loaded": False,
        },
    )
    scaler_arrays = {
        "mean": scaler.mean.numpy(),
        "scale": scaler.scale.numpy(),
        "minimum_scale": np.asarray(scaler.minimum_scale, dtype=np.float64),
    }
    _atomic_savez(run_root / "input_scaler.npz", scaler_arrays)
    write_json(
        run_root / "input_scaler.manifest.json",
        {
            "fit_population": "walk-local encoder_train_sequences only",
            "evaluation_used": False,
            "width": 5,
            "sha256": sha256_file(run_root / "input_scaler.npz"),
            "array_hashes": {name: array_sha256(value) for name, value in scaler_arrays.items()},
        },
    )
    sada_count = int(sum(parameter.numel() for parameter in model.sada_parameters()))
    sassl_count = int(sum(parameter.numel() for parameter in model.sassl_parameters()))
    total_count = int(sum(parameter.numel() for parameter in model.parameters()))
    write_json(
        run_root / "architecture_manifest.json",
        {
            "method": config.method,
            "reporting_label": REPORTING_LABEL,
            "independent_reimplementation": True,
            "upstream_code_reused": False,
            "input_shape": [None, 64, 5],
            "downstream_embedding_dim": 128,
            "extraction_components": ["online_encoders", "representation_wise_attention"],
            "discarded_components": ["sada", "online_projectors", "online_predictors", "ema_targets"],
            "sada_parameter_count": sada_count,
            "sassl_online_parameter_count": sassl_count,
            "total_parameter_count": total_count,
        },
    )


def run_external_encoder(
    dataset_path: Path,
    run_root: Path,
    config: Phase67ExternalEncoderConfig,
) -> dict[str, Any]:
    """Train or resume one frozen Phase 6.7 SaURL trajectory."""

    train, data_manifest = _load_encoder_population(dataset_path, config.walk)
    device = resolve_device(config.device)
    model_config = SaURLConfig()
    if run_root.exists() and (run_root / "training_complete.json").is_file():
        raise FileExistsError(f"refusing to overwrite completed external encoder run: {run_root}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    if not run_root.exists():
        fresh_run = True
        set_seed(config.seed)
        model = build_external_encoder(config, model_config).to(device)
        scaler = SaURLInputScaler.fit(torch.from_numpy(train))
        optimizers = _optimizer_bundle(model, model_config)
        _write_run_manifests(
            run_root,
            dataset_path,
            train,
            data_manifest,
            config,
            model_config,
            model,
            scaler,
            device,
        )
        start_epoch = 1
        global_step = 0
        history: list[dict[str, float]] = []
    else:
        fresh_run = False
        resume_path = run_root / "resume.pth"
        if not resume_path.is_file():
            raise ValueError("incomplete external encoder run has no resume checkpoint")
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        saved_config = _config_from_payload(resume["config"], config.device)
        if saved_config != config:
            raise ValueError("external encoder resume config drift")
        model_config = _model_config_from_payload(resume["model_config"])
        if resume["dataset_sha256"] != sha256_file(dataset_path):
            raise ValueError("external encoder resume dataset drift")
        model = build_external_encoder(config, model_config).to(device)
        model.load_state_dict(resume["model_state_dict"], strict=True)
        scaler = SaURLInputScaler.from_state_dict(resume["input_scaler"])
        optimizers = _optimizer_bundle(model, model_config)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(resume["optimizer_states"][name])
        _restore_rng_state(resume["rng_states"], generator)
        start_epoch = int(resume["completed_epoch"]) + 1
        global_step = int(resume["global_step"])
        history = [dict(row) for row in resume["history"]]

    environment = json.loads((run_root / "environment.json").read_text(encoding="utf-8"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    raw_probe = torch.from_numpy(train[:4].copy())
    if fresh_run:
        _atomic_torch_save(
            run_root / "resume.pth",
            _checkpoint_payload(
                model=model,
                scaler=scaler,
                optimizers=optimizers,
                config=config,
                model_config=model_config,
                dataset_path=dataset_path,
                data_manifest=data_manifest,
                completed_epoch=0,
                global_step=0,
                history=[],
                generator=generator,
                raw_probe=raw_probe,
                code_revision=environment.get("git_commit"),
                device_metadata=environment,
            ),
        )
    elapsed_before_resume = float(sum(row.get("epoch_seconds", 0.0) for row in history))
    training_started = time.perf_counter()
    snapshots: list[dict[str, Any]] = []
    for snapshot_epoch in config.snapshot_epochs:
        metrics_path = run_root / f"e{snapshot_epoch}" / "metrics.json"
        if snapshot_epoch < start_epoch and metrics_path.is_file():
            snapshots.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_started = time.perf_counter()
        batches = no_drop_batch_indices(len(train), model_config.batch_size, generator)
        totals: dict[str, float] = {}
        sada_updates = 0
        for indices in batches:
            raw_batch = torch.from_numpy(train[indices.numpy()]).to(device)
            x = scaler.transform(raw_batch).float()
            step_values = _training_step(
                model,
                x,
                global_step=global_step,
                optimizers=optimizers,
                model_config=model_config,
            )
            if "sada_loss" in step_values:
                sada_updates += 1
            for name, value in step_values.items():
                totals[name] = totals.get(name, 0.0) + value
            global_step += 1
        counts = {"sada": sada_updates, "sassl": len(batches)}
        means = _mean_totals(totals, counts)
        health = _representation_health(model, scaler, train, device)
        row = {
            **means,
            **health,
            "sada_updates": float(sada_updates),
            "sassl_updates": float(len(batches)),
            "global_step": float(global_step),
            "collapse_warning": float(health["embedding_std"] < config.collapse_std_threshold),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError("non-finite SaURL epoch diagnostic")
        history.append(row)
        resume_payload = _checkpoint_payload(
            model=model,
            scaler=scaler,
            optimizers=optimizers,
            config=config,
            model_config=model_config,
            dataset_path=dataset_path,
            data_manifest=data_manifest,
            completed_epoch=epoch,
            global_step=global_step,
            history=history,
            generator=generator,
            raw_probe=raw_probe,
            code_revision=environment.get("git_commit"),
            device_metadata=environment,
        )
        _atomic_torch_save(run_root / "resume.pth", resume_payload)
        print(
            f"walk={config.walk} method={config.method} epoch={epoch} "
            f"sada={row.get('sada_loss', float('nan')):.8f} "
            f"sassl={row['sassl_loss']:.8f} embedding_std={row['embedding_std']:.8f} "
            f"seconds={row['epoch_seconds']:.2f}",
            flush=True,
        )
        if epoch in config.snapshot_epochs:
            snapshot = run_root / f"e{epoch}"
            snapshot.mkdir(exist_ok=False)
            checkpoint_path = snapshot / "checkpoint.pth"
            _atomic_torch_save(checkpoint_path, resume_payload)
            _atomic_savez(snapshot / "history.npz", _history_arrays(history))
            metrics = {
                "epoch": epoch,
                **row,
                "elapsed_training_seconds": elapsed_before_resume
                + time.perf_counter()
                - training_started,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                ),
                "checkpoint_sha256": sha256_file(checkpoint_path),
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
            "principal_checkpoint": str((run_root / "e50" / "checkpoint.pth").resolve()),
            "selection_rule": "epoch 50 predeclared; no evaluation-driven selection",
        },
    )
    return validate_external_encoder(dataset_path, run_root)


def _history_arrays(history: list[Mapping[str, float]]) -> dict[str, np.ndarray]:
    names = sorted({name for row in history for name in row})
    return {
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        **{
            name: np.asarray([row.get(name, np.nan) for row in history], dtype=np.float64)
            for name in names
        },
    }


@torch.no_grad()
def _replay_probe(model: Any, scaler: SaURLInputScaler, probe: Mapping[str, torch.Tensor]) -> None:
    model.eval()
    raw_input = probe["raw_input"]
    normalized = scaler.transform(raw_input)
    if not torch.allclose(
        normalized, probe["normalized_input"], rtol=2e-5, atol=2e-6
    ):
        raise ValueError("SaURL checkpoint scaler probe mismatch")
    parts = model.encode_parts(normalized)
    for name, replayed in (
        ("branches", parts.branches),
        ("attention_weights", parts.attention_weights),
        ("weighted_branches", parts.weighted_branches),
        ("fused", parts.fused),
    ):
        if not torch.allclose(replayed.cpu(), probe[name], rtol=2e-4, atol=2e-5):
            raise ValueError(f"SaURL checkpoint probe mismatch: {name}")


def validate_external_encoder(dataset_path: Path, run_root: Path) -> dict[str, Any]:
    required = {
        "config.json",
        "model_config.json",
        "environment.json",
        "dataset_manifest.json",
        "input_scaler.npz",
        "input_scaler.manifest.json",
        "architecture_manifest.json",
        "sweep_metrics.json",
        "training_complete.json",
    }
    missing = sorted(name for name in required if not (run_root / name).is_file())
    if missing:
        raise ValueError(f"external encoder run is missing artifacts: {missing}")
    config = _config_from_payload(
        json.loads((run_root / "config.json").read_text(encoding="utf-8")), "cpu"
    )
    train, data_manifest = _load_encoder_population(dataset_path, config.walk)
    source = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if source.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("external encoder dataset hash mismatch")
    if source.get("dataset_manifest_sha256") != sha256_file(Path(f"{dataset_path}.manifest.json")):
        raise ValueError("external encoder dataset-manifest hash mismatch")
    if source.get("encoder_train_identity_hash") != data_manifest["identity_hashes"]["encoder_train"]:
        raise ValueError("external encoder identity hash mismatch")
    if source.get("downstream_targets_loaded") or source.get("evaluation_values_loaded"):
        raise ValueError("external encoder target-independence invariant failed")
    scaler_manifest = json.loads(
        (run_root / "input_scaler.manifest.json").read_text(encoding="utf-8")
    )
    scaler_path = run_root / "input_scaler.npz"
    if scaler_manifest.get("sha256") != sha256_file(scaler_path):
        raise ValueError("external encoder scaler hash mismatch")
    with np.load(scaler_path, allow_pickle=False) as stored:
        scaler = SaURLInputScaler(
            mean=torch.from_numpy(np.asarray(stored["mean"])),
            scale=torch.from_numpy(np.asarray(stored["scale"])),
            minimum_scale=float(stored["minimum_scale"]),
        )
    fitted = SaURLInputScaler.fit(torch.from_numpy(train))
    if not torch.equal(scaler.mean, fitted.mean) or not torch.equal(scaler.scale, fitted.scale):
        raise ValueError("external encoder scaler does not replay from training rows")
    rows = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))["snapshots"]
    if [int(row["epoch"]) for row in rows] != list(SNAPSHOT_EPOCHS):
        raise ValueError("external encoder snapshot matrix is incomplete")
    replayed = []
    saved_environment = json.loads(
        (run_root / "environment.json").read_text(encoding="utf-8")
    )
    for row in rows:
        epoch = int(row["epoch"])
        snapshot = run_root / f"e{epoch}"
        checkpoint_path = snapshot / "checkpoint.pth"
        history_path = snapshot / "history.npz"
        metrics_path = snapshot / "metrics.json"
        if not all(path.is_file() for path in (checkpoint_path, history_path, metrics_path)):
            raise ValueError(f"external encoder e{epoch} artifacts are incomplete")
        if sha256_file(checkpoint_path) != row["checkpoint_sha256"]:
            raise ValueError(f"external encoder e{epoch} checkpoint hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("method") != config.method
            or int(checkpoint.get("walk", -1)) != config.walk
            or int(checkpoint.get("completed_epoch", -1)) != epoch
            or checkpoint.get("dataset_sha256") != sha256_file(dataset_path)
        ):
            raise ValueError(f"external encoder e{epoch} checkpoint contract mismatch")
        expected_history_hash = hashlib.sha256(
            json.dumps(
                checkpoint["history"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if checkpoint.get("loss_history_sha256") != expected_history_hash:
            raise ValueError(f"external encoder e{epoch} history hash mismatch")
        if checkpoint.get("code_revision") != saved_environment.get("git_commit"):
            raise ValueError(f"external encoder e{epoch} code-revision metadata mismatch")
        if set(checkpoint.get("optimizer_states", {})) != {
            "sada_time",
            "sada_frequency",
            "sassl",
        }:
            raise ValueError(f"external encoder e{epoch} optimizer inventory mismatch")
        if checkpoint.get("sampler_state") != {
            "next_epoch": epoch + 1,
            "batch_position": 0,
        }:
            raise ValueError(f"external encoder e{epoch} sampler state mismatch")
        model_config = _model_config_from_payload(checkpoint["model_config"])
        model = build_external_encoder(config, model_config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        checkpoint_scaler = SaURLInputScaler.from_state_dict(checkpoint["input_scaler"])
        _replay_probe(model, checkpoint_scaler, checkpoint["probe"])
        health = _representation_health(model, checkpoint_scaler, train, torch.device("cpu"))
        if health["embedding_std"] < config.collapse_std_threshold:
            raise ValueError(f"external encoder e{epoch} checkpoint is collapsed")
        with np.load(history_path, allow_pickle=False) as history:
            if len(history["epochs"]) != epoch or int(history["epochs"][-1]) != epoch:
                raise ValueError(f"external encoder e{epoch} history mismatch")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("checkpoint_sha256") != row["checkpoint_sha256"]:
            raise ValueError(f"external encoder e{epoch} metrics mismatch")
        replayed.append(epoch)
    complete = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("principal_epoch") != 50:
        raise ValueError("external encoder completion marker is invalid")
    return {
        "valid": True,
        "method": config.method,
        "walk": config.walk,
        "snapshots": replayed,
        "encoder_train_identity_hash": data_manifest["identity_hashes"]["encoder_train"],
        "target_independent": True,
    }
