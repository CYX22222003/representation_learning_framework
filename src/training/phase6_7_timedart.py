"""Walk-specific TimeDART pretraining and replay for Phase 6.7."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from baselines.timedart import TimeDARTConfig, TimeDARTEncoder, TimeDARTPretrainer
from data_processing.phase5_walks import sha256_file
from data_processing.phase6_7_timedart_data import (
    METHOD,
    load_timedart_encoder_population,
)
from training.phase5_encoder import environment_manifest, resolve_device, set_seed, write_json


SNAPSHOT_EPOCHS = (5, 15, 50)
SCHEMA_VERSION = "phase6-7-timedart-encoder-v1"
FROZEN_ENCODER_SCHEMA_VERSION = "phase6-7-timedart-frozen-encoder-v1"
CROSS_DEVICE_RELATIVE_L2_TOLERANCE = 5e-4
CROSS_DEVICE_COSINE_TOLERANCE = 0.999999


@dataclass(frozen=True)
class Phase67TimeDARTConfig:
    walk: int
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    scheduler_gamma: float = 0.95
    device: str = "cuda"

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_epochs", tuple(self.snapshot_epochs))
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("TimeDART requires 50 epochs and 5/15/50 snapshots")
        if self.seed != 0 or self.batch_size != 16:
            raise ValueError("TimeDART primary training is frozen to seed 0 and batch 16")
        if self.learning_rate != 1e-3 or self.weight_decay != 0.0:
            raise ValueError("TimeDART optimizer contract changed")
        if self.scheduler_gamma != 0.95:
            raise ValueError("TimeDART scheduler contract changed")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot_epochs"] = list(self.snapshot_epochs)
        return payload


def _config_from_payload(payload: Mapping[str, Any], device: str) -> Phase67TimeDARTConfig:
    values = dict(payload)
    values["snapshot_epochs"] = tuple(values["snapshot_epochs"])
    values["device"] = device
    return Phase67TimeDARTConfig(**values)


def _model_config_from_payload(payload: Mapping[str, Any]) -> TimeDARTConfig:
    return TimeDARTConfig(**dict(payload))


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


def _capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy[0],
            "keys": torch.from_numpy(numpy[1].copy()),
            "position": int(numpy[2]),
            "has_gauss": int(numpy[3]),
            "cached_gaussian": float(numpy[4]),
        },
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
            raise RuntimeError("resume checkpoint requires CUDA RNG state")
        torch.cuda.set_rng_state_all(cuda_states)
    generator.set_state(state["sampler_generator"].cpu())


def drop_last_batch_indices(
    row_count: int, batch_size: int, generator: torch.Generator
) -> tuple[list[torch.Tensor], int]:
    if row_count < batch_size or batch_size <= 0:
        raise ValueError("TimeDART requires at least one complete positive-size batch")
    permutation = torch.randperm(row_count, generator=generator)
    used = row_count - row_count % batch_size
    batches = list(permutation[:used].split(batch_size))
    return batches, row_count - used


def _assert_finite_gradients(model: torch.nn.Module) -> None:
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise RuntimeError("TimeDART update produced no gradients")
    if any(not torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("TimeDART update produced non-finite gradients")


@torch.no_grad()
def _probe_payload(
    model: TimeDARTPretrainer, raw_input: torch.Tensor
) -> dict[str, torch.Tensor]:
    model.eval()
    device = next(model.parameters()).device
    raw = raw_input.to(device)
    rows = raw.shape[0] * model.config.channels
    steps = (
        torch.arange(rows * model.config.patch_count, device=device)
        .reshape(rows, model.config.patch_count)
        .remainder(model.config.diffusion_steps)
    )
    noise_generator = torch.Generator(device=device).manual_seed(731)
    noise = torch.randn(
        rows,
        model.config.patch_count,
        model.config.patch_len,
        generator=noise_generator,
        device=device,
    )
    output = model(raw, timesteps=steps, noise=noise)
    features = model.encoder.extract(raw)
    return {
        "raw_input": raw.cpu(),
        "timesteps": steps.cpu(),
        "noise": noise.cpu(),
        "reconstruction": output.reconstruction.cpu(),
        "features": features.cpu(),
    }


def _checkpoint_payload(
    *,
    model: TimeDARTPretrainer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ExponentialLR,
    config: Phase67TimeDARTConfig,
    dataset_path: Path,
    data_record: Mapping[str, Any],
    completed_epoch: int,
    history: list[Mapping[str, float]],
    generator: torch.Generator,
    raw_probe: torch.Tensor,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "6.7",
        "method": METHOD,
        "walk": config.walk,
        "completed_epoch": completed_epoch,
        "config": config.to_dict(),
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "dataset_sha256": sha256_file(dataset_path),
        "encoder_train_identity_hash": data_record["identity_hash"],
        "history": [dict(row) for row in history],
        "rng_states": _capture_rng_state(generator),
        "sampler_state": {"next_epoch": completed_epoch + 1, "batch_position": 0},
        "probe": _probe_payload(model, raw_probe),
        "code_revision": environment.get("git_commit"),
        "device_metadata": dict(environment),
    }


def smoke_test_timedart() -> dict[str, Any]:
    set_seed(0)
    config = TimeDARTConfig()
    model = TimeDARTPretrainer(config)
    inputs = torch.randn(2, 64, 5)
    output = model(inputs)
    output.loss.backward()
    _assert_finite_gradients(model)
    features = model.frozen_encoder()(inputs)
    return {
        "loss_finite": bool(torch.isfinite(output.loss)),
        "reconstruction_shape": list(output.reconstruction.shape),
        "feature_shape": list(features.shape),
    }


def resource_smoke_timedart(
    dataset_path: Path, *, walk: int, device: str = "cuda"
) -> dict[str, Any]:
    train, record = load_timedart_encoder_population(dataset_path, walk)
    resolved = resolve_device(device)
    set_seed(0)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)
        torch.cuda.synchronize(resolved)
    model = TimeDARTPretrainer().to(resolved)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.from_numpy(train[:16]).to(resolved)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)
    output.loss.backward()
    _assert_finite_gradients(model)
    optimizer.step()
    with torch.no_grad():
        features = model.encoder.extract(inputs)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    elapsed = time.perf_counter() - started
    return {
        "walk": walk,
        "dataset_sha256": record["dataset_sha256"],
        "batch_size": 16,
        "loss": float(output.loss.detach()),
        "loss_finite": bool(torch.isfinite(output.loss)),
        "gradients_finite": True,
        "features_finite": bool(torch.isfinite(features).all()),
        "feature_shape": list(features.shape),
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(resolved)) if resolved.type == "cuda" else 0
        ),
        "pretraining_parameters": sum(p.numel() for p in model.parameters()),
        "retained_encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "environment": environment_manifest(resolved),
    }


def validate_admission_manifest(
    admission_path: Path, dataset_path: Path, walk: int
) -> dict[str, Any]:
    if not Path(admission_path).is_file():
        raise ValueError("TimeDART CUDA admission manifest is missing")
    payload = json.loads(Path(admission_path).read_text(encoding="utf-8"))
    if payload.get("method") != METHOD or payload.get("admitted") is not True:
        raise ValueError("TimeDART CUDA admission was not granted")
    entry = next(
        (item for item in payload.get("walks", []) if int(item.get("walk", -1)) == walk),
        None,
    )
    if entry is None or entry.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("TimeDART CUDA admission data identity mismatch")
    return payload


def _history_arrays(history: list[Mapping[str, float]]) -> dict[str, np.ndarray]:
    names = sorted({name for row in history for name in row})
    return {
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        **{
            name: np.asarray([row[name] for row in history], dtype=np.float64)
            for name in names
        },
    }


def run_timedart_pretraining(
    dataset_path: Path,
    run_root: Path,
    config: Phase67TimeDARTConfig,
) -> dict[str, Any]:
    """Train or resume one target-free TimeDART trajectory."""

    train, data_record = load_timedart_encoder_population(dataset_path, config.walk)
    device = resolve_device(config.device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    complete_path = run_root / "training_complete.json"
    if complete_path.is_file():
        raise FileExistsError(f"refusing to overwrite completed TimeDART run: {run_root}")

    if not run_root.exists():
        set_seed(config.seed)
        model = TimeDARTPretrainer().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.scheduler_gamma
        )
        run_root.mkdir(parents=True)
        # Normalize version-like string subclasses (notably TorchVersion) to
        # plain JSON values so the checkpoint remains weights_only-loadable.
        environment = json.loads(json.dumps(environment_manifest(device)))
        write_json(run_root / "config.json", config.to_dict())
        write_json(run_root / "model_config.json", model.config.to_dict())
        write_json(run_root / "environment.json", environment)
        write_json(
            run_root / "dataset_manifest.json",
            {
                "dataset_path": str(dataset_path.resolve()),
                **data_record,
                "downstream_targets_loaded": False,
                "evaluation_values_loaded": False,
            },
        )
        write_json(
            run_root / "architecture_manifest.json",
            {
                "method": METHOD,
                "pretraining_parameters": sum(p.numel() for p in model.parameters()),
                "retained_encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
                "representation_width": model.config.representation_width,
                "decoder_discarded_after_pretraining": True,
            },
        )
        write_json(
            run_root / "training_contract.json",
            {
                "optimizer": "Adam",
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "scheduler": "ExponentialLR",
                "scheduler_gamma": config.scheduler_gamma,
                "batch_size": config.batch_size,
                "drop_last": True,
                "objective": "mean squared clean-input reconstruction error",
                "validation_split": False,
                "evaluation_used_for_selection": False,
                "principal_epoch": 50,
            },
        )
        start_epoch = 1
        history: list[dict[str, float]] = []
        raw_probe = torch.from_numpy(train[:4].copy())
        _atomic_torch_save(
            run_root / "resume.pth",
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                dataset_path=dataset_path,
                data_record=data_record,
                completed_epoch=0,
                history=history,
                generator=generator,
                raw_probe=raw_probe,
                environment=environment,
            ),
        )
    else:
        resume_path = run_root / "resume.pth"
        if not resume_path.is_file():
            raise ValueError("incomplete TimeDART run has no resume checkpoint")
        resume = torch.load(resume_path, map_location=device, weights_only=True)
        if _config_from_payload(resume["config"], config.device) != config:
            raise ValueError("TimeDART resume configuration drift")
        if resume.get("dataset_sha256") != sha256_file(dataset_path):
            raise ValueError("TimeDART resume dataset drift")
        model = TimeDARTPretrainer(_model_config_from_payload(resume["model_config"])).to(device)
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.scheduler_gamma
        )
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        _restore_rng_state(resume["rng_states"], generator)
        start_epoch = int(resume["completed_epoch"]) + 1
        history = [dict(row) for row in resume["history"]]
        raw_probe = resume["probe"]["raw_input"].cpu()
        environment = json.loads((run_root / "environment.json").read_text(encoding="utf-8"))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    snapshots = []
    for epoch in config.snapshot_epochs:
        metrics_path = run_root / f"e{epoch}" / "metrics.json"
        if epoch < start_epoch and metrics_path.is_file():
            snapshots.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    for epoch in range(start_epoch, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        batches, dropped = drop_last_batch_indices(len(train), config.batch_size, generator)
        loss_total = 0.0
        rows_seen = 0
        for indices in batches:
            inputs = torch.from_numpy(train[indices.numpy()]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            output.loss.backward()
            _assert_finite_gradients(model)
            optimizer.step()
            loss_total += float(output.loss.detach()) * len(inputs)
            rows_seen += len(inputs)
        scheduler.step()
        row = {
            "train_loss": loss_total / rows_seen,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "rows_seen": float(rows_seen),
            "dropped_remainder": float(dropped),
            "epoch_seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError("non-finite TimeDART epoch diagnostic")
        history.append(row)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            dataset_path=dataset_path,
            data_record=data_record,
            completed_epoch=epoch,
            history=history,
            generator=generator,
            raw_probe=raw_probe,
            environment=environment,
        )
        print(
            f"walk={config.walk} method={METHOD} epoch={epoch} "
            f"loss={row['train_loss']:.8f} lr={row['learning_rate']:.8g} "
            f"dropped={dropped} seconds={row['epoch_seconds']:.2f}",
            flush=True,
        )
        if epoch in config.snapshot_epochs:
            snapshot = run_root / f"e{epoch}"
            # A prior interruption may have left this directory partial. The
            # files themselves are replaced atomically, and resume.pth is
            # advanced only after the complete snapshot is materialized.
            snapshot.mkdir(exist_ok=True)
            checkpoint_path = snapshot / "checkpoint.pth"
            encoder_path = snapshot / "encoder.pth"
            _atomic_torch_save(checkpoint_path, payload)
            _atomic_torch_save(
                encoder_path,
                {
                    "schema_version": FROZEN_ENCODER_SCHEMA_VERSION,
                    "phase": "6.7",
                    "method": METHOD,
                    "walk": config.walk,
                    "completed_epoch": epoch,
                    "model_config": model.config.to_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "dataset_sha256": sha256_file(dataset_path),
                    "encoder_train_identity_hash": data_record["identity_hash"],
                },
            )
            _atomic_savez(snapshot / "history.npz", _history_arrays(history))
            metrics = {
                "epoch": epoch,
                **row,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "encoder_sha256": sha256_file(encoder_path),
                "elapsed_training_seconds": float(sum(item["epoch_seconds"] for item in history)),
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                ),
            }
            write_json(snapshot / "metrics.json", metrics)
            snapshots.append(metrics)
        _atomic_torch_save(run_root / "resume.pth", payload)
    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    write_json(
        complete_path,
        {
            "complete": True,
            "snapshots": list(config.snapshot_epochs),
            "principal_epoch": 50,
            "selection_rule": "epoch 50 predeclared; no validation or evaluation selection",
        },
    )
    try:
        return validate_timedart_pretraining(dataset_path, run_root)
    except Exception as exc:
        warning = {
            "valid": False,
            "status": "warning",
            "continued": True,
            "method": METHOD,
            "walk": config.walk,
            "message": str(exc),
        }
        write_json(run_root / "replay_validation.json", warning)
        return warning


def _replay_diagnostics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        return {"accepted": False, "strict": False, "relative_l2": float("inf"), "cosine": float("nan")}
    strict = torch.equal(actual, expected)
    delta = (actual - expected).reshape(-1).double()
    reference = expected.reshape(-1).double()
    denominator = max(float(reference.norm()), torch.finfo(torch.float64).eps)
    relative_l2 = float(delta.norm()) / denominator
    cosine = float(torch.nn.functional.cosine_similarity(
        actual.reshape(1, -1).double(), expected.reshape(1, -1).double(), dim=1
    )[0])
    return {
        "strict": strict,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "accepted": bool(
            strict
            or (
                relative_l2 <= CROSS_DEVICE_RELATIVE_L2_TOLERANCE
                and cosine >= CROSS_DEVICE_COSINE_TOLERANCE
            )
        ),
    }


@torch.no_grad()
def _replay_probe(model: TimeDARTPretrainer, probe: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    model.eval()
    actual = model(
        probe["raw_input"], timesteps=probe["timesteps"], noise=probe["noise"]
    ).reconstruction
    features = model.encoder.extract(probe["raw_input"])
    return {
        "reconstruction": _replay_diagnostics(actual.cpu(), probe["reconstruction"].cpu()),
        "features": _replay_diagnostics(features.cpu(), probe["features"].cpu()),
    }


def validate_timedart_pretraining(dataset_path: Path, run_root: Path) -> dict[str, Any]:
    required = {
        "config.json",
        "model_config.json",
        "environment.json",
        "dataset_manifest.json",
        "architecture_manifest.json",
        "training_contract.json",
        "sweep_metrics.json",
        "training_complete.json",
    }
    missing = sorted(name for name in required if not (run_root / name).is_file())
    if missing:
        raise ValueError(f"TimeDART run is missing artifacts: {missing}")
    config = _config_from_payload(
        json.loads((run_root / "config.json").read_text(encoding="utf-8")), "cpu"
    )
    _, data_record = load_timedart_encoder_population(dataset_path, config.walk)
    saved_data = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    for key in ("dataset_sha256", "identity_hash", "sequence_hash"):
        if saved_data.get(key) != data_record[key]:
            raise ValueError(f"TimeDART encoder data provenance mismatch: {key}")
    if saved_data.get("downstream_targets_loaded") or saved_data.get("evaluation_values_loaded"):
        raise ValueError("TimeDART target-independent training invariant failed")
    rows = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))["snapshots"]
    if [int(row["epoch"]) for row in rows] != list(SNAPSHOT_EPOCHS):
        raise ValueError("TimeDART snapshot matrix is incomplete")
    replay: dict[str, Any] = {"warnings": [], "snapshots": {}}
    for row in rows:
        epoch = int(row["epoch"])
        snapshot = run_root / f"e{epoch}"
        checkpoint_path = snapshot / "checkpoint.pth"
        encoder_path = snapshot / "encoder.pth"
        history_path = snapshot / "history.npz"
        metrics_path = snapshot / "metrics.json"
        if not all(path.is_file() for path in (checkpoint_path, encoder_path, history_path, metrics_path)):
            raise ValueError(f"TimeDART e{epoch} snapshot is incomplete")
        if sha256_file(checkpoint_path) != row["checkpoint_sha256"]:
            raise ValueError(f"TimeDART e{epoch} checkpoint hash mismatch")
        if sha256_file(encoder_path) != row["encoder_sha256"]:
            raise ValueError(f"TimeDART e{epoch} frozen encoder hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("method") != METHOD
            or int(checkpoint.get("walk", -1)) != config.walk
            or int(checkpoint.get("completed_epoch", -1)) != epoch
            or checkpoint.get("dataset_sha256") != sha256_file(dataset_path)
        ):
            raise ValueError(f"TimeDART e{epoch} checkpoint contract mismatch")
        model = TimeDARTPretrainer(_model_config_from_payload(checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        frozen = torch.load(encoder_path, map_location="cpu", weights_only=True)
        if (
            frozen.get("schema_version") != FROZEN_ENCODER_SCHEMA_VERSION
            or frozen.get("dataset_sha256") != sha256_file(dataset_path)
            or frozen.get("encoder_train_identity_hash") != data_record["identity_hash"]
        ):
            raise ValueError(f"TimeDART e{epoch} frozen encoder contract mismatch")
        if set(frozen["encoder_state_dict"]) != set(checkpoint["encoder_state_dict"]):
            raise ValueError(f"TimeDART e{epoch} frozen encoder fields mismatch")
        for name, value in frozen["encoder_state_dict"].items():
            if not torch.equal(value, checkpoint["encoder_state_dict"][name]):
                raise ValueError(f"TimeDART e{epoch} frozen encoder state mismatch: {name}")
        diagnostics = _replay_probe(model, checkpoint["probe"])
        replay["snapshots"][f"e{epoch}"] = diagnostics
        for name, result in diagnostics.items():
            if not result["accepted"]:
                raise ValueError(f"TimeDART e{epoch} material replay drift: {name}")
            if not result["strict"]:
                replay["warnings"].append(
                    {
                        "epoch": epoch,
                        "tensor": name,
                        "severity": "warning",
                        **result,
                    }
                )
        with np.load(history_path, allow_pickle=False) as history:
            if len(history["epochs"]) != epoch or int(history["epochs"][-1]) != epoch:
                raise ValueError(f"TimeDART e{epoch} history mismatch")
    complete = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("principal_epoch") != 50:
        raise ValueError("TimeDART completion marker is invalid")
    replay["valid"] = True
    replay["status"] = "warning" if replay["warnings"] else "pass"
    write_json(run_root / "replay_validation.json", replay)
    return {
        "valid": True,
        "status": replay["status"],
        "warnings": replay["warnings"],
        "method": METHOD,
        "walk": config.walk,
        "snapshots": list(SNAPSHOT_EPOCHS),
        "target_independent": True,
    }


def load_timedart_encoder_checkpoint(
    checkpoint_path: Path, *, walk: int, device: str | torch.device = "cpu"
) -> tuple[TimeDARTEncoder, Mapping[str, Any]]:
    resolved = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved, weights_only=True)
    if (
        checkpoint.get("schema_version") != FROZEN_ENCODER_SCHEMA_VERSION
        or
        checkpoint.get("method") != METHOD
        or int(checkpoint.get("walk", -1)) != walk
        or int(checkpoint.get("completed_epoch", -1)) != 50
    ):
        raise ValueError("only the matching epoch-50 TimeDART checkpoint may extract features")
    config = _model_config_from_payload(checkpoint["model_config"])
    encoder = TimeDARTEncoder(config).to(resolved)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    encoder.eval().requires_grad_(False)
    return encoder, checkpoint
