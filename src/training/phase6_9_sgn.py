"""Leakage-safe, gated, resumable Phase 6.9 SGN-C lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch

from baselines.sgn import GroupInitialization, SGNClassifier, SGNConfig, build_group_initialization
from data_processing.phase5_walks import CLASS_NAMES, sha256_file, validate_phase5_bundle_files
from tasks.phase2_classification.metrics import probabilistic_classification_metrics
from tasks.phase2_classification.protocols import LogitAdjustedCrossEntropy, class_priors
from training.phase5_baselines import load_baseline_data
from training.phase5_encoder import environment_manifest, resolve_device, set_seed, write_json


SCHEMA = "phase6-9-sgn-c-v1"
METHOD = "sgn_c"
SNAPSHOTS = (5, 15, 50)
EXPECTED_ROWS = {1: (37_864, 30_340), 2: (57_521, 13_887)}
REPLAY_RTOL = 2e-5
REPLAY_ATOL = 2e-6


@dataclass(frozen=True)
class SGNTrainingConfig:
    walk: int
    device: str = "cuda"
    seed: int = 0
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOTS
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 4.0
    logit_adjustment_strength: float = 1.0
    grouping_regularizer_beta: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_epochs", tuple(self.snapshot_epochs))
        if self.walk not in (1, 2):
            raise ValueError("SGN-C walk must be 1 or 2")
        actual = (
            self.seed, self.epochs, self.snapshot_epochs, self.batch_size,
            self.learning_rate, self.weight_decay, self.gradient_clip_norm,
            self.logit_adjustment_strength, self.grouping_regularizer_beta,
        )
        expected = (0, 50, SNAPSHOTS, 256, 1e-3, 0.0, 4.0, 1.0, 0.1)
        if actual != expected:
            raise ValueError("SGN-C training recipe differs from the owner-approved freeze")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot_epochs"] = list(self.snapshot_epochs)
        return payload


def phase_root(root: Path) -> Path:
    return Path(root) / "experiments/phase6_9/sgn_classification"


def dataset_path(root: Path, walk: int) -> Path:
    return Path(root) / f"experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz"


def initialization_root(root: Path, walk: int) -> Path:
    return phase_root(root) / f"initialization/walk{walk}"


def run_root(root: Path, walk: int) -> Path:
    return phase_root(root) / f"downstream/classification_h2/walk{walk}/{METHOD}/seed0"


def implementation_fingerprint(root: Path | None = None) -> str:
    repository = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    paths = (
        "src/baselines/sgn/__init__.py", "src/baselines/sgn/config.py",
        "src/baselines/sgn/grouping.py", "src/baselines/sgn/model.py",
        "src/training/phase6_9_sgn.py",
        "scripts_v8/prepare_phase6_9_sgn_initialization.py",
        "scripts_v8/validate_phase6_9_sgn_initialization.py",
        "scripts_v8/bootstrap_phase6_9_sgn.py",
        "scripts_v8/validate_phase6_9_sgn.py",
        "scripts_v8/report_phase6_9_sgn.py",
    )
    digest = sha256()
    for relative in paths:
        digest.update(relative.encode())
        digest.update((repository / relative).read_bytes())
    return digest.hexdigest()


def load_sgn_data(root: Path, walk: int) -> tuple[dict[str, Any], dict[str, Any]]:
    path = dataset_path(root, walk)
    validate_phase5_bundle_files(path)
    data = load_baseline_data(path, "classification_h2")
    source = json.loads(Path(f"{path}.manifest.json").read_text(encoding="utf-8"))
    counts = (len(data["X_train"]), len(data["X_test"]))
    if counts != EXPECTED_ROWS[walk]:
        raise ValueError("SGN-C must preserve every original classification row")
    manifest = {
        "schema_version": SCHEMA,
        "walk": walk,
        "dataset_path": str(path.resolve()),
        "dataset_sha256": sha256_file(path),
        "dataset_manifest_sha256": sha256_file(Path(f"{path}.manifest.json")),
        "train_identity_sha256": source["identity_hashes"]["train"],
        "evaluation_identity_sha256": source["identity_hashes"]["test"],
        "train_label_sha256": source["label_hashes"]["train"],
        "evaluation_label_sha256": source["label_hashes"]["test"],
        "train_rows": counts[0],
        "evaluation_rows": counts[1],
        "input_shape": [64, 5],
        "feature_order": source["feature_columns"],
        "preprocessing": source["preprocessing"],
        "class_names": list(CLASS_NAMES),
        "tau": source["classification"]["tau"],
        "evaluation_used_for_fitting_or_selection": False,
    }
    return data, manifest


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pth", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _initialization_arrays(value: GroupInitialization) -> dict[str, np.ndarray]:
    return {
        "sample_indices": value.sample_indices.astype(np.int64),
        "bdc_matrix": value.bdc_matrix.astype(np.float64),
        "labels": value.labels.astype(np.int64),
        "centroids": value.centroids.astype(np.float64),
        "initial_logits": value.initial_logits.astype(np.float32),
        "fft_bins": value.fft_bins.astype(np.int64),
        "fft_periods": value.fft_periods.astype(np.int64),
    }


def prepare_initialization(root: Path, walk: int) -> dict[str, Any]:
    data, data_manifest = load_sgn_data(root, walk)
    metadata = data["train_metadata"]
    value = build_group_initialization(
        data["X_train"], metadata["condition_ids"], metadata["decision_date_ns"], sample_size=2000, seed=0
    )
    location = initialization_root(root, walk)
    arrays_path = location / "initialization.npz"
    manifest_path = location / "manifest.json"
    manifest = {
        "schema_version": SCHEMA,
        "purpose": "training_only_sgn_group_and_period_initialization",
        "walk": walk,
        "data": data_manifest,
        "config": SGNConfig().to_dict(),
        "initialization": value.to_manifest(),
        "sample_uses_labels": False,
        "sample_uses_evaluation_rows": False,
        "group_label_order": "ascending cluster size, then earliest feature index",
        "group_members": {
            str(group): [str(name) for name, label in zip(data_manifest["feature_order"], value.labels) if int(label) == group]
            for group in range(2)
        },
    }
    if arrays_path.exists() or manifest_path.exists():
        return validate_initialization(root, walk)
    _atomic_savez(arrays_path, _initialization_arrays(value))
    manifest["initialization_npz_sha256"] = sha256_file(arrays_path)
    write_json(manifest_path, manifest)
    return {"valid": True, "walk": walk, "manifest": str(manifest_path), **manifest["initialization"]}


def validate_initialization(root: Path, walk: int) -> dict[str, Any]:
    location = initialization_root(root, walk)
    arrays_path, manifest_path = location / "initialization.npz", location / "manifest.json"
    if not arrays_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing SGN initialization for walk {walk}")
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    data, data_manifest = load_sgn_data(root, walk)
    if existing.get("data") != data_manifest or existing.get("config") != SGNConfig().to_dict():
        raise ValueError("SGN initialization provenance differs")
    metadata = data["train_metadata"]
    expected = build_group_initialization(
        data["X_train"], metadata["condition_ids"], metadata["decision_date_ns"], sample_size=2000, seed=0
    )
    with np.load(arrays_path, allow_pickle=False) as saved:
        expected_arrays = _initialization_arrays(expected)
        if set(saved.files) != set(expected_arrays):
            raise ValueError("SGN initialization array schema differs")
        for name, values in expected_arrays.items():
            if not np.array_equal(saved[name], values):
                raise ValueError(f"SGN initialization replay differs: {name}")
    if existing.get("initialization") != expected.to_manifest() or existing.get("initialization_npz_sha256") != sha256_file(arrays_path):
        raise ValueError("SGN initialization manifest/hash differs")
    return {"valid": True, "walk": walk, "manifest": str(manifest_path), **expected.to_manifest()}


def load_initial_logits(root: Path, walk: int) -> tuple[np.ndarray, dict[str, Any]]:
    validate_initialization(root, walk)
    location = initialization_root(root, walk)
    with np.load(location / "initialization.npz", allow_pickle=False) as saved:
        logits = np.asarray(saved["initial_logits"], dtype=np.float32)
    return logits, json.loads((location / "manifest.json").read_text(encoding="utf-8"))


def _finite_gradients(model: torch.nn.Module) -> None:
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise FloatingPointError("non-finite SGN-C gradients")


def smoke_test_sgn(device: str = "cpu", *, batch_size: int = 2,
                   initial_logits: np.ndarray | None = None,
                   contexts: np.ndarray | None = None,
                   labels: np.ndarray | None = None,
                   priors: np.ndarray | None = None) -> dict[str, Any]:
    resolved = resolve_device(device)
    set_seed(0)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)
    initial = np.asarray(
        initial_logits if initial_logits is not None else [[0.0, 1.0]] * 4 + [[1.0, 0.0]],
        dtype=np.float32,
    )
    model = SGNClassifier(initial).to(resolved)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_priors_value = np.asarray(priors if priors is not None else [0.25, 0.5, 0.25], dtype=np.float32)
    criterion = LogitAdjustedCrossEntropy(class_priors_value).to(resolved)
    if contexts is None:
        values = torch.rand(batch_size, 64, 5, device=resolved)
    else:
        source = np.asarray(contexts, dtype=np.float32)
        if source.ndim != 3 or tuple(source.shape[1:]) != (64, 5) or len(source) < batch_size:
            raise ValueError("SGN-C smoke contexts do not cover the physical batch")
        values = torch.from_numpy(source[:batch_size]).to(resolved)
    if labels is None:
        targets = torch.arange(batch_size, device=resolved) % 3
    else:
        source_labels = np.asarray(labels, dtype=np.int64)
        if len(source_labels) < batch_size:
            raise ValueError("SGN-C smoke labels do not cover the physical batch")
        targets = torch.from_numpy(source_labels[:batch_size]).to(resolved)
    started = time.perf_counter()
    for size in (batch_size, 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, diagnostics = model(values[:size], return_diagnostics=True)
        task = criterion(logits, targets[:size])
        loss = task + 0.1 * diagnostics["similarity_regularizer"]
        loss.backward()
        _finite_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0, error_if_nonfinite=True)
        optimizer.step()
        model.advance_temperature()
    model.eval()
    probe = values[: min(2, batch_size)].detach()
    expected = model(probe).detach()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    replay = SGNClassifier(initial).to(resolved)
    replay.load_state_dict(torch.load(stream, map_location=resolved, weights_only=True))
    replay.eval()
    torch.testing.assert_close(replay(probe), expected, rtol=1e-6, atol=1e-7)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    return {
        "valid": True,
        "batch_size": batch_size,
        "one_row_remainder_passed": True,
        "same_backend_checkpoint_replay": True,
        "uses_real_contexts": contexts is not None,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(resolved)) if resolved.type == "cuda" else 0,
    }


def _capture_rng(generator: torch.Generator) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "sampler": generator.get_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng(payload: Mapping[str, Any], generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    numpy_state = payload["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"], np.asarray(numpy_state["state"], dtype=np.uint32),
        int(numpy_state["position"]), int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(payload["torch_cpu"])
    generator.set_state(payload["sampler"])
    if "torch_cuda" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def _batches(size: int, batch_size: int, generator: torch.Generator) -> tuple[list[torch.Tensor], int]:
    permutation = torch.randperm(size, generator=generator)
    batches = list(permutation.split(batch_size))
    return batches, int(len(batches[-1]))


def _predict(model: SGNClassifier, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            output = model(torch.from_numpy(values[start : start + batch_size]).to(device))
            outputs.append(output.detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _classification_evaluation(logits: np.ndarray, data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels = np.asarray(data["y_test"], dtype=np.int64)
    metrics, arrays = probabilistic_classification_metrics(logits, labels, CLASS_NAMES)
    stable = np.full((len(labels), 3), -30.0, dtype=np.float32)
    stable[:, 1] = 0.0
    stable_metrics, _ = probabilistic_classification_metrics(stable, labels, CLASS_NAMES)
    priors = class_priors(np.asarray(data["y_train"]), 3)
    prior_logits = np.broadcast_to(np.log(priors), (len(labels), 3)).copy()
    prior_metrics, _ = probabilistic_classification_metrics(prior_logits, labels, CLASS_NAMES)
    return {
        "sgn_c": metrics,
        "always_stable_reference": stable_metrics,
        "repeated_training_prior_reference": prior_metrics,
        "class_names": list(CLASS_NAMES),
    }, {**arrays, **data["test_metadata"]}


def _checkpoint(model: SGNClassifier, optimizer: torch.optim.Optimizer, generator: torch.Generator,
                epoch: int, history: list[dict[str, Any]], provenance: Mapping[str, Any],
                probe: torch.Tensor, peak_memory: int) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        probe_output = model(probe.to(next(model.parameters()).device)).cpu()
    return {
        "schema_version": SCHEMA,
        "completed_epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_states": _capture_rng(generator),
        "history": history,
        "provenance": dict(provenance),
        "probe_input": probe,
        "probe_output": probe_output,
        "peak_cuda_memory_bytes": peak_memory,
    }


def _provenance(root: Path, config: SGNTrainingConfig, data_manifest: Mapping[str, Any],
                init_manifest: Mapping[str, Any], admission: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "method": METHOD,
        "config": config.to_dict(),
        "architecture": SGNConfig().to_dict(),
        "implementation_sha256": implementation_fingerprint(root),
        "data": dict(data_manifest),
        "initialization": dict(init_manifest),
        "admission": dict(admission),
        "validation_split": False,
        "early_stopping": False,
        "principal_epoch": 50,
    }


def _validate_admission(root: Path, config: SGNTrainingConfig, data_manifest: Mapping[str, Any],
                        init_manifest: Mapping[str, Any], environment: Mapping[str, Any]) -> dict[str, Any]:
    path = phase_root(root) / "feasibility/cuda_admission.json"
    admission = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": SCHEMA,
        "method": METHOD,
        "implementation_sha256": implementation_fingerprint(root),
        "batch_size": config.batch_size,
        "environment": dict(environment),
    }
    if admission.get("admitted") is not True or any(admission.get(key) != value for key, value in required.items()):
        raise ValueError("SGN-C runtime admission mismatch")
    if admission.get("data", {}).get(str(config.walk)) != dict(data_manifest):
        raise ValueError("SGN-C admitted data differs")
    if admission.get("initialization", {}).get(str(config.walk)) != dict(init_manifest):
        raise ValueError("SGN-C admitted initialization differs")
    return admission


def run_sgn_training(root: Path, config: SGNTrainingConfig, *, stop_after_epoch: int | None = None,
                     _fixture: bool = False) -> dict[str, Any]:
    device = resolve_device(config.device)
    if device.type != "cuda" and not _fixture:
        raise ValueError("full SGN-C training requires admitted CUDA")
    data, data_manifest = load_sgn_data(root, config.walk)
    initial_logits, init_manifest = load_initial_logits(root, config.walk)
    environment = environment_manifest(device)
    admission = {"fixture": True} if _fixture else _validate_admission(root, config, data_manifest, init_manifest, environment)
    provenance = _provenance(root, config, data_manifest, init_manifest, admission)
    location = run_root(root, config.walk)
    if (location / "training_complete.json").is_file():
        return validate_sgn_training(root, config)
    limit = config.epochs if stop_after_epoch is None else int(stop_after_epoch)
    if limit < 0 or limit > config.epochs:
        raise ValueError("invalid SGN-C interruption epoch")
    set_seed(config.seed)
    model = SGNClassifier(initial_logits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    priors = class_priors(np.asarray(data["y_train"]), 3)
    criterion = LogitAdjustedCrossEntropy(priors, config.logit_adjustment_strength).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    probe = torch.from_numpy(np.asarray(data["X_train"][:4], dtype=np.float32).copy())
    history: list[dict[str, Any]] = []
    completed_epoch, peak_memory = 0, 0
    if location.exists():
        resume_path = location / "resume.pth"
        if not resume_path.is_file():
            raise ValueError("occupied SGN-C run has no resumable checkpoint")
        saved = torch.load(resume_path, map_location=device, weights_only=True)
        if saved.get("provenance") != provenance:
            raise ValueError("SGN-C resume provenance mismatch")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        _restore_rng(saved["rng_states"], generator)
        history = saved["history"]
        completed_epoch = int(saved["completed_epoch"])
        peak_memory = int(saved["peak_cuda_memory_bytes"])
    else:
        location.mkdir(parents=True)
        write_json(location / "config.json", config.to_dict())
        write_json(location / "architecture_manifest.json", SGNConfig().to_dict())
        write_json(location / "dataset_manifest.json", data_manifest)
        write_json(location / "initialization_manifest.json", init_manifest)
        write_json(location / "environment.json", environment)
        write_json(location / "imbalance_manifest.json", {
            "sampling": "natural rows", "loss": "logit-adjusted cross-entropy",
            "strength": 1.0, "training_class_counts": np.bincount(data["y_train"], minlength=3).tolist(),
            "training_class_priors": priors.tolist(), "evaluation_distribution_untouched": True,
        })
        _atomic_torch_save(location / "resume.pth", _checkpoint(model, optimizer, generator, 0, history, provenance, probe, 0))
    if [int(row["epoch"]) for row in history] != list(range(1, completed_epoch + 1)):
        raise ValueError("SGN-C resume history is discontinuous")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(completed_epoch + 1, limit + 1):
        started = time.perf_counter()
        model.train()
        totals = {"loss": 0.0, "task": 0.0, "group": 0.0, "entropy": 0.0}
        seen = 0
        batches, remainder = _batches(len(data["X_train"]), config.batch_size, generator)
        for batch in batches:
            indices = batch.numpy()
            values = torch.from_numpy(data["X_train"][indices]).to(device)
            labels = torch.from_numpy(data["y_train"][indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, diagnostics = model(values, return_diagnostics=True)
            task_loss = criterion(logits, labels)
            group_loss = diagnostics["similarity_regularizer"]
            loss = task_loss + config.grouping_regularizer_beta * group_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite SGN-C loss")
            loss.backward()
            _finite_gradients(model)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            model.advance_temperature()
            count = len(indices)
            totals["loss"] += float(loss.detach()) * count
            totals["task"] += float(task_loss.detach()) * count
            totals["group"] += float(group_loss.detach()) * count
            totals["entropy"] += float(diagnostics["assignment_entropy"].detach()) * count
            seen += count
        if seen != len(data["X_train"]):
            raise RuntimeError("SGN-C did not consume every training row")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / seen,
            "task_loss": totals["task"] / seen,
            "grouping_loss": totals["group"] / seen,
            "assignment_entropy": totals["entropy"] / seen,
            "temperature": float(model.assignment.temperature),
            "optimizer_steps": int(model.assignment.training_step),
            "gradient_norm_last_batch": float(gradient_norm),
            "seen_rows": seen,
            "last_batch_size": remainder,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        checkpoint = _checkpoint(model, optimizer, generator, epoch, history, provenance, probe, peak_memory)
        _atomic_torch_save(location / "resume.pth", checkpoint)
        _atomic_savez(location / "history.npz", {key: np.asarray([item[key] for item in history]) for key in history[0]})
        if epoch in SNAPSHOTS:
            _atomic_torch_save(location / f"e{epoch}/checkpoint.pth", checkpoint)
        print(f"walk={config.walk} method={METHOD} epoch={epoch} loss={row['train_loss']:.8f}", flush=True)
    if len(history) < config.epochs:
        return {"complete": False, "walk": config.walk, "completed_epoch": len(history)}
    rows: list[dict[str, Any]] = []
    for epoch in SNAPSHOTS:
        snapshot = location / f"e{epoch}"
        checkpoint_path = snapshot / "checkpoint.pth"
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        evaluated = SGNClassifier(initial_logits).to(device)
        evaluated.load_state_dict(saved["model_state_dict"], strict=True)
        started = time.perf_counter()
        logits = _predict(evaluated, data["X_test"], config.batch_size, device)
        metrics, arrays = _classification_evaluation(logits, data)
        inference_seconds = time.perf_counter() - started
        _atomic_savez(snapshot / "predictions.npz", arrays)
        evaluated.eval()
        assignment = evaluated.assignment.assignment().cpu().numpy()
        metric_row = {
            "schema_version": SCHEMA, "method": METHOD, "walk": config.walk,
            "epoch": epoch, "seed": config.seed, "train_loss": history[epoch - 1]["train_loss"],
            "parameter_count": int(sum(parameter.numel() for parameter in evaluated.parameters())),
            "elapsed_training_seconds": float(sum(item["seconds"] for item in history[:epoch])),
            "inference_seconds": inference_seconds, "peak_cuda_memory_bytes": peak_memory,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(snapshot / "predictions.npz"),
            "hard_assignment": assignment.tolist(),
            "hard_group_ids": assignment.argmax(axis=1).tolist(),
            "temperature": float(evaluated.assignment.temperature),
            **metrics,
        }
        write_json(snapshot / "metrics.json", metric_row)
        rows.append({
            "walk": config.walk, "epoch": epoch,
            "macro_f1": metrics["sgn_c"]["macro_f1"],
            "balanced_accuracy": metrics["sgn_c"]["balanced_accuracy"],
            "accuracy": metrics["sgn_c"]["accuracy"],
            "checkpoint_sha256": metric_row["checkpoint_sha256"],
            "predictions_sha256": metric_row["predictions_sha256"],
        })
    write_json(location / "sweep_metrics.json", {"snapshots": rows, "principal_epoch": 50})
    write_json(location / "training_complete.json", {
        "complete": True, "method": METHOD, "walk": config.walk,
        "snapshots": list(SNAPSHOTS), "principal_epoch": 50,
        "selection_rule": "epoch 50 predeclared; evaluation did not select a checkpoint",
    })
    return validate_sgn_training(root, config)


def validate_sgn_training(root: Path, config: SGNTrainingConfig) -> dict[str, Any]:
    location = run_root(root, config.walk)
    required = (
        "config.json", "architecture_manifest.json", "dataset_manifest.json",
        "initialization_manifest.json", "environment.json", "imbalance_manifest.json",
        "history.npz", "sweep_metrics.json", "training_complete.json",
    )
    missing = [name for name in required if not (location / name).is_file()]
    if missing:
        raise ValueError(f"SGN-C run is incomplete: {missing}")
    data, data_manifest = load_sgn_data(root, config.walk)
    if json.loads((location / "dataset_manifest.json").read_text()) != data_manifest:
        raise ValueError("SGN-C dataset provenance mismatch")
    initial_logits, init_manifest = load_initial_logits(root, config.walk)
    if json.loads((location / "initialization_manifest.json").read_text()) != init_manifest:
        raise ValueError("SGN-C initialization provenance mismatch")
    rows = json.loads((location / "sweep_metrics.json").read_text())["snapshots"]
    if [int(row["epoch"]) for row in rows] != list(SNAPSHOTS):
        raise ValueError("SGN-C snapshot matrix is incomplete")
    for row in rows:
        epoch = int(row["epoch"])
        snapshot = location / f"e{epoch}"
        checkpoint_path, predictions_path, metrics_path = snapshot / "checkpoint.pth", snapshot / "predictions.npz", snapshot / "metrics.json"
        if not all(path.is_file() for path in (checkpoint_path, predictions_path, metrics_path)):
            raise ValueError(f"SGN-C epoch {epoch} artifacts are incomplete")
        metrics = json.loads(metrics_path.read_text())
        if metrics["checkpoint_sha256"] != sha256_file(checkpoint_path) or metrics["predictions_sha256"] != sha256_file(predictions_path):
            raise ValueError(f"SGN-C epoch {epoch} artifact hash mismatch")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved.get("completed_epoch") != epoch or saved.get("provenance", {}).get("data") != data_manifest:
            raise ValueError(f"SGN-C epoch {epoch} checkpoint provenance mismatch")
        model = SGNClassifier(initial_logits)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        replay = _predict(model, data["X_test"], config.batch_size, torch.device("cpu"))
        with np.load(predictions_path, allow_pickle=False) as stored:
            if not np.allclose(replay, stored["logits"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL):
                raise ValueError(f"SGN-C epoch {epoch} CPU prediction replay mismatch")
            for name, expected in data["test_metadata"].items():
                if not np.array_equal(stored[name], expected):
                    raise ValueError(f"SGN-C epoch {epoch} identity mismatch: {name}")
            recomputed, _ = probabilistic_classification_metrics(replay, data["y_test"], CLASS_NAMES)
            if abs(float(recomputed["macro_f1"]) - float(metrics["sgn_c"]["macro_f1"])) > 1e-12:
                raise ValueError(f"SGN-C epoch {epoch} metric replay mismatch")
    complete = json.loads((location / "training_complete.json").read_text())
    if complete.get("complete") is not True or complete.get("snapshots") != list(SNAPSHOTS):
        raise ValueError("SGN-C completion marker differs")
    return {
        "valid": True, "method": METHOD, "walk": config.walk,
        "snapshots": list(SNAPSHOTS), "principal_epoch": 50,
        "epoch50_checkpoint_sha256": rows[-1]["checkpoint_sha256"],
        "epoch50_predictions_sha256": rows[-1]["predictions_sha256"],
        "cpu_replay_rtol": REPLAY_RTOL, "cpu_replay_atol": REPLAY_ATOL,
    }


def freeze_training_matrix(root: Path) -> dict[str, Any]:
    entries, data_manifests, init_manifests = [], {}, {}
    for walk in (1, 2):
        _, data_manifest = load_sgn_data(root, walk)
        _, init_manifest = load_initial_logits(root, walk)
        config = SGNTrainingConfig(walk=walk)
        entries.append({"walk": walk, "config": config.to_dict(), "run_root": str(run_root(root, walk).resolve())})
        data_manifests[str(walk)] = data_manifest
        init_manifests[str(walk)] = init_manifest
    payload = {
        "schema_version": SCHEMA, "method": METHOD,
        "implementation_sha256": implementation_fingerprint(root),
        "architecture": SGNConfig().to_dict(), "entry_count": 2,
        "entries": entries, "data": data_manifests, "initialization": init_manifests,
        "validation_split": False, "early_stopping": False,
        "default_action_trains_models": False, "principal_epoch": 50,
    }
    path = phase_root(root) / "manifests/training_seed0.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        if any(run_root(root, walk).exists() for walk in (1, 2)):
            raise ValueError("existing SGN-C matrix differs after a run root was created")
        write_json(path, payload)
    elif not path.exists():
        write_json(path, payload)
    return payload
