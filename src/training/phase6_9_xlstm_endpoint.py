"""Exact-row, endpoint-only XM-C8 lifecycle; legacy XM-MV8 is untouched."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from baselines.xlstm_mixer.config import SOURCE_CONTRACT
from baselines.xlstm_mixer.endpoint import METHOD, METHOD_ID, XLSTMMixerEndpoint, architecture_manifest
from data_processing.phase5_walks import sha256_file
from features.phase5_features import IDENTITY_FIELDS
from training.phase5_absolute_price import price_breakdowns, relative_skill
from training.phase5_baselines import load_baseline_data
from training.phase5_downstream import predict, regression_metrics
from training.phase5_encoder import resolve_device, set_seed, write_json
from training.phase6_9_xlstm_mixer import (
    _assert_finite_gradients, _atomic_savez, _atomic_torch_save,
    _capture_rng_state, _restore_rng_state, _probe_output,
    runtime_environment_payload, shuffled_batches, xlstm_dependency_manifest,
)

SCHEMA = "phase6-9-xlstm-endpoint-v1"
SNAPSHOTS = (5, 15, 50)
EXPECTED_ROWS = {1: (36_773, 29_834), 2: (56_652, 13_506)}


@dataclass(frozen=True)
class EndpointTrainingConfig:
    walk: int
    device: str = "cuda"
    backend: str = "vanilla"
    seed: int = 0
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.walk not in (1, 2) or self.backend not in ("vanilla", "cuda"):
            raise ValueError("invalid XM-C8 walk/backend")
        if (self.seed, self.epochs, self.batch_size, self.learning_rate,
                self.weight_decay, self.gradient_clip_norm) != (0, 50, 512, 1e-4, 0.0, 1.0):
            raise ValueError("XM-C8 training recipe differs from the endpoint freeze")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def phase_root(root: Path) -> Path:
    return root / "experiments/phase6_9/xlstm_mixer_endpoint"


def provenance_path(path: Path) -> str:
    """The Windows E: mount is case-insensitive; preserve other filesystems."""
    value = str(path.resolve())
    return value.casefold() if value.casefold().startswith("/mnt/e/") else value


def dependency_identity() -> dict[str, Any]:
    payload = xlstm_dependency_manifest()
    # Record the same installed module independent of WSL path spelling.
    payload["module_path"] = provenance_path(Path(payload["module_path"]))
    return payload


def original_data_path(root: Path, walk: int) -> Path:
    return root / f"experiments/phase5/downstream_addons/shared/h8/data/walk{walk}/market_1h_seq64_h8.npz"


def run_root(root: Path, walk: int) -> Path:
    return phase_root(root) / f"downstream/absolute_price_h8/walk{walk}/{METHOD}/seed0"


def implementation_fingerprint(root: Path | None = None) -> str:
    repository = root or Path(__file__).resolve().parents[2]
    # Include the actually imported implementation, including legacy helpers.
    paths = (
        "src/baselines/xlstm_mixer/endpoint.py", "src/baselines/xlstm_mixer/config.py",
        "src/baselines/xlstm_mixer/normalization.py", "src/baselines/xlstm_mixer/backend.py",
        "src/training/phase6_9_xlstm_endpoint.py", "src/training/phase6_9_xlstm_mixer.py",
        "src/training/phase5_baselines.py", "src/training/phase5_absolute_price.py",
        "src/training/phase5_downstream.py", "src/training/phase5_encoder.py",
        "src/data_processing/phase5_walks.py", "src/features/phase5_features.py",
        "scripts_v8/run_phase6_9_xlstm_endpoint.py",
    )
    digest = sha256()
    for relative in paths:
        digest.update(relative.encode())
        digest.update((repository / relative).read_bytes())
    return digest.hexdigest()


def load_endpoint_data(root: Path, walk: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read original price data verbatim; never construct/filter a future path."""
    path = original_data_path(root, walk)
    data = load_baseline_data(path, "absolute_price_h8")
    source = json.loads(Path(f"{path}.manifest.json").read_text())
    if source.get("walk") != walk or source.get("horizon_bars") != 8:
        raise ValueError("original endpoint bundle walk/horizon mismatch")
    counts = (len(data["y_train"]), len(data["y_test"]))
    if counts != EXPECTED_ROWS[walk]:
        raise ValueError("XM-C8 must preserve all original price rows")
    for split in ("train", "test"):
        metadata = data[f"{split}_metadata"]
        if not np.array_equal(metadata["target_close"], data[f"y_{split}"]):
            raise ValueError("original endpoint target identity mismatch")
        if not np.array_equal(
            metadata["target_date_ns"], metadata["decision_date_ns"] + 8 * 3_600_000_000_000
        ):
            raise ValueError("target is not exactly eight hours ahead")
    manifest = {
        "schema_version": SCHEMA, "method_id": METHOD_ID, "walk": walk,
        "dataset_sha256": sha256_file(path),
        "data_manifest_sha256": sha256_file(Path(f"{path}.manifest.json")),
        "identity_hashes": source["identity_hashes"],
        "train_rows": counts[0], "test_rows": counts[1],
        "target": "close[t+8h]", "loss": "mse_mean_raw_probability",
        "auxiliary_supervision": False, "row_filtering": False,
        "source_path": provenance_path(path),
    }
    return data, manifest


def audit_existing_controls(root: Path, walk: int, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Prove source/recipe/prediction matches before reusing old control rows."""
    locations = {
        "h0_d0": root / f"experiments/phase5/downstream_addons/tasks/absolute_price_h8/walk{walk}/seed0",
        "raw_lstm": root / f"experiments/phase5/baselines/tasks/absolute_price_h8/raw_ohlcv_lstm/walk{walk}/seed0",
    }
    with np.load(original_data_path(root, walk), allow_pickle=False) as source:
        result = {}
        for name, location in locations.items():
            provenance = json.loads((location / "dataset_manifest.json").read_text())
            config = json.loads((location / "config.json").read_text())
            if provenance["dataset_sha256"] != manifest["dataset_sha256"]:
                raise ValueError(f"{name}: original dataset hash mismatch")
            if (provenance["train_rows"], provenance["test_rows"]) != EXPECTED_ROWS[walk]:
                raise ValueError(f"{name}: training/evaluation row count mismatch")
            for field, expected in (("seed", 0), ("epochs", 50), ("batch_size", 512),
                                    ("learning_rate", 1e-4), ("weight_decay", 0.0)):
                if config.get(field) != expected:
                    raise ValueError(f"{name}: {field} differs from endpoint recipe")
            if config.get("snapshot_epochs") != list(SNAPSHOTS):
                raise ValueError(f"{name}: snapshot recipe differs")
            checkpoint = torch.load(location / "e50/checkpoint.pth", map_location="cpu", weights_only=True)
            if checkpoint.get("dataset_sha256") != manifest["dataset_sha256"]:
                raise ValueError(f"{name}: checkpoint training source mismatch")
            if checkpoint.get("completed_epoch") != 50:
                raise ValueError(f"{name}: checkpoint epoch mismatch")
            predictions_path = location / "e50/predictions.npz"
            metrics = json.loads((location / "e50/metrics.json").read_text())
            if metrics.get("predictions_sha256") != sha256_file(predictions_path):
                raise ValueError(f"{name}: prediction hash mismatch")
            with np.load(predictions_path, allow_pickle=False) as saved:
                for field in (*IDENTITY_FIELDS, "target_date_ns", "current_close"):
                    if not np.array_equal(saved[field], source[f"test_{field}"]):
                        raise ValueError(f"{name}: evaluation identity mismatch: {field}")
                if not np.array_equal(saved["target_future_price"], source["test_target_close"]):
                    raise ValueError(f"{name}: evaluation targets differ")
            result[name] = {"valid": True, "run_root": provenance_path(location),
                            "dataset_sha256": provenance["dataset_sha256"],
                            "predictions_sha256": sha256_file(predictions_path),
                            "checkpoint_sha256": sha256_file(location / "e50/checkpoint.pth")}
    return result


def smoke_test(device: str = "cpu", backend: str = "vanilla", *, batch_size: int = 2) -> dict[str, Any]:
    """Synthetic-only backward/update/reload; no experiment trajectory."""
    resolved = resolve_device(device)
    set_seed(0)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)
    started = time.perf_counter()
    model = XLSTMMixerEndpoint(backend=backend).to(resolved)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    contexts = torch.rand(batch_size, 64, 5, device=resolved)
    target = torch.rand(batch_size, 1, device=resolved)
    for size in (batch_size, 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(contexts[:size])
        loss = model.endpoint_mse_loss(output, target[:size])
        loss.backward()
        _assert_finite_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    probe = contexts[:2].detach().cpu()
    expected = _probe_output(model, probe)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    replay = XLSTMMixerEndpoint(backend=backend).to(resolved)
    replay.load_state_dict(torch.load(stream, map_location=resolved, weights_only=True))
    torch.testing.assert_close(_probe_output(replay, probe), expected, rtol=1e-6, atol=1e-7)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    return {"valid": True, "backend": backend, "batch_size": batch_size,
            "one_row_remainder_passed": True, "same_backend_checkpoint_replay": True,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(resolved)) if resolved.type == "cuda" else 0}


def _provenance(manifest: Mapping[str, Any], config: EndpointTrainingConfig,
                admission: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA, "method_id": METHOD_ID,
            "config": config.to_dict(), "architecture": architecture_manifest(),
            "implementation_sha256": implementation_fingerprint(),
            "data": dict(manifest), "source_contract": SOURCE_CONTRACT.to_dict(),
            "dependency": dependency_identity(), "admission": dict(admission)}


def evaluate_endpoint(prediction: np.ndarray, data: Mapping[str, Any]) -> tuple[dict, dict]:
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(data["y_test"], dtype=np.float64)
    metadata = data["test_metadata"]
    if predicted.shape != target.shape or not np.isfinite(predicted).all():
        raise ValueError("invalid endpoint predictions")
    current = np.asarray(metadata["current_close"], dtype=np.float64)
    delta, actual_delta = predicted - current, target - current
    price, contracts = price_breakdowns(predicted, target, metadata)
    persistence, _ = price_breakdowns(current, target, metadata)
    rank_ic = []
    times = np.asarray(metadata["decision_date_ns"])
    for timestamp in np.unique(times):
        selected = times == timestamp
        if np.count_nonzero(selected) < 5:
            continue
        value = regression_metrics(delta[selected], actual_delta[selected])["spearman"]
        if math.isfinite(value):
            rank_ic.append(value)
    reversal = -(data["X_test"][:, -1, 3].astype(np.float64) - data["X_test"][:, -2, 3].astype(np.float64))
    nonzero = actual_delta != 0
    metrics = {"price": price, "per_contract": contracts, "persistence_reference": persistence,
               "persistence_relative_skill": relative_skill(price["overall"], persistence["overall"]),
               "implied_movement": regression_metrics(delta, actual_delta),
               "last_hour_reversal": regression_metrics(reversal, actual_delta),
               "nonzero_movement_sign_agreement": float(np.mean(np.sign(delta[nonzero]) == np.sign(actual_delta[nonzero]))) if nonzero.any() else float("nan"),
               "timestamp_cross_sectional_rank_ic": {"minimum_contract_count": 5,
                    "timestamp_count": len(rank_ic), "mean": float(np.mean(rank_ic)) if rank_ic else float("nan")},
               "outside_probability_range_fraction": float(np.mean((predicted < 0) | (predicted > 1))),
               "supervision": "only close[t+8h]; raw-probability MSE; no auxiliary labels"}
    arrays = {"prediction_future_price": predicted, "target_future_price": target,
              "prediction_delta_h8": delta, "target_delta_h8": actual_delta,
              "last_hour_reversal": reversal, **metadata}
    return metrics, arrays


def _checkpoint(model, optimizer, generator, epoch, history, provenance, probe, peak) -> dict:
    return {"provenance": provenance, "completed_epoch": epoch,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "rng_states": _capture_rng_state(generator), "history": history,
            "probe_input": probe, "probe_output": _probe_output(model, probe),
            "peak_cuda_memory_bytes": peak}


def _history_arrays(history: list[dict]) -> dict[str, np.ndarray]:
    return {name: np.asarray([row[name] for row in history])
            for name in ("epoch", "train_loss", "seconds", "seen_rows", "last_batch_size")}


def run_endpoint_training(root: Path, config: EndpointTrainingConfig, *,
                          stop_after_epoch: int | None = None, _fixture: bool = False) -> dict:
    device = resolve_device(config.device)
    if device.type != "cuda" and not _fixture:
        raise ValueError("XM-C8 full training requires selected-backend CUDA admission")
    data, manifest = load_endpoint_data(root, config.walk)
    location = run_root(root, config.walk)
    admission_path = phase_root(root) / "feasibility/cuda_admission.json"
    admission = json.loads(admission_path.read_text())
    environment = runtime_environment_payload(device)
    if not _fixture:
        if (admission.get("admitted") is not True or admission.get("method_id") != METHOD_ID
                or admission.get("implementation_sha256") != implementation_fingerprint()
                or admission.get("backend") != config.backend
                or admission.get("environment") != environment
                or admission.get("dependency") != dependency_identity()
                or admission.get("batch_size") != config.batch_size
                or admission.get("data", {}).get(str(config.walk)) != manifest):
            raise ValueError("XM-C8 runtime/data admission mismatch")
    provenance = _provenance(manifest, config, admission)
    if (location / "training_complete.json").exists():
        return validate_endpoint_training(root, config)
    limit = config.epochs if stop_after_epoch is None else stop_after_epoch
    if not 0 <= limit <= config.epochs:
        raise ValueError("invalid interruption epoch")
    set_seed(config.seed)
    model = XLSTMMixerEndpoint(backend=config.backend).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    probe = torch.from_numpy(data["X_train"][:4].copy())
    history, epoch, peak = [], 0, 0
    if location.exists():
        if not (location / "resume.pth").is_file():
            raise ValueError("occupied XM-C8 run has no resumable checkpoint")
        saved = torch.load(location / "resume.pth", map_location=device, weights_only=True)
        if saved["provenance"] != provenance:
            raise ValueError("XM-C8 resume provenance mismatch")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        _restore_rng_state(saved["rng_states"], generator)
        epoch, history, peak = saved["completed_epoch"], saved["history"], saved["peak_cuda_memory_bytes"]
        torch.testing.assert_close(_probe_output(model, probe), saved["probe_output"].cpu(), rtol=1e-6, atol=1e-7)
        if [row["epoch"] for row in history] != list(range(1, epoch + 1)):
            raise ValueError("XM-C8 resume history is discontinuous")
        # A crash may occur after the atomic resume save but before snapshot save.
        if epoch in SNAPSHOTS and not (location / f"e{epoch}/checkpoint.pth").exists():
            _atomic_torch_save(location / f"e{epoch}/checkpoint.pth", saved)
    else:
        location.mkdir(parents=True)
        write_json(location / "config.json", config.to_dict())
        write_json(location / "architecture_manifest.json", architecture_manifest())
        write_json(location / "dataset_manifest.json", manifest)
        write_json(location / "environment.json", environment)
        _atomic_torch_save(location / "resume.pth", _checkpoint(model, optimizer, generator, 0, history, provenance, probe, peak))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epoch + 1, limit + 1):
        started = time.perf_counter()
        model.train()
        total, seen = 0.0, 0
        batches, remainder = shuffled_batches(len(data["X_train"]), config.batch_size, generator)
        for indices in batches:
            index = indices.numpy()
            x = torch.from_numpy(data["X_train"][index]).to(device)
            y = torch.from_numpy(data["y_train"][index].astype(np.float32).reshape(-1, 1)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.endpoint_mse_loss(model(x), y)
            loss.backward()
            _assert_finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(index)
            seen += len(index)
        if seen != len(data["X_train"]):
            raise RuntimeError("XM-C8 did not train on every original row")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
        history.append({"epoch": epoch, "train_loss": total / seen,
                        "seconds": time.perf_counter() - started, "seen_rows": seen,
                        "last_batch_size": remainder})
        checkpoint = _checkpoint(model, optimizer, generator, epoch, history, provenance, probe, peak)
        _atomic_torch_save(location / "resume.pth", checkpoint)
        _atomic_savez(location / "history.npz", _history_arrays(history))
        if epoch in SNAPSHOTS:
            _atomic_torch_save(location / f"e{epoch}/checkpoint.pth", checkpoint)
        print(f"walk={config.walk} method={METHOD} epoch={epoch} mse={total/seen:.8f}", flush=True)
    completed = len(history)
    if completed < config.epochs:
        return {"complete": False, "completed_epoch": completed}
    # No evaluation during trajectory or resume, including the early snapshots.
    for epoch in SNAPSHOTS:
        snapshot = location / f"e{epoch}"
        saved = torch.load(snapshot / "checkpoint.pth", map_location=device, weights_only=True)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        prediction = predict(model, data["X_test"], config.batch_size, device)
        metrics, arrays = evaluate_endpoint(prediction, data)
        _atomic_savez(snapshot / "predictions.npz", arrays)
        metrics.update({"method_id": METHOD_ID, "completed_epoch": epoch,
                        "predictions_sha256": sha256_file(snapshot / "predictions.npz"),
                        "checkpoint_sha256": sha256_file(snapshot / "checkpoint.pth")})
        write_json(snapshot / "metrics.json", metrics)
    replay = validate_endpoint_training(root, config)
    write_json(location / "training_complete.json", {"method_id": METHOD_ID,
               "completed_epoch": 50, "principal_epoch": 50, "replay_valid": replay["valid"]})
    return replay


def _equal_metrics(actual: Any, expected: Any) -> bool:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_equal_metrics(actual[k], expected[k]) for k in actual)
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        return (math.isnan(float(actual)) and math.isnan(float(expected))) or math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)
    return actual == expected


def validate_endpoint_training(root: Path, config: EndpointTrainingConfig) -> dict:
    data, manifest = load_endpoint_data(root, config.walk)
    location, device = run_root(root, config.walk), resolve_device(config.device)
    admission = json.loads((phase_root(root) / "feasibility/cuda_admission.json").read_text())
    expected_provenance = _provenance(manifest, config, admission)
    model = XLSTMMixerEndpoint(backend=config.backend).to(device)
    for epoch in SNAPSHOTS:
        snapshot = location / f"e{epoch}"
        checkpoint = torch.load(snapshot / "checkpoint.pth", map_location=device, weights_only=True)
        if checkpoint["provenance"] != expected_provenance or checkpoint["completed_epoch"] != epoch:
            raise ValueError("XM-C8 replay checkpoint provenance mismatch")
        history = checkpoint["history"]
        if [row["epoch"] for row in history] != list(range(1, epoch + 1)) or any(
                row["seen_rows"] != len(data["y_train"]) or not math.isfinite(row["train_loss"]) for row in history):
            raise ValueError("XM-C8 replay history mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        torch.testing.assert_close(_probe_output(model, checkpoint["probe_input"].cpu()), checkpoint["probe_output"].cpu(), rtol=1e-6, atol=1e-7)
        output = predict(model, data["X_test"], config.batch_size, device)
        metrics, arrays = evaluate_endpoint(output, data)
        saved_metrics = json.loads((snapshot / "metrics.json").read_text())
        if saved_metrics["checkpoint_sha256"] != sha256_file(snapshot / "checkpoint.pth") or saved_metrics["predictions_sha256"] != sha256_file(snapshot / "predictions.npz"):
            raise ValueError("XM-C8 replay artifact hash mismatch")
        if any(not _equal_metrics(value, saved_metrics.get(key)) for key, value in metrics.items()):
            raise ValueError("XM-C8 replay metric mismatch")
        with np.load(snapshot / "predictions.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError("XM-C8 replay prediction fields mismatch")
            for key, value in arrays.items():
                if key == "prediction_future_price" or key == "prediction_delta_h8":
                    if not np.allclose(saved[key], value, rtol=1e-6, atol=1e-7):
                        raise ValueError("XM-C8 checkpoint predictions failed replay")
                elif not np.array_equal(saved[key], value):
                    raise ValueError(f"XM-C8 replay identity/target mismatch: {key}")
    with np.load(location / "history.npz", allow_pickle=False) as saved_history:
        expected_history = _history_arrays(history)
        if any(not np.array_equal(saved_history[key], value) for key, value in expected_history.items()):
            raise ValueError("XM-C8 cumulative history failed replay")
    result = {"valid": True, "method_id": METHOD_ID, "walk": config.walk,
              "snapshots": list(SNAPSHOTS), "implementation_sha256": implementation_fingerprint()}
    write_json(location / "replay_validation.json", result)
    return result
