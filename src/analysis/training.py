"""Fresh fixed-budget heads; failed attempts are preserved and restarted."""

from __future__ import annotations

import io
import random
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .artifacts import (atomic_bytes, hash_file, identity_hash, json_value, read_json,
                        record_run_state, save_prediction, snapshot_path, write_arrays_atomic, write_json_atomic)
from .contracts import task_recipe
from .profiles import source_branches
from .features import (apply_standardizer, fit_standardizer, load_standardizer,
                       save_standardizer, select_features, validate_scaler_parity)
from .tasks import build_head, build_loss, evaluate_predictions
from .types import (AnalysisContext, ExecutionEnvironment, PredictionRecord, ProbeRunSpec,
                    RunResult, RunState, SnapshotEpoch, TaskDataset)


def source_code_hashes(context):
    files = list((context.repo_root / "src/analysis").glob("*.py"))
    files += [context.repo_root / path for path in (
        "src/tasks/trend_classification.py", "src/tasks/volatility_prediction.py",
        "src/training/phase5_absolute_price.py", "src/tasks/phase2_classification/protocols.py",
        "src/tasks/phase2_classification/metrics.py")]
    if context.profiles:
        registries = {"temporal": "phase6_encoder_variant_features.py",
                      "lstm_capacity": "phase6_5_lstm_capacity_features.py",
                      "residual_cnn": "phase6_5_residual_cnn_features.py"}
        files += [context.repo_root / "src/features" / registries[family]
                  for family in sorted({p.family for p in context.profiles} - {"canonical"})]
    return {str(p.relative_to(context.repo_root)): hash_file(p) for p in files}


def run_manifest(spec, dataset, context):
    value = {"schema": 1, "task": spec.source.task, "walk": spec.source.walk,
            "configuration": spec.ablation, "recipe": spec.recipe,
            "train_identity": identity_hash(dataset.train.identity),
            "test_identity": identity_hash(dataset.test.identity),
            "source_hashes": dataset.source_hashes, "code_hashes": source_code_hashes(context)}
    if spec.source.profile:
        value["profile"] = spec.source.profile
    return value


def require_admission(context, environment):
    if read_json(context.output_root / "manifests/replay_policy.json") != json_value(context.replay_policy):
        raise ValueError("Replay tolerances differ from the frozen policy")
    saved = read_json(context.output_root / "manifests/preflight.json")
    if saved["validation"]["status"] not in ("valid", "valid_with_warnings"):
        raise ValueError("Production runtime has not passed preflight")
    if saved["environment"] != json_value(environment):
        raise ValueError("Runtime differs from admitted environment")
    if saved["code_hashes"] != source_code_hashes(context):
        raise ValueError("Source code changed since preflight")


def atomic_checkpoint(path, payload):
    import torch
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_bytes(path, buffer.getvalue())


def predict_output(model, values, device, multiplier=1.0):
    import torch
    model.eval()
    with torch.no_grad():
        outputs = [model(torch.from_numpy(np.array(batch, dtype=np.float32)).to(device)).cpu().numpy()
                   for batch in (values[start:start + 512] for start in range(0, len(values), 512))]
    return np.concatenate(outputs) / multiplier


class ProbeRunner:
    def __init__(self, context: AnalysisContext, environment: ExecutionEnvironment):
        self.context, self.environment = context, environment

    def train(self, spec: ProbeRunSpec, dataset: TaskDataset) -> RunResult:
        import fcntl
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        require_admission(self.context, self.environment)
        if dataset.source != spec.source:
            raise ValueError("Run source differs from supplied dataset")
        root = spec.output_path
        locks = self.context.output_root / "locks"
        locks.mkdir(parents=True, exist_ok=True)
        lock_path = locks / f"{spec.source.task.value}-{spec.ablation.configuration}-w{spec.source.walk}.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = json_value(run_manifest(spec, dataset, self.context))
            if root.exists():
                existing = root / "dataset_manifest.json"
                if not existing.is_file() or read_json(existing) != manifest:
                    raise ValueError("Occupied run has unknown or conflicting provenance")
                if (root / "training_complete.json").is_file():
                    from .replay import replay_snapshot
                    validations = [replay_snapshot(spec, dataset, e, context=self.context) for e in spec.recipe.snapshots]
                    if all(v.usable for v in validations):
                        return RunResult(spec, RunState.REPLAY_VALID, tuple(snapshot_path(root, e) for e in spec.recipe.snapshots))
                archive = self.context.output_root / "failed_attempts" / f"{root.parent.name}-{root.parent.parent.name}-{root.parent.parent.parent.name}-{uuid.uuid4().hex}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(root), archive)
            root.mkdir(parents=True, exist_ok=False)
            write_json_atomic(root / "dataset_manifest.json", manifest)
            write_json_atomic(root / "config.json", {"task": spec.source.task, "walk": spec.source.walk,
                              "configuration": spec.ablation.configuration, **asdict(spec.recipe)})
            write_json_atomic(root / "environment.json", json_value(self.environment))
            record_run_state(root, RunState.INITIALIZED)
            try:
                branches = source_branches(spec.source)
                raw = select_features(dataset.train, spec.ablation, branches)
                scaler = fit_standardizer(raw, dataset.train.identity)
                validate_scaler_parity(scaler, load_standardizer(spec.source.h0_scaler_path), spec.ablation, branches)
                save_standardizer(root / "feature_standardizer.npz", scaler)
                values = apply_standardizer(raw, scaler)
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
                device = torch.device(spec.device)
                model, criterion = build_head(spec).to(device), build_loss(spec, dataset.train.targets).to(device)
                recipe = task_recipe(spec.source.task)
                y = np.array(dataset.train.targets)
                targets = torch.from_numpy(y.astype(np.int64)) if recipe.output_dim == 3 else torch.from_numpy((y * recipe.target_multiplier).astype(np.float32).reshape(-1, 1))
                loader = DataLoader(TensorDataset(torch.from_numpy(values), targets),
                    batch_size=512, shuffle=True, drop_last=False, generator=torch.Generator().manual_seed(0))
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.)
                write_json_atomic(root / "architecture_manifest.json", {
                    "branches": spec.ablation.branches, "input_dim": spec.ablation.width,
                    "hidden_dims": [128, 64], "task_recipe": recipe})
                losses = []
                record_run_state(root, RunState.TRAINING)
                for epoch in range(1, 51):
                    model.train()
                    total, count = 0., 0
                    for x, target in loader:
                        optimizer.zero_grad(set_to_none=True)
                        loss = criterion(model(x.to(device)), target.to(device))
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite training loss")
                        loss.backward()
                        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                            raise FloatingPointError("Non-finite training gradient")
                        if recipe.gradient_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip_norm)
                        optimizer.step()
                        total += float(loss.detach()) * len(x)
                        count += len(x)
                    losses.append(total / count)
                    print(f"{spec.source.task.value}/{spec.ablation.configuration}/walk{spec.source.walk} epoch={epoch} loss={losses[-1]:.8g}", flush=True)
                    if epoch in spec.recipe.snapshots:
                        folder = snapshot_path(root, epoch)
                        atomic_checkpoint(folder / "checkpoint.pth", {
                            "model_state_dict": model.state_dict(), "completed_epoch": epoch,
                            "manifest": manifest, "scaler_sha256": hash_file(root / "feature_standardizer.npz")})
                        write_arrays_atomic(folder / "history.npz", {"epochs": np.arange(1, epoch + 1), "train_loss": np.asarray(losses)})
                write_arrays_atomic(root / "history.npz", {"epochs": np.arange(1, 51), "train_loss": np.asarray(losses)})
                record_run_state(root, RunState.TRAINED)
                for epoch in spec.recipe.snapshots:
                    record = self.predict_snapshot(spec, dataset, epoch)
                    folder = snapshot_path(root, epoch)
                    save_prediction(folder / "predictions.npz", record)
                    write_json_atomic(folder / "metrics.json", {"metrics": evaluate_predictions(record),
                                      "checkpoint_sha256": hash_file(folder / "checkpoint.pth"),
                                      "predictions_sha256": hash_file(folder / "predictions.npz")})
                write_json_atomic(root / "training_complete.json", {"complete": True, "snapshots": [5, 15, 50], "principal_epoch": 50})
                record_run_state(root, RunState.EVALUATED)
                return RunResult(spec, RunState.EVALUATED, tuple(snapshot_path(root, e) for e in spec.recipe.snapshots))
            except Exception as exc:
                record_run_state(root, RunState.FAILED)
                write_json_atomic(root / "failure.json", {"exception": type(exc).__name__, "message": str(exc)})
                raise

    def predict_snapshot(self, spec: ProbeRunSpec, dataset: TaskDataset, epoch: SnapshotEpoch) -> PredictionRecord:
        import torch
        folder = snapshot_path(spec.output_path, epoch)
        checkpoint = torch.load(folder / "checkpoint.pth", map_location=spec.device, weights_only=True)
        if checkpoint["manifest"] != json_value(run_manifest(spec, dataset, self.context)) or checkpoint["completed_epoch"] != epoch:
            raise ValueError("Snapshot provenance/epoch mismatch")
        path = spec.output_path / "feature_standardizer.npz"
        if hash_file(path) != checkpoint["scaler_sha256"]:
            raise ValueError("Snapshot scaler hash mismatch")
        scaler = load_standardizer(path)
        values = apply_standardizer(select_features(dataset.test, spec.ablation, source_branches(spec.source)), scaler)
        model = build_head(spec).to(spec.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        recipe = task_recipe(spec.source.task)
        output = predict_output(model, values, torch.device(spec.device), recipe.target_multiplier)
        if recipe.output_dim == 1:
            output = output.reshape(-1).astype(np.float64)
        units = {"classification_h2": "class_logits", "absolute_price_h8": "raw_probability",
                 "realised_variance": "raw_realised_variance"}[spec.source.task.value]
        return PredictionRecord(spec.source.task, spec.source.walk, spec.ablation.configuration, 0, epoch,
            dataset.test.identity, dataset.test.targets, output, units,
            {**dataset.source_hashes, "checkpoint": hash_file(folder / "checkpoint.pth")}, dataset.test.metadata)
