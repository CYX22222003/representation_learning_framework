"""Atomic persistence, manifests and prediction serialization."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Mapping

import numpy as np

from .types import (
    AnalysisContext, AnalysisTask, Array, PredictionRecord, ProbeRunSpec,
    ReplayPolicy, RowIdentity, RunState, SnapshotEpoch,
)

IDENTITY_FIELDS = ("condition_ids", "window_start_ns", "decision_date_ns", "decision_availability_ns")


def json_value(value):
    if is_dataclass(value):
        fields = asdict(value)
        # Preserve historical frozen-policy bytes unless this override is explicit.
        if isinstance(value, ReplayPolicy) and value.rank_median_metric_atol is None:
            fields.pop("rank_median_metric_atol")
        return json_value(fields)
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (Path, Enum)):
        return str(value.value if isinstance(value, Enum) else value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Non-finite JSON value needs an explicit availability reason")
    return value


def read_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    atomic_bytes(path, (json.dumps(json_value(payload), indent=2, sort_keys=True,
                                 allow_nan=False) + "\n").encode())


def atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_arrays_atomic(path: Path, arrays: Mapping[str, Array]) -> None:
    import io
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(path, buffer.getvalue())


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_array(values: Array) -> str:
    a = np.ascontiguousarray(values)
    digest = hashlib.sha256(a.dtype.str.encode())
    digest.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    digest.update(a.tobytes())
    return digest.hexdigest()


def run_path(context: AnalysisContext, spec: ProbeRunSpec) -> Path:
    return context.output_root / "downstream" / spec.source.task.value / spec.ablation.configuration / f"walk{spec.source.walk}" / f"seed{spec.recipe.seed}"


def snapshot_path(run_root: Path, epoch: SnapshotEpoch) -> Path:
    if epoch not in (5, 15, 50):
        raise ValueError("Undeclared snapshot epoch")
    return Path(run_root) / f"e{epoch}"


def record_run_state(run_root: Path, state: RunState) -> None:
    write_json_atomic(Path(run_root) / "state.json", {"state": state.value})


def identity_arrays(identity: RowIdentity) -> dict[str, Array]:
    return {**{name: np.asarray(getattr(identity, name)) for name in IDENTITY_FIELDS},
            **{f"target__{k}": np.asarray(v) for k, v in identity.target_fields.items()}}


def identity_hash(identity: RowIdentity) -> str:
    payload = {"task": identity.task.value, "walk": identity.walk,
               "arrays": {k: hash_array(v) for k, v in identity_arrays(identity).items()}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def subset_identity(identity: RowIdentity, mask) -> RowIdentity:
    return RowIdentity(identity.task, identity.walk,
                       *(np.asarray(getattr(identity, f))[mask] for f in IDENTITY_FIELDS),
                       {k: v[mask] for k, v in identity.target_fields.items()})


def subset_record(record: PredictionRecord, mask) -> PredictionRecord:
    return replace(record, identity=subset_identity(record.identity, mask),
                   targets=record.targets[mask], outputs=record.outputs[mask],
                   metadata={k: v[mask] for k, v in record.metadata.items()})


def save_prediction(path: Path, record: PredictionRecord) -> None:
    write_arrays_atomic(path, {"outputs": record.outputs, "targets": record.targets,
                        **identity_arrays(record.identity),
                        **{f"metadata__{k}": v for k, v in record.metadata.items()}})
    write_json_atomic(Path(f"{path}.json"), {
        "task": record.task, "walk": record.walk, "configuration": record.configuration,
        "seed": record.seed, "epoch": record.epoch, "units": record.units,
        "hashes": record.hashes, "identity_hash": identity_hash(record.identity),
        "file_sha256": hash_file(path),
    })


def load_prediction(path: Path) -> PredictionRecord:
    meta = read_json(Path(f"{path}.json"))
    if hash_file(path) != meta["file_sha256"]:
        raise ValueError("Prediction artifact hash mismatch")
    with np.load(path, allow_pickle=False) as stored:
        a = {k: np.asarray(stored[k]) for k in stored.files}
    identity = RowIdentity(AnalysisTask(meta["task"]), meta["walk"],
                           *(a[f] for f in IDENTITY_FIELDS),
                           {k[8:]: v for k, v in a.items() if k.startswith("target__")})
    if identity_hash(identity) != meta["identity_hash"]:
        raise ValueError("Prediction identity hash mismatch")
    return PredictionRecord(identity.task, identity.walk, meta["configuration"], meta["seed"],
                            meta["epoch"], identity, a["targets"], a["outputs"], meta["units"],
                            meta["hashes"], {k[10:]: v for k, v in a.items() if k.startswith("metadata__")})


def freeze_json(path: Path, payload: Mapping[str, object]) -> None:
    normalized = json_value(payload)
    if path.exists():
        if read_json(path) != normalized:
            raise ValueError(f"Frozen manifest differs: {path}")
    else:
        write_json_atomic(path, normalized)
