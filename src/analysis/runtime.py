"""Persistent Lumid runtime checks and hash-verified dependency closure."""

from __future__ import annotations

import importlib.metadata
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .artifacts import hash_file, read_json
from .types import AnalysisContext, DependencyFile, ExecutionEnvironment, TaskWalkSource, ValidationResult


def resolve_recorded_path(recorded_path: str, context: AnalysisContext) -> Path:
    normalized = recorded_path.replace("\\", "/")
    for mapping in sorted(context.path_mappings, key=lambda m: -len(m.recorded_root)):
        prefix = mapping.recorded_root.replace("\\", "/").rstrip("/")
        if normalized == prefix or normalized.startswith(prefix + "/"):
            path = mapping.resolved_root / normalized[len(prefix):].lstrip("/")
            if not path.resolve().is_relative_to(mapping.resolved_root.resolve()):
                raise ValueError("Path mapping escapes its declared root")
            return path
    path = Path(normalized)
    if path.is_absolute():
        if not path.resolve().is_relative_to(context.repo_root.resolve()):
            raise ValueError(f"Absolute source path needs an explicit root mapping: {path}")
        return path
    resolved = context.repo_root / path
    if not resolved.resolve().is_relative_to(context.repo_root.resolve()):
        raise ValueError("Relative source path escapes repository")
    return resolved


def collect_environment(context: AnalysisContext) -> ExecutionEnvironment:
    import torch
    versions = {}
    for name in ("numpy", "scipy", "torch", "pandas", "matplotlib"):
        versions[name] = importlib.metadata.version(name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=context.repo_root, text=True).strip()
    return ExecutionEnvironment(context.site, context.sandbox_name or "", socket.gethostname(),
        context.repo_root.resolve(), commit, Path(sys.executable).absolute(), versions,
        "cuda" if torch.cuda.is_available() else "cpu",
        f"torch={torch.__version__};cuda={torch.version.cuda}",
        context.output_root / "logs", context.output_root)


def inventory_dependencies(sources: Sequence[TaskWalkSource], context: AnalysisContext) -> tuple[DependencyFile, ...]:
    pending = []
    for source in sources:
        pending += [(p, "source") for p in (
            source.dataset_path, source.feature_path, source.dataset_manifest_path,
            source.feature_manifest_path, source.h0_scaler_path)]
        for name in ("config.json", "dataset_manifest.json", "architecture_manifest.json",
                     "training_complete.json", "sweep_metrics.json"):
            pending.append((source.h0_run_path / name, "h0"))
        for epoch in (5, 15, 50):
            for name in ("checkpoint.pth", "predictions.npz", "metrics.json", "history.npz"):
                pending.append((source.h0_run_path / f"e{epoch}" / name, "h0_snapshot"))
    seen = {}

    def references(obj, manifest_directory):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    # Only hash-backed file references are dependencies, not descriptive paths.
                    hash_key = "sha256" if key == "path" else key.removesuffix("_path") + "_sha256"
                    if key == "manifest_path":
                        hash_key = "manifest_sha256"
                    expected = obj.get(hash_key)
                    if isinstance(expected, str) and len(expected) == 64:
                        file_path = Path(value)
                        # Bare/relative lineage filenames belong to the containing manifest.
                        project_roots = {"experiments", "data_new", "data", "src", "docs", "checkpoints"}
                        resolved = (manifest_directory / file_path if not file_path.is_absolute()
                                    and file_path.parts[0] not in project_roots
                                    else resolve_recorded_path(value, context))
                        if not resolved.resolve().is_relative_to(context.repo_root.resolve()):
                            raise ValueError("Dependency escapes the project root")
                        if not resolved.is_file() or hash_file(resolved) != expected:
                            raise ValueError(f"Dependency missing or hash mismatch: {resolved}")
                        pending.append((resolved, "manifest_dependency"))
                elif isinstance(value, (dict, list)):
                    references(value, manifest_directory)
        elif isinstance(obj, list):
            for value in obj:
                references(value, manifest_directory)

    while pending:
        path, role = pending.pop()
        path = Path(path).resolve()
        if path in seen:
            continue
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty required input: {path}")
        relative = path.relative_to(context.repo_root.resolve())
        seen[path] = DependencyFile(relative, role, path.stat().st_size, hash_file(path))
        if path.suffix == ".json":
            references(read_json(path), path.parent)
        sidecar = Path(f"{path}.manifest.json")
        if sidecar.is_file():
            pending.append((sidecar, "sidecar"))
    return tuple(sorted(seen.values(), key=lambda item: str(item.relative_path)))


def preflight(context: AnalysisContext, environment: ExecutionEnvironment) -> ValidationResult:
    import numpy as np
    import torch
    checks = {}
    try:
        if not context.sandbox_name or context.sandbox_name != environment.hostname:
            raise ValueError("Supply --sandbox matching the actual Lumid hostname")
        if context.site == "unspecified":
            raise ValueError("Supply the verified Lumid --site")
        if not hasattr(np, "trapezoid"):
            raise ValueError("Existing classification metrics require NumPy >= 2 (trapezoid support)")
        for path in (context.repo_root, context.output_root, environment.python_executable):
            if not path.absolute().is_relative_to(Path("/home")):
                raise ValueError(f"Production path must be persistent under /home: {path}")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable on the target runtime")
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"],
                                        cwd=context.repo_root, text=True)
        if dirty.strip():
            raise ValueError("Production checkout must match its committed source")
        context.output_root.mkdir(parents=True, exist_ok=True)
        if not os.access(context.output_root, os.W_OK):
            raise ValueError("Output root is not writable")
        # Bounded synthetic probe; no experiment head or data is fitted.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            model = torch.nn.Linear(8, 1).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss = model(torch.ones(2, 8, device="cuda")).square().mean()
            loss.backward()
            optimizer.step()
            if not torch.isfinite(loss):
                raise ValueError("CUDA smoke loss is not finite")
        checks.update(hostname=environment.hostname, commit=environment.commit_sha,
                      gpu=torch.cuda.get_device_name(0), paths="persistent", cuda_probe="passed")
        return ValidationResult("preflight", "valid", checks)
    except (ValueError, OSError, RuntimeError) as exc:
        return ValidationResult("preflight", "invalid", {"failure": str(exc)})
