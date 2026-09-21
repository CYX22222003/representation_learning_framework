"""Freeze or execute the six-run Phase 5 canonical encoder matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_processing.phase5_walks import validate_phase5_bundle_files  # noqa: E402
from training.phase5_encoder import ENCODERS, SNAPSHOT_EPOCHS, sha256_file, write_json  # noqa: E402
from train_phase5_encoder import dataset_path, default_run_root  # noqa: E402


MANIFEST_PATH = ROOT / "experiments" / "phase5" / "manifests" / "encoder_pretraining_seed0.json"


def commands(device: str) -> list[list[str]]:
    python = str(ROOT / ".venv" / "bin" / "python3")
    trainer = str(ROOT / "scripts_v3" / "train_phase5_encoder.py")
    return [
        [
            python,
            trainer,
            "--walk",
            str(walk),
            "--encoder",
            encoder,
            "--dataset",
            str(dataset_path(walk)),
            "--run-root",
            str(default_run_root(walk, encoder)),
            "--device",
            device,
        ]
        for walk in (1, 2)
        for encoder in ENCODERS
    ]


def freeze(device: str) -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        raise FileExistsError(f"matrix manifest already exists: {MANIFEST_PATH}")
    datasets: dict[str, Any] = {}
    for walk in (1, 2):
        path = dataset_path(walk)
        validation = validate_phase5_bundle_files(path)
        datasets[str(walk)] = {
            "path": str(path),
            "sha256": validation["npz_sha256"],
            "encoder_train_identity_hash": validation["identity_hashes"]["encoder_train"],
            "encoder_train_rows": validation["row_counts"]["encoder_train"],
        }
    manifest = {
        "phase": 5,
        "subtask": "canonical_encoder_pretraining",
        "status": "frozen_not_executed",
        "walks": [1, 2],
        "encoders": list(ENCODERS),
        "deterministic_branches": ["statistical", "transformed"],
        "seed": 0,
        "epochs": 50,
        "snapshot_epochs": list(SNAPSHOT_EPOCHS),
        "principal_downstream_epoch": 50,
        "trajectory_policy": "one uninterrupted trajectory per walk and neural encoder",
        "augmentation_policy": "existing five-channel scaling/jitter/time-mask; no explicit mask input",
        "selection_policy": "epoch 50 predeclared; no evaluation-driven selection",
        "datasets": datasets,
        "commands": commands(device),
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def execute() -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("freeze and review the Phase 5 matrix before --execute")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_not_executed":
        raise ValueError("matrix is not in frozen_not_executed state")
    if manifest.get("encoders") != list(ENCODERS) or manifest.get("snapshot_epochs") != list(SNAPSHOT_EPOCHS):
        raise ValueError("frozen matrix differs from the code contract")
    for walk in (1, 2):
        path = dataset_path(walk)
        validation = validate_phase5_bundle_files(path)
        frozen = manifest["datasets"][str(walk)]
        if validation["npz_sha256"] != frozen["sha256"]:
            raise ValueError(f"walk {walk} dataset changed after matrix freeze")
        if validation["identity_hashes"]["encoder_train"] != frozen["encoder_train_identity_hash"]:
            raise ValueError(f"walk {walk} encoder identities changed after matrix freeze")
    for command in manifest["commands"]:
        if Path(command[1]).resolve().parent != (ROOT / "scripts_v3").resolve():
            raise ValueError("matrix contains a non-scripts_v3 training entry point")
        subprocess.run(command, cwd=ROOT, check=True)
    completed = {**manifest, "status": "complete"}
    write_json(MANIFEST_PATH, completed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute:
            execute()
            print("Phase 5 encoder-pretraining matrix completed", flush=True)
        else:
            manifest = freeze(args.device)
            print(f"Phase 5 encoder matrix frozen for review: {MANIFEST_PATH}", flush=True)
            print(f"runs: {len(manifest['commands'])}; no training was launched", flush=True)
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"Phase 5 encoder launcher stopped after command failure: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Phase 5 encoder launcher failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
