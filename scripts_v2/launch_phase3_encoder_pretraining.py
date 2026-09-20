"""Freeze or execute the seven-run Phase 3 encoder-pretraining matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_encoder_common import (  # noqa: E402
    DEFAULT_ENCODER_ROOT,
    DEFAULT_PROCESSED_NPZ,
    ENCODER_VARIANTS,
    EPOCH_BUDGETS,
    PHASE3_EXPERIMENT_ROOT,
    SEED,
    read_json,
    require_phase3_path,
    sha256_file,
    write_json,
)
from train_phase3_encoder import default_run_root  # noqa: E402
from validate_phase3_encoder_data import validate_bundle  # noqa: E402


MANIFEST_PATH = PHASE3_EXPERIMENT_ROOT / "manifests" / "encoder_pretraining_seed0.json"


def _commands(processed_npz: Path, device: str) -> list[list[str]]:
    python = str((ROOT / ".venv" / "bin" / "python3").resolve())
    trainer = str((ROOT / "scripts_v2" / "train_phase3_encoder.py").resolve())
    return [
        [
            python, trainer, "--variant", variant, "--processed-npz", str(processed_npz.resolve()),
            "--run-root", str(default_run_root(variant).resolve()), "--device", device,
        ]
        for variant in ENCODER_VARIANTS
    ]


def freeze(processed_npz: Path, device: str) -> dict:
    require_phase3_path(MANIFEST_PATH, PHASE3_EXPERIMENT_ROOT / "manifests")
    if MANIFEST_PATH.exists():
        raise FileExistsError(f"matrix manifest already exists: {MANIFEST_PATH}")
    validation = validate_bundle(processed_npz)
    commands = _commands(processed_npz, device)
    manifest = {
        "phase": 3,
        "subtask": "encoder_pretraining",
        "status": "frozen_not_executed",
        "processed_npz": str(processed_npz.resolve()),
        "processed_npz_sha256": validation["processed_npz_sha256"],
        "seed": SEED,
        "epoch_budgets": list(EPOCH_BUDGETS),
        "trajectory_policy": "one uninterrupted trajectory with snapshots",
        "principal_downstream_epoch": 50,
        "test_selection_policy": "no test-driven selection",
        "variants": list(ENCODER_VARIANTS),
        "commands": commands,
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def execute(processed_npz: Path) -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("freeze and review the matrix manifest before --execute")
    manifest = read_json(MANIFEST_PATH)
    if manifest.get("status") != "frozen_not_executed":
        raise ValueError("matrix manifest is not in frozen_not_executed state")
    validation = validate_bundle(processed_npz)
    if str(processed_npz.resolve()) != manifest["processed_npz"]:
        raise ValueError("processed path differs from frozen manifest")
    if validation["processed_npz_sha256"] != manifest["processed_npz_sha256"]:
        raise ValueError("processed data hash differs from frozen manifest")
    if manifest.get("variants") != list(ENCODER_VARIANTS):
        raise ValueError("frozen encoder matrix differs from code contract")
    for command in manifest["commands"]:
        if Path(command[1]).resolve().parent != (ROOT / "scripts_v2").resolve():
            raise ValueError("manifest attempts to execute a non-scripts_v2 entry point")
        subprocess.run(command, cwd=ROOT, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-npz", type=Path, default=DEFAULT_PROCESSED_NPZ)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute:
            execute(args.processed_npz)
            print("Phase 3 encoder-pretraining matrix completed")
        else:
            manifest = freeze(args.processed_npz, args.device)
            print(f"Phase 3 encoder matrix frozen for review: {MANIFEST_PATH}")
            print(f"runs: {len(manifest['commands'])}; no training was launched")
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"Phase 3 encoder launcher stopped after command failure: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Phase 3 encoder launcher failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
