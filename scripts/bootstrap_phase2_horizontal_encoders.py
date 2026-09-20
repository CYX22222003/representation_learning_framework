"""Freeze, execute, and resume the Phase-2 horizontal encoder matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("experiments/framework/phase2/encoder_refinement_horizontal")
PROCESSED = "data/processed/market_4h_seq64_top50.npz"
CANONICAL = "data/features/features_4h_seq64_top50_phase1.npz"
PRICE_LABELS = "data/task_labels/price_prediction/price_4h_h1_seq64_top50.npz"
VOLATILITY_LABELS = "data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz"
MOVEMENT_LABELS = "data/task_labels/trend_classification/probability_movement_4h_h2_tau005_seq64_top50.npz"
MOVEMENT_ALIGNMENT = "data/features/phase2_ta_probability_movement_4h_h2_tau005.npz"
CANONICAL_BRANCHES = "statistical,transformed,vae,contrastive,byol"


CONFIGS = {
    "H0": ("shared", CANONICAL_BRANCHES, None),
    "HC-SL": ("contrastive", "statistical,transformed,vae,contrastive_lstm,byol", None),
    "HC-ST": ("contrastive", "statistical,transformed,vae,contrastive_transformer,byol", None),
    "HC-AL": ("contrastive", f"{CANONICAL_BRANCHES},contrastive_lstm", None),
    "HC-AT": ("contrastive", f"{CANONICAL_BRANCHES},contrastive_transformer", None),
    "HC-DC": ("shared", CANONICAL_BRANCHES, "contrastive_dup1=contrastive"),
    "HC-ALT": ("contrastive", f"{CANONICAL_BRANCHES},contrastive_lstm,contrastive_transformer", None),
    "HC-DD": ("shared", CANONICAL_BRANCHES, "contrastive_dup1=contrastive,contrastive_dup2=contrastive"),
    "HB-SL": ("byol", "statistical,transformed,vae,contrastive,byol_lstm", None),
    "HB-ST": ("byol", "statistical,transformed,vae,contrastive,byol_transformer", None),
    "HB-AL": ("byol", f"{CANONICAL_BRANCHES},byol_lstm", None),
    "HB-AT": ("byol", f"{CANONICAL_BRANCHES},byol_transformer", None),
    "HB-DC": ("shared", CANONICAL_BRANCHES, "byol_dup1=byol"),
    "HB-ALT": ("byol", f"{CANONICAL_BRANCHES},byol_lstm,byol_transformer", None),
    "HB-DD": ("shared", CANONICAL_BRANCHES, "byol_dup1=byol,byol_dup2=byol"),
}


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _variant_paths(root: Path, variant: str, seed: int) -> tuple[Path, Path, Path]:
    run = root / "pretraining" / variant / f"seed{seed}"
    checkpoint = root / "checkpoints" / variant / f"seed{seed}.pth"
    features = root / "features" / "branches" / variant / f"seed{seed}.npz"
    return run, checkpoint, features


def _superset_path(root: Path, family: str, seed: int) -> Path:
    return root / "features" / "supersets" / family / f"seed{seed}.npz"


def _pretrain_commands(args: argparse.Namespace, root: Path) -> list[list[str]]:
    py = sys.executable
    commands: list[list[str]] = []
    for seed in args.seed_values:
        for variant in ("contrastive_lstm", "contrastive_transformer"):
            run, checkpoint, _ = _variant_paths(root, variant, seed)
            commands.append([
                py, "scripts/train_phase2_contrastive_encoder.py", "--variant", variant,
                "--run-name", f"horizontal-{variant}-seed{seed}", "--run-root", str(run),
                "--checkpoint-path", str(checkpoint), "--epoch-budgets", args.epoch_budgets,
                "--seed", str(seed), "--batch-size", "256", "--learning-rate", "0.001",
                "--weight-decay", "0.0001", "--temperature", "0.2", "--device", args.device,
            ])
        for variant in ("byol_lstm", "byol_transformer"):
            run, checkpoint, _ = _variant_paths(root, variant, seed)
            commands.append([
                py, "scripts/train_phase2_byol_encoder.py", "--variant", variant,
                "--run-root", str(run), "--checkpoint-path", str(checkpoint),
                "--epoch-budgets", args.epoch_budgets, "--seed", str(seed),
                "--batch-size", "256", "--learning-rate", "0.001", "--weight-decay", "0.0001",
                "--target-decay", "0.99", "--device", args.device,
            ])
    return commands


def _feature_commands(args: argparse.Namespace, root: Path) -> list[list[str]]:
    commands = []
    for seed in args.seed_values:
        for variant in ("contrastive_lstm", "contrastive_transformer", "byol_lstm", "byol_transformer"):
            _, checkpoint, features = _variant_paths(root, variant, seed)
            commands.append([
                sys.executable, "scripts/extract_phase2_encoder_features.py",
                "--processed-npz", PROCESSED, "--checkpoint", str(checkpoint),
                "--out-path", str(features), "--batch-size", "1024", "--device", args.device,
            ])
    return commands


def _bundle_commands(args: argparse.Namespace, root: Path) -> list[list[str]]:
    commands = []
    for seed in args.seed_values:
        for family, variants in {
            "contrastive": ("contrastive_lstm", "contrastive_transformer"),
            "byol": ("byol_lstm", "byol_transformer"),
        }.items():
            command = [
                sys.executable, "scripts/build_phase2_horizontal_feature_bundle.py",
                "--canonical-features", CANONICAL,
            ]
            for variant in variants:
                command.extend(["--candidate-features", str(_variant_paths(root, variant, seed)[2])])
            command.extend(["--out-path", str(_superset_path(root, family, seed))])
            commands.append(command)
    return commands


def _cka_commands(args: argparse.Namespace, root: Path) -> list[list[str]]:
    commands = []
    settings = {
        "contrastive": (
            "contrastive,contrastive_lstm,contrastive_transformer",
            "contrastive_dup1=contrastive,contrastive_dup2=contrastive",
        ),
        "byol": (
            "byol,byol_lstm,byol_transformer",
            "byol_dup1=byol,byol_dup2=byol",
        ),
    }
    for seed in args.seed_values:
        for family, (branches, aliases) in settings.items():
            commands.append([
                sys.executable, "scripts/compute_horizontal_linear_cka.py",
                "--features-npz", str(_superset_path(root, family, seed)),
                "--branches", branches, "--duplicate-aliases", aliases,
                "--out-root", str(root / "diagnostics" / "linear_cka" / family / f"seed{seed}"),
            ])
    return commands


def _probe_commands(args: argparse.Namespace, root: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    for seed in args.seed_values:
        for config_id, (family, branches, aliases) in CONFIGS.items():
            features = Path(CANONICAL) if family == "shared" else _superset_path(root, family, seed)
            common = ["--features-npz", str(features), "--branches", branches]
            if aliases:
                common.extend(["--branch-aliases", aliases])
            for task, labels in (("price_prediction", PRICE_LABELS), ("volatility_prediction", VOLATILITY_LABELS)):
                commands.append([
                    sys.executable, "scripts/train_framework.py", "--task", task,
                    "--processed-npz", PROCESSED, *common, "--labels-npz", labels,
                    "--run-root", str(root / "probes" / task / config_id / f"seed{seed}"),
                    "--epoch-budgets", args.epoch_budgets, "--seed", str(seed),
                    "--batch-size", "512", "--learning-rate", "0.0001", "--device", args.device,
                ])
            commands.append([
                sys.executable, "scripts/train_phase2_framework_classifier.py", *common,
                "--model-id", config_id, "--labels-npz", MOVEMENT_LABELS,
                "--alignment-npz", MOVEMENT_ALIGNMENT, "--protocol", "P2",
                "--run-root", str(root / "probes" / "probability_movement" / config_id / f"seed{seed}"),
                "--epoch-budgets", args.epoch_budgets, "--seed", str(seed),
                "--batch-size", "512", "--learning-rate", "0.0001", "--device", args.device,
            ])
    return commands


def build_stage_commands(args: argparse.Namespace, root: Path) -> dict[str, list[list[str]]]:
    return {
        "pretrain": _pretrain_commands(args, root),
        "features": _feature_commands(args, root),
        "bundles": _bundle_commands(args, root),
        "cka": _cka_commands(args, root),
        "probes": _probe_commands(args, root),
    }


def _artifact_target(command: Sequence[str]) -> tuple[str, Path]:
    script = command[1]
    if "train_phase2_" in script and "encoder.py" in script:
        return "pretrain", Path(command[command.index("--run-root") + 1])
    if script.endswith("extract_phase2_encoder_features.py"):
        return "npz", Path(command[command.index("--out-path") + 1])
    if script.endswith("build_phase2_horizontal_feature_bundle.py"):
        return "npz", Path(command[command.index("--out-path") + 1])
    if script.endswith("compute_horizontal_linear_cka.py"):
        return "cka", Path(command[command.index("--out-root") + 1])
    return "probe", Path(command[command.index("--run-root") + 1])


def _complete(
    kind: str,
    path: Path,
    budgets: list[int],
    command: Sequence[str] | None = None,
) -> bool:
    if kind == "pretrain":
        required = [path / "sweep_metrics.json", path / "summary.md"]
        required += [path / f"e{value}" / name for value in budgets for name in ("checkpoint.pth", "metrics.json", "history.npz")]
        if command is not None and "--checkpoint-path" in command:
            required.append(Path(command[command.index("--checkpoint-path") + 1]))
        return all(item.exists() for item in required)
    if kind == "npz":
        return all(item.exists() for item in (path, Path(f"{path}.index.npz"), Path(f"{path}.manifest.json")))
    if kind == "cka":
        return (path / "linear_cka.json").exists() and (path / "linear_cka.npz").exists()
    required = [path / "sweep_metrics.json", path / "summary.md"]
    required += [path / f"e{value}" / name for value in budgets for name in ("checkpoint.pth", "metrics.json", "predictions.npz", "replay.json")]
    return all(item.exists() for item in required)


def _execute(stages: list[str], stage_commands: dict[str, list[list[str]]], budgets: list[int]) -> None:
    total = sum(len(stage_commands[name]) for name in stages)
    position = 0
    for stage in stages:
        for command in stage_commands[stage]:
            position += 1
            kind, target = _artifact_target(command)
            if _complete(kind, target, budgets, command):
                print(f"[{position}/{total}] complete; skipping {target}", flush=True)
                continue
            if target.exists() and (target.is_file() or any(target.iterdir())):
                raise FileExistsError(f"partial artifact requires inspection: {target}")
            print(f"[{position}/{total}] {shlex.join(command)}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--seeds", default="0,1,2")
    result.add_argument("--epoch-budgets", default="15,50,100")
    result.add_argument("--stages", default="pretrain,features,bundles,cka,probes")
    result.add_argument("--device", default="cuda")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--replace-manifest", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.seed_values = [int(item) for item in _csv(args.seeds)]
        budgets = [int(item) for item in _csv(args.epoch_budgets)]
        stages = _csv(args.stages)
        allowed = {"pretrain", "features", "bundles", "cka", "probes"}
        if not args.seed_values or not budgets or set(stages).difference(allowed):
            raise ValueError("invalid seeds, budgets, or stages")
        sources = [Path(PROCESSED), Path(CANONICAL), Path(f"{CANONICAL}.index.npz"),
                   Path(PRICE_LABELS), Path(VOLATILITY_LABELS), Path(MOVEMENT_LABELS), Path(MOVEMENT_ALIGNMENT)]
        missing = [str(path) for path in sources if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing frozen sources: {missing}")
        stage_commands = build_stage_commands(args, args.output_root)
        all_commands = [command for name in ("pretrain", "features", "bundles", "cka", "probes") for command in stage_commands[name]]
        manifest = {
            "experiment": "phase2_horizontal_encoder_branch_expansion",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(args.output_root), "seeds": args.seed_values,
            "epoch_budgets": budgets,
            "configurations": {name: list(values) for name, values in CONFIGS.items()},
            "tasks": ["price_prediction", "volatility_prediction", "probability_movement"],
            "probability_movement_protocol": "P2", "fusion": "concat",
            "probe_hidden_dim": 128, "encoder_embedding_dim": 128,
            "encoder_probe_seed_pairing": "candidate encoder seed s is paired with probe seed s; H0 and duplicate controls use canonical seed-0 encoders with probe seed s",
            "leave_one_out_scope": "new temporal branches only; represented by ALT versus AL/AT arms",
            "linear_cka_filtering": "none; CKA is recorded descriptively on full aligned train rows",
            "fixed_width_fusion": "deferred", "comparison_script_required": False,
            "current_test_is_characterization_only": True,
            "no_validation_or_early_stopping": True,
            "source_sha256": {str(path): _sha256(path) for path in sources},
            "stage_command_counts": {name: len(commands) for name, commands in stage_commands.items()},
            "commands": [shlex.join(command) for command in all_commands],
        }
        manifest_path = args.output_root / "matrix_manifest.json"
        if manifest_path.exists() and not args.replace_manifest:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            keys = set(manifest).difference({"frozen_at_utc"})
            if any(existing.get(key) != manifest.get(key) for key in keys):
                raise ValueError("requested matrix differs from frozen manifest")
            manifest = existing
        else:
            args.output_root.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (args.output_root / "commands.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(manifest["commands"]) + "\n",
                encoding="utf-8",
            )
        print(
            f"horizontal encoder matrix ready: {manifest_path} "
            f"({sum(manifest['stage_command_counts'].values())} commands; 135 probe trajectories)",
            flush=True,
        )
        if args.execute:
            _execute(stages, stage_commands, budgets)
        return 0
    except Exception as exc:
        print(f"horizontal encoder bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
