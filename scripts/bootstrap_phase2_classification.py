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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Create or resume the isolated Phase 2 classification matrix.")
    result.add_argument("--matrix-name", default="4h_h2_tau005")
    result.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    result.add_argument("--features-npz", default="data/features/features_4h_seq64_top50_phase1.npz")
    result.add_argument("--labels-npz", default="data/task_labels/trend_classification/probability_movement_4h_h2_tau005_seq64_top50.npz")
    result.add_argument("--ta-features-npz", default="data/features/phase2_ta_probability_movement_4h_h2_tau005.npz")
    result.add_argument("--protocols", default="P1U,P1O,P2")
    result.add_argument("--include-p0-reference", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--seeds", default="0,1,2")
    result.add_argument("--epoch-budgets", default="15,50,100")
    result.add_argument("--device", default="cuda")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument(
        "--replace-manifest", action="store_true",
        help="Replace only the frozen matrix manifest/commands after an approved scope change.",
    )
    return result


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_commands(args: argparse.Namespace, root: Path) -> list[list[str]]:
    py = sys.executable
    overwrite = ["--overwrite"] if args.overwrite else []
    commands = [
        [py, "scripts/prepare_probability_movement_labels.py", "--processed-npz", args.processed_npz,
         "--out-path", args.labels_npz, *overwrite],
        [py, "scripts/prepare_phase2_ta_features.py", "--labels-npz", args.labels_npz,
         "--out-path", args.ta_features_npz, *overwrite],
    ]
    protocols = _csv(args.protocols)
    if args.include_p0_reference and "P0" not in protocols:
        protocols.append("P0")
    unknown = sorted(set(protocols).difference({"P0", "P1U", "P1O", "P2"}))
    if unknown:
        raise ValueError(f"unknown protocols: {unknown}")
    for seed_text in _csv(args.seeds):
        int(seed_text)
        for protocol in protocols:
            shared = [
                "--labels-npz", args.labels_npz, "--alignment-npz", args.ta_features_npz,
                "--protocol", protocol, "--seed", seed_text,
                "--epoch-budgets", args.epoch_budgets, "--device", args.device, *overwrite,
            ]
            commands.append([
                py, "scripts/train_phase2_raw_mlp.py", "--processed-npz", args.processed_npz, *shared,
                "--run-root", str(root / "C1_raw_ohlcv_mlp" / protocol / f"seed{seed_text}"),
            ])
            commands.append([
                py, "scripts/train_phase2_framework_classifier.py", "--features-npz", args.features_npz,
                "--model-id", "C2_framework", *shared,
                "--run-root", str(root / "C2_framework" / protocol / f"seed{seed_text}"),
            ])
            commands.append([
                py, "scripts/train_phase2_ta_mlp.py", "--labels-npz", args.labels_npz,
                "--ta-features-npz", args.ta_features_npz, "--protocol", protocol,
                "--seed", seed_text, "--epoch-budgets", args.epoch_budgets,
                "--device", args.device, *overwrite,
                "--run-root", str(root / "C5_ta_mlp" / protocol / f"seed{seed_text}"),
            ])
    return commands


def _run_root(command: Sequence[str]) -> Path | None:
    if "--run-root" not in command:
        return None
    return Path(command[command.index("--run-root") + 1])


def _is_complete(run_root: Path, budgets: Sequence[int]) -> bool:
    required = [run_root / "sweep_metrics.json", run_root / "summary.md", run_root / "training_diagnostics.json"]
    for budget in budgets:
        required.extend([
            run_root / f"e{budget}" / "checkpoint.pth", run_root / f"e{budget}" / "metrics.json",
            run_root / f"e{budget}" / "predictions.npz", run_root / f"e{budget}" / "replay.json",
        ])
    return all(path.exists() for path in required)


def _execute(commands: Sequence[Sequence[str]], args: argparse.Namespace) -> None:
    budgets = [int(item) for item in _csv(args.epoch_budgets)]
    for index, command in enumerate(commands, start=1):
        run_root = _run_root(command)
        if run_root is not None and run_root.exists() and any(run_root.iterdir()) and not args.overwrite:
            if _is_complete(run_root, budgets):
                print(f"[{index}/{len(commands)}] complete; skipping {run_root}", flush=True)
                continue
            raise FileExistsError(f"partial run requires inspection or --overwrite: {run_root}")
        if index == 1 and Path(args.labels_npz).exists() and not args.overwrite:
            verify = [command[0], command[1], "--out-path", args.labels_npz, "--verify"]
            print(f"[{index}/{len(commands)}] verifying existing labels", flush=True)
            subprocess.run(verify, cwd=ROOT, check=True)
            continue
        if index == 2 and Path(args.ta_features_npz).exists() and not args.overwrite:
            verify = [command[0], command[1], "--labels-npz", args.labels_npz,
                      "--out-path", args.ta_features_npz, "--verify"]
            print(f"[{index}/{len(commands)}] verifying existing TA alignment", flush=True)
            subprocess.run(verify, cwd=ROOT, check=True)
            continue
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    matrix_root = ROOT / "experiments" / "framework" / "phase2" / "classification_relabelling" / args.matrix_name
    manifest_path = matrix_root / "matrix_manifest.json"
    try:
        commands = build_commands(args, matrix_root)
        source_paths = {
            "processed_npz": Path(args.processed_npz), "features_npz": Path(args.features_npz),
            "features_index_npz": Path(f"{args.features_npz}.index.npz"),
        }
        missing = [str(path) for path in source_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing frozen source artifacts: {missing}")
        manifest = {
            "matrix_name": args.matrix_name, "predeclared_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_protocols": [item for item in _csv(args.protocols) if item != "P0"],
            "includes_p0_untreated_reference": args.include_p0_reference or "P0" in _csv(args.protocols),
            "seeds": [int(item) for item in _csv(args.seeds)],
            "epoch_budgets": [int(item) for item in _csv(args.epoch_budgets)],
            "included_models": ["C1_raw_ohlcv_mlp", "C2_framework", "C5_ta_mlp"],
            "comparison_scope": "strict_ta_aligned_rows",
            "label_contract": {"horizon": 2, "threshold": 0.005, "classes": ["DOWN", "STABLE", "UP"]},
            "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
            "locked_test": True, "no_validation_or_early_stopping": True,
            "current_test_is_characterization_only": True,
            "training_run_count": sum(_run_root(command) is not None for command in commands),
            "commands": [shlex.join(command) for command in commands],
        }
        if manifest_path.exists() and not args.overwrite and not args.replace_manifest:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            comparable_keys = set(manifest).difference({"predeclared_at_utc"})
            if any(existing.get(key) != manifest.get(key) for key in comparable_keys):
                raise ValueError("requested matrix differs from the existing frozen manifest")
            manifest = existing
        else:
            matrix_root.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (matrix_root / "commands.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(manifest["commands"]) + "\n",
                encoding="utf-8",
            )
        print(f"Phase 2 matrix ready: {manifest_path} ({manifest['training_run_count']} training runs)")
        if args.execute:
            _execute(commands, args)
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Phase 2 bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
