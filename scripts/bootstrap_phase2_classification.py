from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Create or execute the isolated Phase 2 classification matrix.")
    result.add_argument("--matrix-name", default="4h_h2_tau005")
    result.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    result.add_argument("--features-npz", default="data/features/features_4h_seq64_top50_phase1.npz")
    result.add_argument("--labels-npz", default="data/task_labels/trend_classification/probability_movement_4h_h2_tau005_seq64_top50.npz")
    result.add_argument("--ta-features-npz", default="data/features/phase2_ta_probability_movement_4h_h2_tau005.npz")
    result.add_argument("--protocols", default="P1U,P1O,P2")
    result.add_argument("--include-p0-reference", action="store_true")
    result.add_argument("--skip-full-row-primary", action="store_true")
    result.add_argument("--seeds", default="0,1,2")
    result.add_argument("--epoch-budgets", default="15,50,100")
    result.add_argument("--device", default="cuda")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
            shared = ["--labels-npz", args.labels_npz, "--alignment-npz", args.ta_features_npz,
                      "--protocol", protocol, "--seed", seed_text, "--epoch-budgets", args.epoch_budgets,
                      "--device", args.device, *overwrite]
            commands.append(
                [py, "scripts/train_phase2_raw_mlp.py", "--processed-npz", args.processed_npz, *shared,
                 "--run-root", str(root / "C1_raw_ohlcv_mlp" / protocol / f"seed{seed_text}")]
            )
            commands.append(
                [py, "scripts/train_phase2_framework_classifier.py", "--features-npz", args.features_npz, *shared,
                 "--run-root", str(root / "C2_framework" / protocol / f"seed{seed_text}")]
            )
            commands.append(
                [py, "scripts/train_phase2_ta_mlp.py", "--labels-npz", args.labels_npz,
                 "--ta-features-npz", args.ta_features_npz, "--protocol", protocol, "--seed", seed_text,
                 "--epoch-budgets", args.epoch_budgets, "--device", args.device, *overwrite,
                 "--run-root", str(root / "C5_ta_mlp" / protocol / f"seed{seed_text}")]
            )
        if not args.skip_full_row_primary:
            full_shared = [
                "--labels-npz", args.labels_npz, "--protocol", "P2", "--seed", seed_text,
                "--epoch-budgets", args.epoch_budgets, "--device", args.device, *overwrite,
            ]
            commands.append(
                [py, "scripts/train_phase2_raw_mlp.py", "--processed-npz", args.processed_npz,
                 *full_shared,
                 "--run-root", str(root / "C1_raw_ohlcv_mlp_full" / "P2" / f"seed{seed_text}")]
            )
            commands.append(
                [py, "scripts/train_phase2_framework_classifier.py", "--features-npz", args.features_npz,
                 *full_shared,
                 "--run-root", str(root / "C2_framework_full" / "P2" / f"seed{seed_text}")]
            )
    return commands


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    matrix_root = ROOT / "experiments" / "framework" / "phase2" / "classification_relabelling" / args.matrix_name
    manifest_path = matrix_root / "matrix_manifest.json"
    try:
        if manifest_path.exists() and not args.overwrite:
            raise FileExistsError(f"matrix manifest exists; pass --overwrite: {manifest_path}")
        commands = build_commands(args, matrix_root)
        matrix_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "matrix_name": args.matrix_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_protocols": [item for item in _csv(args.protocols) if item != "P0"],
            "includes_p0_untreated_reference": args.include_p0_reference or "P0" in _csv(args.protocols),
            "seeds": [int(item) for item in _csv(args.seeds)],
            "epoch_budgets": [int(item) for item in _csv(args.epoch_budgets)],
            "models": ["C1_raw_ohlcv_mlp", "C2_framework", "C5_ta_mlp"],
            "includes_full_row_c1_c2_p2": not args.skip_full_row_primary,
            "locked_test": True,
            "no_validation_or_early_stopping": True,
            "commands": [shlex.join(command) for command in commands],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (matrix_root / "commands.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(manifest["commands"]) + "\n",
            encoding="utf-8",
        )
        print(f"Phase 2 matrix bootstrapped: {manifest_path}")
        if args.execute:
            for index, command in enumerate(commands, start=1):
                print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"Phase 2 bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
