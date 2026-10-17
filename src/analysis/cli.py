"""Package commands; production training requires explicit Lumid admission."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from .artifacts import read_json, write_json_atomic
from .types import AnalysisContext, PathMapping, ReplayPolicy


def build_parser() -> argparse.ArgumentParser:
    from .profiles import SELECTED_PROFILES_PATH

    parser = argparse.ArgumentParser(description="Phase 7 branch analysis: canonical H0 or selected variants")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--path-map", type=Path, help="JSON list of recorded_root/resolved_root mappings")
    parser.add_argument("--tolerances", type=Path, help="ReplayPolicy JSON, frozen before evaluation")
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--profiles", type=Path, help="JSON assigning one existing framework per task")
    profiles.add_argument("--phase7c", dest="profiles", action="store_const", const=SELECTED_PROFILES_PATH,
                          help="Use the approved HC-ST / HB-AL / HC-AL task profiles")
    parser.add_argument("--sandbox", help="Verified target hostname; required for production training")
    parser.add_argument("--site", default="unspecified")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "prepare", "similarity", "conditional"):
        commands.add_parser(name)
    for name in ("launch", "run"):
        command = commands.add_parser(name)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        command.add_argument("--execute", action="store_true")
        if name == "run":
            command.add_argument("--detach", action="store_true", help="Detach the full pipeline with persistent combined logs")
    commands.add_parser("validate").add_argument("--stage", choices=("preparation", "runs", "all"), default="all")
    commands.add_parser("report").add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = args.repo_root.resolve()
        output = (args.output_root or root / ("experiments/phase7/selected_variants" if args.profiles
                                              else "experiments/phase7/ablation")).resolve()
        mappings = tuple(PathMapping(m["recorded_root"], Path(m["resolved_root"])) for m in read_json(args.path_map)) if args.path_map else ()
        policy = ReplayPolicy(**read_json(args.tolerances)) if args.tolerances else ReplayPolicy()
        from .profiles import load_profiles
        profiles = load_profiles(args.profiles) if args.profiles else ()
        context = AnalysisContext(root, output, mappings, policy, args.sandbox, args.site, profiles)
        if getattr(args, "detach", False):
            if not args.execute:
                raise ValueError("--detach requires --execute")
            from .pipeline import AnalysisPipeline
            AnalysisPipeline(context)._admit()
            folder = output / "logs" / uuid.uuid4().hex
            folder.mkdir(parents=True)
            incoming = list(argv if argv is not None else sys.argv[1:])
            incoming.remove("--detach")
            for option in ("--repo-root", "--output-root", "--path-map", "--tolerances", "--profiles"):
                if option in incoming:
                    index = incoming.index(option) + 1
                    incoming[index] = str(Path(incoming[index]).resolve())
            command = [sys.executable, "-m", "analysis", *incoming]
            with (folder / "combined.log").open("ab") as log:
                child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
            write_json_atomic(folder / "launch.json", {"pid": child.pid, "command": command, "log": folder / "combined.log"})
            (folder / "launcher.pid").write_text(str(child.pid) + "\n")
            print(f"Started pid={child.pid}; log={folder / 'combined.log'}", flush=True)
            return 0
        from .pipeline import AnalysisPipeline
        pipeline = AnalysisPipeline(context)
        if args.command in ("audit", "prepare", "similarity", "conditional"):
            value = getattr(pipeline, args.command)()
        elif args.command in ("launch", "run"):
            value = getattr(pipeline, args.command)(device=args.device, execute=args.execute)
        elif args.command == "validate":
            value = pipeline.validate(args.stage)
        else:
            value = pipeline.report(allow_partial=args.allow_partial)
        if hasattr(value, "usable"):
            print(f"{value.subject}: {value.status}")
            return 0 if value.usable else 1
        if hasattr(value, "complete"):
            print(f"Report complete={value.complete}: {value.output_path}")
            return 0 if value.complete or getattr(args, "allow_partial", False) or not getattr(args, "execute", True) else 1
        return 0
    except Exception as exc:
        print(f"analysis: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
