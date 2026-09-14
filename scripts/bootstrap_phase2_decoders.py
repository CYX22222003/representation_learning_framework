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

ROOT=Path(__file__).resolve().parents[1]
TASK_LABELS={
    "price_prediction":"data/task_labels/price_prediction/price_4h_h1_seq64_top50.npz",
    "volatility_prediction":"data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz",
}


def _csv(value: str) -> list[str]: return [v.strip() for v in value.split(",") if v.strip()]
def _sha(path: str|Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Freeze or execute the Phase 2 D0-D4 decoder matrix.")
    p.add_argument("--matrix-name",default="4h_k8"); p.add_argument("--processed-npz",default="data/processed/market_4h_seq64_top50.npz")
    p.add_argument("--features-npz",default="data/features/features_4h_seq64_top50_phase1.npz")
    p.add_argument("--temporal-index-npz",default="data/features/phase2/temporal_index_4h_seq64_top50_k8.npz")
    p.add_argument("--price-labels-npz",default=TASK_LABELS["price_prediction"])
    p.add_argument("--volatility-labels-npz",default=TASK_LABELS["volatility_prediction"])
    p.add_argument("--tasks",default="price_prediction,volatility_prediction"); p.add_argument("--decoders",default="D0,D1,D2,D3,D4")
    p.add_argument("--seeds",default="0,1,2"); p.add_argument("--epoch-budgets",default="15,50,100")
    p.add_argument("--context-length",type=int,default=8); p.add_argument("--device",default="cuda")
    p.add_argument("--execute",action="store_true"); p.add_argument("--overwrite",action="store_true")
    p.add_argument("--replace-manifest",action="store_true"); return p


def build_commands(args: argparse.Namespace, root: Path) -> list[list[str]]:
    py=sys.executable; overwrite=["--overwrite"] if args.overwrite else []
    commands=[
        [py,"scripts/prepare_phase2_temporal_index.py","--processed-npz",args.processed_npz,"--features-npz",args.features_npz,
         "--out-path",args.temporal_index_npz,"--context-length",str(args.context_length),*overwrite],
        [py,"scripts/prepare_price_labels.py","--processed-npz",args.processed_npz,"--out-path",args.price_labels_npz,*overwrite],
    ]
    labels={"price_prediction":args.price_labels_npz,"volatility_prediction":args.volatility_labels_npz}
    tasks=_csv(args.tasks); decoders=_csv(args.decoders); seeds=_csv(args.seeds)
    if set(tasks)-set(labels): raise ValueError("unknown Stage-1 task")
    if set(decoders)-{"D0","D1","D2","D3","D4"}: raise ValueError("unknown decoder")
    for task in tasks:
        for decoder in decoders:
            for seed in seeds:
                int(seed)
                commands.append([py,"scripts/train_phase2_decoder.py","--task",task,"--decoder-id",decoder,
                    "--features-npz",args.features_npz,"--temporal-index-npz",args.temporal_index_npz,
                    "--labels-npz",labels[task],"--run-root",str(root/task/decoder/f"seed{seed}"),
                    "--context-length",str(args.context_length),"--epoch-budgets",args.epoch_budgets,
                    "--seed",seed,"--device",args.device,*overwrite])
    return commands


def _run_root(command: Sequence[str]) -> Path|None:
    return Path(command[command.index("--run-root")+1]) if "--run-root" in command else None


def _complete(root: Path,budgets: list[int]) -> bool:
    required=[root/"sweep_metrics.json",root/"summary.md",root/"training_diagnostics.json"]
    for epoch in budgets: required += [root/f"e{epoch}"/name for name in ("checkpoint.pth","metrics.json","predictions.npz","replay.json")]
    return all(p.exists() for p in required)


def _execute(commands: list[list[str]],args: argparse.Namespace) -> None:
    budgets=[int(v) for v in _csv(args.epoch_budgets)]
    for i,command in enumerate(commands,1):
        run_root=_run_root(command)
        if run_root and run_root.exists() and any(run_root.iterdir()) and not args.overwrite:
            if _complete(run_root,budgets): print(f"[{i}/{len(commands)}] complete; skipping {run_root}",flush=True); continue
            raise FileExistsError(f"partial run requires inspection or --overwrite: {run_root}")
        if run_root is None:
            output=Path(command[command.index("--out-path")+1])
            if output.exists() and not args.overwrite:
                verify=[command[0],command[1],"--out-path",str(output),"--verify"]
                print(f"[{i}/{len(commands)}] verifying {output}",flush=True); subprocess.run(verify,cwd=ROOT,check=True); continue
        print(f"[{i}/{len(commands)}] {shlex.join(command)}",flush=True); subprocess.run(command,cwd=ROOT,check=True)


def main(argv: Sequence[str]|None=None) -> int:
    args=parser().parse_args(argv); root=ROOT/"experiments/framework/phase2/decoder_refinement"/args.matrix_name
    try:
        commands=build_commands(args,root); sources={"processed_npz":Path(args.processed_npz),"features_npz":Path(args.features_npz),
            "features_index_npz":Path(f"{args.features_npz}.index.npz"),"temporal_index_npz":Path(args.temporal_index_npz),
            "price_labels_npz":Path(args.price_labels_npz),"volatility_labels_npz":Path(args.volatility_labels_npz)}
        missing=[str(p) for p in sources.values() if not p.exists()]
        if missing: raise FileNotFoundError(f"missing frozen sources: {missing}")
        manifest={"matrix_name":args.matrix_name,"predeclared_at_utc":datetime.now(timezone.utc).isoformat(),
            "tasks":_csv(args.tasks),"decoders":_csv(args.decoders),"seeds":[int(v) for v in _csv(args.seeds)],
            "epoch_budgets":[int(v) for v in _csv(args.epoch_budgets)],"context_length":args.context_length,
            "training_run_count":sum(_run_root(c) is not None for c in commands),"source_sha256":{k:_sha(v) for k,v in sources.items()},
            "fixed_phase1_encoders":True,"no_validation_or_early_stopping":True,"current_test_is_characterization_only":True,
            "commands":[shlex.join(c) for c in commands]}
        path=root/"matrix_manifest.json"
        if path.exists() and not args.overwrite and not args.replace_manifest:
            old=json.loads(path.read_text()); keys=set(manifest)-{"predeclared_at_utc"}
            if any(old.get(k)!=manifest.get(k) for k in keys): raise ValueError("requested matrix differs from frozen manifest")
            manifest=old
        else:
            root.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(manifest,indent=2));
            (root/"commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n"+"\n".join(manifest["commands"])+"\n")
        print(f"Phase 2 decoder matrix ready: {path} ({manifest['training_run_count']} runs)")
        if args.execute: _execute(commands,args)
        return 0
    except FileExistsError as exc: print(exc,file=sys.stderr); return 2
    except Exception as exc: print(f"decoder bootstrap failed: {exc}",file=sys.stderr); return 1


if __name__=="__main__": raise SystemExit(main())
