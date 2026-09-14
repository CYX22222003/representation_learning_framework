from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


def _metric(pred: np.ndarray,target: np.ndarray,name: str) -> float:
    error=pred-target
    if name=="mae": return float(np.mean(np.abs(error)))
    if name=="rmse": return float(np.sqrt(np.mean(error**2)))
    if name=="mse": return float(np.mean(error**2))
    if name=="corr": return float(np.corrcoef(pred,target)[0,1]) if np.std(pred)>0 and np.std(target)>0 else float("nan")
    raise ValueError(name)


def paired_contract_interval(candidate: Path,reference: Path,metric: str,replicates: int=2000,seed: int=20260914) -> dict[str,float]:
    with np.load(candidate,allow_pickle=False) as c, np.load(reference,allow_pickle=False) as r:
        for key in ("targets","row_indices","contract_ids","window_starts"):
            if not np.array_equal(c[key],r[key]): raise ValueError(f"paired prediction identity mismatch: {key}")
        cp,rp,target,contracts=c["predictions"],r["predictions"],c["targets"],c["contract_ids"]
    ids=np.unique(contracts); rng=np.random.default_rng(seed); differences=[]
    for _ in range(replicates):
        sampled=rng.choice(ids,size=len(ids),replace=True); positions=np.concatenate([np.flatnonzero(contracts==cid) for cid in sampled])
        differences.append(_metric(cp[positions],target[positions],metric)-_metric(rp[positions],target[positions],metric))
    low,high=np.nanpercentile(differences,[2.5,97.5])
    return {"difference":_metric(cp,target,metric)-_metric(rp,target,metric),"ci95_low":float(low),"ci95_high":float(high)}


def main(argv: Sequence[str]|None=None) -> int:
    p=argparse.ArgumentParser(description="Aggregate the Phase 2 decoder matrix."); p.add_argument("matrix_root"); args=p.parse_args(argv)
    root=Path(args.matrix_root)
    try:
        paths=sorted(root.glob("*/*/seed*/e*/metrics.json"))
        if not paths: raise FileNotFoundError(f"no decoder metrics under {root}")
        groups=defaultdict(list); identity={}
        for path in paths:
            row=json.loads(path.read_text()); manifest=json.loads((path.parents[1]/"dataset_manifest.json").read_text())
            if not json.loads((path.parent/"replay.json").read_text()).get("verified"): raise ValueError(f"replay failed: {path}")
            key=(row["task"],int(row["epoch"]),int(row["seed"])); value=manifest["test_identity_hash"]
            if key in identity and identity[key]!=value: raise ValueError(f"identity mismatch for {key}")
            identity[key]=value; groups[(row["task"],row["decoder_id"],int(row["epoch"]))].append(row)
        records=[]
        for (task,decoder,epoch),rows in sorted(groups.items()):
            record={"task":task,"decoder_id":decoder,"epoch":epoch,"n_seeds":len(rows)}
            for metric in ("mae","rmse","mse","corr"):
                values=np.asarray([r[metric] for r in rows],float); record[f"{metric}_mean"]=float(np.nanmean(values)); record[f"{metric}_std"]=float(np.nanstd(values,ddof=1)) if len(values)>1 else 0.0
            record["parameter_count"]=rows[0]["parameter_count"]; records.append(record)
        intervals=[]
        for path in paths:
            row=json.loads(path.read_text())
            if row["decoder_id"]=="D0": continue
            reference=root/row["task"]/"D0"/f"seed{row['seed']}"/f"e{row['epoch']}"/"predictions.npz"
            if reference.exists():
                for metric in ("mae","rmse","mse","corr"):
                    intervals.append({"task":row["task"],"decoder_id":row["decoder_id"],"seed":row["seed"],"epoch":row["epoch"],"metric":metric,
                                      **paired_contract_interval(path.parent/"predictions.npz",reference,metric)})
        (root/"aggregate_metrics.json").write_text(json.dumps(records,indent=2,allow_nan=True)); (root/"paired_intervals.json").write_text(json.dumps(intervals,indent=2,allow_nan=True))
        lines=["# Phase 2 decoder results","","All fixed budgets are reported; no best-on-test decoder or epoch is selected.","", "| task | decoder | epoch | seeds | MAE | RMSE | correlation | parameters |","|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in records: lines.append(f"| {r['task']} | {r['decoder_id']} | {r['epoch']} | {r['n_seeds']} | {r['mae_mean']:.6f} ± {r['mae_std']:.6f} | {r['rmse_mean']:.6f} ± {r['rmse_std']:.6f} | {r['corr_mean']:.6f} ± {r['corr_std']:.6f} | {r['parameter_count']} |")
        (root/"aggregate_summary.md").write_text("\n".join(lines)+"\n"); print(f"wrote decoder reports under {root}"); return 0
    except Exception as exc: print(f"decoder report failed: {exc}",file=sys.stderr); return 1


if __name__=="__main__": raise SystemExit(main())
