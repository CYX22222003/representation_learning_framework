"""Validate and report the complete Phase 3 framework price matrix."""

from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
from typing import Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase3_encoder_common import DEFAULT_PRICE_ROOT,PHASE3_EXPERIMENT_ROOT,write_json  # noqa:E402
from phase3_price_common import PRICE_CONFIGS,run_root  # noqa:E402

REPLAYED_METRICS=("mae","rmse","mse","corr","prediction_below_zero_fraction","prediction_above_one_fraction")

def _replay_metrics(predictions,targets):
    pred=np.asarray(predictions); target=np.asarray(targets)
    if pred.ndim!=1 or target.ndim!=1 or pred.shape!=target.shape or len(pred)==0: raise ValueError("invalid prediction/target shape")
    if not np.isfinite(pred).all() or not np.isfinite(target).all(): raise ValueError("non-finite prediction or target")
    error=pred-target; mse=float(np.mean(error**2))
    corr=float(np.corrcoef(pred,target)[0,1]) if len(pred)>1 and np.std(pred)>0 and np.std(target)>0 else float("nan")
    return {"mae":float(np.mean(np.abs(error))),"rmse":float(np.sqrt(mse)),"mse":mse,"corr":corr,
        "prediction_below_zero_fraction":float(np.mean(pred<0)),"prediction_above_one_fraction":float(np.mean(pred>1))}

def _same_metric(recorded,replayed):
    if np.isnan(replayed): return np.isnan(recorded)
    return np.isfinite(recorded) and np.isclose(recorded,replayed,rtol=1e-7,atol=1e-8)

def report(out_dir):
    if out_dir.exists() and any(out_dir.iterdir()): raise FileExistsError(out_dir)
    rows=[]; reference_targets=None; reference_ids=None
    for cid in PRICE_CONFIGS:
        root=run_root(cid); complete=json.loads((root/"training_complete.json").read_text())
        if complete.get("complete") is not True or complete.get("budgets")!=[5,15,50]: raise ValueError(f"incomplete {cid}")
        snapshots=json.loads((root/"sweep_metrics.json").read_text())["snapshots"]
        if [r["epoch"] for r in snapshots]!=[5,15,50]: raise ValueError(f"snapshot mismatch {cid}")
        for row in snapshots:
            p=root/f"e{row['epoch']}/predictions.npz"
            with np.load(p,allow_pickle=False) as data:
                targets=data["targets"]; ids=np.stack([data["contract_ids"],data["window_starts"],data["timestamps_ns"]],axis=1)
                if reference_targets is None: reference_targets,reference_ids=targets,ids
                elif not np.array_equal(targets,reference_targets) or not np.array_equal(ids,reference_ids): raise ValueError(f"row identity mismatch {cid}")
                replayed=_replay_metrics(data["predictions"],targets)
                for name in REPLAYED_METRICS:
                    if name not in row or not _same_metric(row[name],replayed[name]): raise ValueError(f"metric replay mismatch {cid} e{row['epoch']}: {name}")
            rows.append({"config_id":cid,**row,**replayed})
    out_dir.mkdir(parents=True,exist_ok=False); fields=sorted({k for r in rows for k in r})
    with (out_dir/"price_metrics.csv").open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
    write_json(out_dir/"price_metrics.json",{"snapshots":rows})
    fig,ax=plt.subplots(figsize=(12,6))
    for cid in PRICE_CONFIGS:
        subset=[r for r in rows if r["config_id"]==cid]; ax.plot([r["epoch"] for r in subset],[r["rmse"] for r in subset],marker="o",label=cid)
    ax.set(xlabel="Epoch",ylabel="RMSE",title="Phase 3 framework price prediction"); ax.grid(alpha=.25); ax.legend(ncol=3,fontsize=8); fig.tight_layout(); fig.savefig(out_dir/"price_rmse_by_budget.png",dpi=160); plt.close(fig)
    lines=["# Phase 3 framework price report","","All configurations use identical contract-local horizon-1 targets and rows. No best checkpoint is selected.","","| config | epoch | MAE | RMSE | correlation | below 0 | above 1 |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['config_id']} | {r['epoch']} | {r['mae']:.8f} | {r['rmse']:.8f} | {r['corr']:.8f} | {r['prediction_below_zero_fraction']:.6f} | {r['prediction_above_one_fraction']:.6f} |")
    (out_dir/"summary.md").write_text("\n".join(lines)+"\n"); result={"valid":True,"configurations":len(PRICE_CONFIGS),"snapshots":len(rows),"test_rows":len(reference_targets)}; write_json(out_dir/"report_manifest.json",result); return result
def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--out-dir",type=Path,default=PHASE3_EXPERIMENT_ROOT/"reports"/"price_framework_seed0"); a=p.parse_args(argv)
    try: print(f"Phase 3 price report complete: {report(a.out_dir)}"); return 0
    except Exception as exc: print(f"Phase 3 price report failed: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
