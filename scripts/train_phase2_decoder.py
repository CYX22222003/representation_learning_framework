from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))

from tasks.phase2_decoders.data import align_contexts_to_labels, load_npz, load_scaled_features, read_manifest, validate_temporal_index
from tasks.phase2_decoders.runner import DecoderRunConfig, run_decoder_experiment


def _budgets(value: str) -> tuple[int,...]: return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _normalize_labels(path: str|Path, task: str) -> dict[str,np.ndarray]:
    raw=load_npz(path); result={}
    for split in ("train","test"):
        prefix=f"{split}_"
        row_key=prefix+("row_indices" if prefix+"row_indices" in raw else "indices")
        result[prefix+"labels"]=np.asarray(raw[prefix+"labels"],np.float32)
        result[prefix+"row_indices"]=np.asarray(raw[row_key],np.int64)
        for name in ("contract_ids","window_starts"):
            result[prefix+name]=np.asarray(raw[prefix+name])
    return result


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Train one Phase 2 decoder trajectory.")
    p.add_argument("--task",choices=("price_prediction","volatility_prediction"),required=True)
    p.add_argument("--decoder-id",choices=("D0","D1","D2","D3","D4"),required=True)
    p.add_argument("--features-npz",default="data/features/features_4h_seq64_top50_phase1.npz")
    p.add_argument("--temporal-index-npz",default="data/features/phase2/temporal_index_4h_seq64_top50_k8.npz")
    p.add_argument("--labels-npz",required=True); p.add_argument("--run-root",required=True)
    p.add_argument("--context-length",type=int,default=8); p.add_argument("--epoch-budgets",default="15,50,100")
    p.add_argument("--seed",type=int,default=0); p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--learning-rate",type=float,default=1e-4); p.add_argument("--device",default="auto")
    p.add_argument("--overwrite",action="store_true"); return p


def main(argv: Sequence[str]|None=None) -> int:
    args=parser().parse_args(argv)
    try:
        temporal=load_npz(args.temporal_index_npz); validation=validate_temporal_index(temporal)
        if validation["context_length"]!=args.context_length: raise ValueError("context length differs from temporal index")
        labels=_normalize_labels(args.labels_npz,args.task)
        train_contexts,y_train,train_ids=align_contexts_to_labels(temporal,labels,"train")
        test_contexts,y_test,test_ids=align_contexts_to_labels(temporal,labels,"test")
        train_features,test_features,branch_dims,scaler=load_scaled_features(args.features_npz)
        config=DecoderRunConfig(args.task,args.decoder_id,args.context_length,_budgets(args.epoch_budgets),args.seed,args.batch_size,args.learning_rate,0.0,args.device)
        identities={f"train_{k}":v for k,v in train_ids.items()}|{f"test_{k}":v for k,v in test_ids.items()}
        manifest={"features_npz":args.features_npz,"temporal_index_npz":args.temporal_index_npz,"labels_npz":args.labels_npz,
                  "feature_manifest":read_manifest(args.features_npz),"temporal_manifest":read_manifest(args.temporal_index_npz),
                  "label_manifest":read_manifest(args.labels_npz),"fixed_phase1_encoders":True}
        run_decoder_experiment(train_features=train_features,test_features=test_features,train_contexts=train_contexts,
            test_contexts=test_contexts,y_train=y_train,y_test=y_test,identities=identities,branch_dims=branch_dims,
            scaler=scaler,run_root=args.run_root,config=config,dataset_manifest=manifest,overwrite=args.overwrite)
        print(f"completed Phase 2 decoder run: {args.run_root}"); return 0
    except FileExistsError as exc: print(exc,file=sys.stderr); return 2
    except Exception as exc: print(f"Phase 2 decoder training failed: {exc}",file=sys.stderr); return 1


if __name__=="__main__": raise SystemExit(main())
