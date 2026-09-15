from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tasks.phase2_decoders.data import build_price_label_bundle, sha256_file, validate_regression_labels
from tasks.phase2_decoders.reconstruct import reconstruct_contract_sequences


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build split-safe contract-aware price labels.")
    p.add_argument("--processed-npz", default="data/processed/market_4h_seq64_top50.npz")
    p.add_argument("--out-path", default="data/task_labels/price_prediction/price_4h_h1_seq64_top50.npz")
    p.add_argument("--timeframe", choices=("1h","4h","1d"), default="4h")
    p.add_argument("--seq-len", type=int, default=64); p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--data-dir", default="data"); p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--horizon", type=int, default=1); p.add_argument("--price-index", type=int, default=3)
    p.add_argument("--overwrite", action="store_true"); p.add_argument("--verify", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv); out = Path(args.out_path)
    try:
        if args.verify:
            with np.load(out, allow_pickle=False) as data: bundle={k:data[k].copy() for k in data.files}
            manifest=json.loads(Path(f"{out}.manifest.json").read_text())
            validation=validate_regression_labels(bundle, train_size=manifest["train_size"], test_size=manifest["test_size"])
            if sha256_file(manifest["processed_npz"]) != manifest["processed_npz_sha256"]:
                raise ValueError("processed NPZ checksum differs from price-label manifest")
            print(json.dumps(validation, indent=2)); return 0
        if (out.exists() or Path(f"{out}.manifest.json").exists()) and not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {out}")
        sequences,timestamps,contracts,train,test=reconstruct_contract_sequences(
            args.processed_npz,timeframe=args.timeframe,top_k=args.top_k,seq_len=args.seq_len,
            data_dir=args.data_dir,train_ratio=args.train_ratio)
        bundle=build_price_label_bundle(sequences,horizon=args.horizon,price_index=args.price_index,
                                        train_ratio=args.train_ratio,timestamps_ns=timestamps)
        validation=validate_regression_labels(bundle,train_size=len(train),test_size=len(test))
        manifest={"task":"price_prediction","label_mode":"next_stride_window_close","horizon":args.horizon,
                  "price_index":args.price_index,"timeframe":args.timeframe,"seq_len":args.seq_len,"top_k":args.top_k,
                  "train_ratio":args.train_ratio,"train_size":len(train),"test_size":len(test),
                  "processed_npz":args.processed_npz,"processed_npz_sha256":sha256_file(args.processed_npz),
                  "contracts":contracts,"split_safety":"labels built independently within contract splits",**validation}
        out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,**bundle)
        Path(f"{out}.manifest.json").write_text(json.dumps(manifest,indent=2)); print(json.dumps({"wrote":str(out),**validation},indent=2)); return 0
    except FileExistsError as exc: print(exc,file=sys.stderr); return 2
    except Exception as exc: print(f"price-label preparation failed: {exc}",file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
