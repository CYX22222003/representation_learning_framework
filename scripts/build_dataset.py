import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from data_processing.file_list import list_top_k
from data_processing.data_processing import build_from_file_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", choices=["1h", "4h", "1d"], default="4h")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="data/processed")
    args = parser.parse_args()

    file_list = list_top_k(args.timeframe, args.top_k)
    train, test = build_from_file_list(file_list, args.seq_len)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"market_{args.timeframe}_seq{args.seq_len}_top{args.top_k}.npz"
    )
    np.savez_compressed(out_path, train=train, test=test)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()