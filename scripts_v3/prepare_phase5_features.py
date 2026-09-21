#!/usr/bin/env python3
"""Extract one walk's canonical Phase 5 frozen five-branch features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features.phase5_features import build_phase5_feature_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--encoder-dataset", type=Path)
    parser.add_argument("--encoder-root", type=Path, default=Path("experiments/phase5/encoder_pretraining"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset or Path(
        f"experiments/phase5/data_preparation/walk{args.walk}/market_1h_seq64_h2.npz"
    )
    output = args.output or Path(
        f"experiments/phase5/features/walk{args.walk}/five_branch_epoch50.npz"
    )
    result = build_phase5_feature_bundle(
        dataset,
        args.encoder_root,
        output,
        walk=args.walk,
        encoder_dataset_path=args.encoder_dataset,
        device=args.device,
        batch_size=args.batch_size,
        deterministic_chunk_size=args.chunk_size,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
