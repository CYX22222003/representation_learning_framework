#!/usr/bin/env python3
"""Extract frozen five-branch features for the eight-hour downstream add-ons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features.phase5_features import build_phase5_feature_bundle, validate_phase5_feature_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    results = []
    for walk in (1, 2):
        dataset = Path(
            f"experiments/phase5/downstream_addons/shared/h8/data/walk{walk}/market_1h_seq64_h8.npz"
        )
        encoder_dataset = Path(
            f"experiments/phase5/data_preparation/walk{walk}/market_1h_seq64_h2.npz"
        )
        output = Path(
            f"experiments/phase5/downstream_addons/shared/h8/features/walk{walk}/five_branch_epoch50.npz"
        )
        reuse_feature = Path(
            f"experiments/phase5/features/walk{walk}/five_branch_epoch50.npz"
        )
        if output.is_file() and Path(f"{output}.manifest.json").is_file():
            result = validate_phase5_feature_bundle(output, dataset_path=dataset)
            result["action"] = "validated_existing"
        else:
            result = build_phase5_feature_bundle(
                dataset,
                Path("experiments/phase5/encoder_pretraining"),
                output,
                walk=walk,
                encoder_dataset_path=encoder_dataset,
                reuse_feature_path=reuse_feature,
                reuse_dataset_path=encoder_dataset,
                device=args.device,
                batch_size=args.batch_size,
                deterministic_chunk_size=args.chunk_size,
                workers=args.workers,
            )
            result["action"] = "extracted"
        results.append(result)
    print(json.dumps({"valid": True, "walks": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
