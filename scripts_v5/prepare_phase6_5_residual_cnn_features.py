#!/usr/bin/env python3
"""Extract the two Phase 6.5D future-price feature stores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.phase6_5_residual_cnn_features import (  # noqa: E402
    build_residual_cnn_feature_store,
    validate_residual_cnn_feature_store,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--walks", default="1,2")
    args = parser.parse_args()
    try:
        walks = tuple(int(value.strip()) for value in args.walks.split(",") if value.strip())
        if not walks or any(walk not in (1, 2) for walk in walks):
            raise ValueError("--walks must contain 1 and/or 2")
        results = []
        root = ROOT / "experiments" / "phase6_5" / "residual_cnn"
        h8_root = ROOT / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8"
        for walk in walks:
            encoder_dataset = (
                ROOT
                / "experiments"
                / "phase5"
                / "data_preparation"
                / f"walk{walk}"
                / "market_1h_seq64_h2.npz"
            )
            dataset = h8_root / "data" / f"walk{walk}" / "market_1h_seq64_h8.npz"
            h0 = h8_root / "features" / f"walk{walk}" / "five_branch_epoch50.npz"
            output = (
                root
                / "features"
                / f"walk{walk}"
                / "absolute_price_h8"
                / "features.npz"
            )
            if output.is_file() and Path(f"{output}.manifest.json").is_file():
                result = validate_residual_cnn_feature_store(output)
                result["action"] = "validated_existing"
            else:
                result = build_residual_cnn_feature_store(
                    dataset,
                    h0,
                    encoder_dataset,
                    root,
                    output,
                    walk=walk,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                result["action"] = "extracted"
            results.append(result)
        print(json.dumps({"valid": True, "stores": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5D feature preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
