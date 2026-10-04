#!/usr/bin/env python3
"""Extract one three-task SaURL master representation store per walk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.phase6_7_external_features import (  # noqa: E402
    METHOD,
    build_external_feature_store,
    validate_external_feature_store,
)


def task_paths(walk: int) -> dict[str, Path]:
    return {
        "classification_h2": ROOT
        / "experiments"
        / "phase5"
        / "data_preparation"
        / f"walk{walk}"
        / "market_1h_seq64_h2.npz",
        "absolute_price_h8": ROOT
        / "experiments"
        / "phase5"
        / "downstream_addons"
        / "shared"
        / "h8"
        / "data"
        / f"walk{walk}"
        / "market_1h_seq64_h8.npz",
        "realised_variance": ROOT
        / "experiments"
        / "phase6"
        / "volatility_prediction"
        / "data_preparation"
        / f"walk{walk}"
        / "volatility_1h_seq64_h8.npz",
    }


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
        phase_root = ROOT / "experiments" / "phase6_7"
        for walk in walks:
            encoder_dataset = (
                ROOT
                / "experiments"
                / "phase5"
                / "data_preparation"
                / f"walk{walk}"
                / "market_1h_seq64_h2.npz"
            )
            encoder_run = (
                phase_root
                / "encoder_pretraining"
                / METHOD
                / f"walk{walk}"
                / "seed0"
            )
            output = phase_root / "features" / METHOD / f"walk{walk}" / "representations.npz"
            if output.is_file() and Path(f"{output}.manifest.json").is_file():
                result = validate_external_feature_store(output, replay=False)
                result["action"] = "validated_existing"
            else:
                result = build_external_feature_store(
                    task_paths(walk),
                    encoder_dataset,
                    encoder_run,
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
        print(f"Phase 6.7 feature preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
