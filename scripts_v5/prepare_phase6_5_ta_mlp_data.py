#!/usr/bin/env python3
"""Build causal Phase 6.5C TA stores and matched classification intersections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.ta_mlp_baseline.phase6_5 import build_phase65_ta_store  # noqa: E402


def paths(walk: int) -> tuple[Path, Path, Path]:
    dataset = ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
    features = ROOT / "experiments" / "phase6" / "encoder_variants" / "features" / f"walk{walk}" / "classification_h2.npz"
    output = ROOT / "experiments" / "phase6_5" / "ta_mlp" / "data_preparation" / f"walk{walk}" / "classification_h2_ta36.npz"
    return dataset, features, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walks", default="1,2")
    args = parser.parse_args()
    try:
        walks = tuple(int(value.strip()) for value in args.walks.split(",") if value.strip())
        if not walks or any(walk not in (1, 2) for walk in walks):
            raise ValueError("--walks must contain 1 and/or 2")
        results = []
        for walk in walks:
            dataset, features, output = paths(walk)
            results.append(build_phase65_ta_store(dataset, features, output))
        print(json.dumps({"valid": True, "prepared_walks": list(walks), "stores": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5C TA preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
