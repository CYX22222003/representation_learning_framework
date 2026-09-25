#!/usr/bin/env python3
"""Align frozen canonical 445-dimensional features to Phase 6 H=8 rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.phase6_volatility_features import (  # noqa: E402
    build_phase6_volatility_feature_bundle,
    validate_phase6_volatility_feature_bundle,
)


PHASE6_VOLATILITY_ROOT = ROOT / "experiments" / "phase6" / "volatility_prediction"
DEFAULT_OUTPUT_ROOT = PHASE6_VOLATILITY_ROOT / "features"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", choices=("1", "2", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        output_root = args.output_root.resolve()
        allowed = PHASE6_VOLATILITY_ROOT.resolve()
        if output_root != allowed and allowed not in output_root.parents:
            raise ValueError(f"Phase 6 volatility outputs must be under {allowed}")
        selected = (1, 2) if args.walk == "all" else (int(args.walk),)
        results = []
        for walk in selected:
            labels = (
                PHASE6_VOLATILITY_ROOT
                / "data_preparation"
                / f"walk{walk}"
                / "volatility_1h_seq64_h8.npz"
            )
            phase5_dataset = (
                ROOT
                / "experiments"
                / "phase5"
                / "data_preparation"
                / f"walk{walk}"
                / "market_1h_seq64_h2.npz"
            )
            phase5_features = (
                ROOT
                / "experiments"
                / "phase5"
                / "features"
                / f"walk{walk}"
                / "five_branch_epoch50.npz"
            )
            output = output_root / f"walk{walk}" / "five_branch_epoch50_h8.npz"
            if output.is_file() and Path(f"{output}.manifest.json").is_file():
                result = validate_phase6_volatility_feature_bundle(output)
                result["action"] = "validated_existing"
            else:
                result = build_phase6_volatility_feature_bundle(
                    labels, phase5_features, phase5_dataset, output, walk=walk
                )
                result["action"] = "aligned"
            results.append(result)
        print(json.dumps({"valid": True, "walks": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 volatility feature preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
