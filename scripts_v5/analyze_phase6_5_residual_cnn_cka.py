#!/usr/bin/env python3
"""Compute the predeclared Phase 6.5D residual-CNN CKA diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.phase6_5_residual_cnn import (  # noqa: E402
    analyze_residual_cnn_cka,
    validate_residual_cnn_cka,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        results = []
        residual = ROOT / "experiments" / "phase6_5" / "residual_cnn"
        for walk in (1, 2):
            dataset = (
                ROOT
                / "experiments"
                / "phase5"
                / "data_preparation"
                / f"walk{walk}"
                / "market_1h_seq64_h2.npz"
            )
            output = residual / "diagnostics" / "cka" / f"walk{walk}.json"
            if output.is_file():
                result = validate_residual_cnn_cka(output, dataset)
            else:
                analyze_residual_cnn_cka(
                    dataset,
                    ROOT / "experiments" / "phase5" / "encoder_pretraining",
                    residual,
                    output,
                    walk=walk,
                    device=args.device,
                )
                result = validate_residual_cnn_cka(output, dataset)
            results.append(result)
        print(json.dumps({"valid": True, "walks": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6.5D CKA analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
