#!/usr/bin/env python3
"""Compute predeclared descriptive CKA for Phase 6 temporal encoders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.phase6_encoder_variants import analyze_walk_cka, validate_cka_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        results = []
        phase6 = ROOT / "experiments" / "phase6" / "encoder_variants"
        for walk in (1, 2):
            dataset = ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
            output = phase6 / "diagnostics" / "cka" / f"walk{walk}.json"
            if output.is_file():
                result = validate_cka_report(output, dataset)
            else:
                analyze_walk_cka(dataset, ROOT / "experiments" / "phase5" / "encoder_pretraining", phase6, output, walk=walk, device=args.device)
                result = validate_cka_report(output, dataset)
            results.append(result)
        print(json.dumps({"valid": True, "walks": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 CKA analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
