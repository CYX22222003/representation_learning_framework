#!/usr/bin/env python3
"""Extract task-aligned Phase 6 temporal branches and duplicate controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.phase6_encoder_variant_features import (  # noqa: E402
    TASKS,
    build_variant_feature_store,
    validate_variant_feature_store,
)


def task_paths(task: str, walk: int) -> tuple[Path, Path]:
    if task == "classification_h2":
        return (
            ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz",
            ROOT / "experiments" / "phase5" / "features" / f"walk{walk}" / "five_branch_epoch50.npz",
        )
    if task == "absolute_price_h8":
        base = ROOT / "experiments" / "phase5" / "downstream_addons" / "shared" / "h8"
        return (base / "data" / f"walk{walk}" / "market_1h_seq64_h8.npz", base / "features" / f"walk{walk}" / "five_branch_epoch50.npz")
    base = ROOT / "experiments" / "phase6" / "volatility_prediction"
    return (base / "data_preparation" / f"walk{walk}" / "volatility_1h_seq64_h8.npz", base / "features" / f"walk{walk}" / "five_branch_epoch50_h8.npz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    try:
        results = []
        root = ROOT / "experiments" / "phase6" / "encoder_variants"
        for walk in (1, 2):
            encoder_dataset = ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"
            for task in TASKS:
                dataset, h0 = task_paths(task, walk)
                output = root / "features" / f"walk{walk}" / f"{task}.npz"
                if output.is_file() and Path(f"{output}.manifest.json").is_file():
                    result = validate_variant_feature_store(output)
                    result["action"] = "validated_existing"
                else:
                    result = build_variant_feature_store(
                        dataset, h0, encoder_dataset, root, output,
                        task=task, walk=walk, device=args.device, batch_size=args.batch_size,
                    )
                    result["action"] = "extracted"
                results.append(result)
        print(json.dumps({"valid": True, "stores": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Phase 6 variant feature preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
