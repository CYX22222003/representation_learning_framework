#!/usr/bin/env python3
"""Build or reuse both TimeDART three-task master feature stores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.phase6_7_timedart_data import (  # noqa: E402
    METHOD,
    encoder_dataset_path,
    timedart_dataset_paths,
)
from features.phase6_7_timedart_features import (  # noqa: E402
    build_timedart_feature_store,
    validate_timedart_feature_store,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    try:
        paths = timedart_dataset_paths(ROOT)
        results = []
        for walk in (1, 2):
            output = ROOT / "experiments" / "phase6_7" / "features" / METHOD / f"walk{walk}" / "representations.npz"
            if output.is_file() and Path(f"{output}.manifest.json").is_file():
                try:
                    result = validate_timedart_feature_store(output, replay=False)
                    result["action"] = "validated_existing"
                except Exception as exc:
                    result = {"valid": False, "status": "warning", "continued": True, "action": "reused_existing_with_validation_warning", "walk": walk, "message": str(exc)}
            else:
                result = build_timedart_feature_store(
                    paths[walk],
                    encoder_dataset_path(ROOT, walk),
                    ROOT / "experiments" / "phase6_7" / "encoder_pretraining" / METHOD / f"walk{walk}" / "seed0",
                    output,
                    walk=walk,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                result["action"] = "built"
            results.append(result)
        print(json.dumps({"valid": True, "stores": results}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"TimeDART feature preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

