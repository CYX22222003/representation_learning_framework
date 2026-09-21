"""Train one canonical Phase 5 neural encoder for one global-calendar walk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.phase5_encoder import (  # noqa: E402
    ENCODERS,
    Phase5EncoderConfig,
    run_encoder_training,
)


def dataset_path(walk: int) -> Path:
    return ROOT / "experiments" / "phase5" / "data_preparation" / f"walk{walk}" / "market_1h_seq64_h2.npz"


def default_run_root(walk: int, encoder: str) -> Path:
    return ROOT / "experiments" / "phase5" / "encoder_pretraining" / f"walk{walk}" / encoder / "seed0"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", type=int, required=True, choices=(1, 2))
    parser.add_argument("--encoder", required=True, choices=ENCODERS)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    source = args.dataset or dataset_path(args.walk)
    output = args.run_root or default_run_root(args.walk, args.encoder)
    try:
        run_encoder_training(
            source,
            output,
            Phase5EncoderConfig(encoder=args.encoder, walk=args.walk, device=args.device),
        )
        print(f"Phase 5 encoder pretraining complete: {output}", flush=True)
        return 0
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Phase 5 encoder pretraining failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
