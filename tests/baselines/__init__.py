from __future__ import annotations

from pathlib import Path

_src_baselines = Path(__file__).resolve().parents[2] / "src" / "baselines"
if _src_baselines.exists():
    __path__.append(str(_src_baselines))
