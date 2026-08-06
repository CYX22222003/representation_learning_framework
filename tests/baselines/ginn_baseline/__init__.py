from __future__ import annotations

from pathlib import Path

_src_ginn = Path(__file__).resolve().parents[3] / "src" / "baselines" / "ginn_baseline"
if _src_ginn.exists():
    __path__.append(str(_src_ginn))
