"""Regression coverage for native control metric nesting; no training edits."""

from pathlib import Path
import runpy

import pytest


@pytest.mark.parametrize("nested", [False, True])
def test_endpoint_report_reads_overall_movement(nested):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[3] /
                                   "scripts_v8/report_phase6_9_xlstm_endpoint.py"))
    movement = {"spearman": 0.25}
    metrics = {"price": {"overall": {"count": 3, "mae": 0.1, "rmse": 0.2}},
               "implied_movement": {"overall": movement} if nested else movement}
    assert namespace["principal_row"](1, "control", metrics) == {
        "walk": 1, "method": "control", "count": 3, "mae": 0.1,
        "rmse": 0.2, "implied_movement_spearman": 0.25}
