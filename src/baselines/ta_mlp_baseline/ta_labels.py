"""Tri-class BUY / HOLD / SELL labeling for the TA-MLP baseline.

Ported from the upstream project (see REFERENCE/) with two changes:
  * vectorized — no per-row df.apply
  * alpha / beta are fitted per contract on TRAIN rows only

Label convention (CrossEntropy-friendly, remapped from the original
{-1, 0, +1}):

    0 = BUY   (forward price > backward MA, by more than alpha)
    1 = HOLD  (move below alpha, or above the effective beta cap)
    2 = SELL  (forward price < backward MA, by more than alpha)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUY, HOLD, SELL = 0, 1, 2


def compute_thresholds(
    close: pd.Series,
    hold_q: float = 0.85,
    buy_sell_q: float = 0.997,
) -> tuple[float, float]:
    """Quantile-based (alpha, beta) from |pct_change(close)|.

    Mirrors the upstream ``_find_alpha_beta`` but operates on whatever slice
    of close prices you pass in — typically the training rows only.
    """
    pct = close.pct_change().abs().dropna()
    if len(pct) == 0:
        return 0.0, 0.0
    return float(pct.quantile(hold_q)), float(pct.quantile(buy_sell_q))


def assign_labels(
    close: pd.Series,
    b_window: int,
    f_window: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Vectorized tri-class labels.

    Formula (matches REFERENCE/technical_analysis_tool.py::assign_labels):

        close_MA       = EWM(close, span=b_window).mean()
        s_1            = close.shift(-f_window)             # forward look
        r              = |s_1 - close_MA| / close_MA
        effective_beta = beta * (1 + f_window * 0.1)
        if alpha < r < effective_beta:
            label = BUY  if s_1 > close_MA
                    SELL if s_1 < close_MA
        else:
            label = HOLD

    Returns
    -------
    labels : np.ndarray[int64], shape (len(close),)
        Last ``f_window`` rows are unlabelable (forward shift yields NaN) —
        callers should drop them.  These rows are filled with HOLD here but
        should never be used.
    """
    close_ma = close.ewm(span=b_window).mean()
    s_1 = close.shift(-f_window)
    diff = s_1 - close_ma
    r = (diff.abs() / close_ma).to_numpy()
    sign = np.sign(diff.to_numpy())

    effective_beta = beta * (1 + f_window * 0.1)

    in_band = (r > alpha) & (r < effective_beta) & np.isfinite(r)

    labels = np.full(len(close), HOLD, dtype=np.int64)
    labels[in_band & (sign > 0)] = BUY
    labels[in_band & (sign < 0)] = SELL
    return labels
