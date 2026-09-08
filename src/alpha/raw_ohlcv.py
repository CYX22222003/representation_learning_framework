"""Causal, cross-sectional raw-OHLCV factor diagnostics.

This module is intentionally small.  It implements a predeclared handful of
price/volume formulae suitable for a dry run, not a genetic-programming search
or a trading backtest.  Every feature at timestamp ``t`` uses observations at
or before ``t``; the caller supplies the next-bar return separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FORMULA_NAMES = (
    "reversal_1",
    "intraday_reversal",
    "open_volume_corr_10",
    "vwap_high_corr_10",
    "volume_weighted_momentum_3",
)


@dataclass(frozen=True)
class CrossSectionScore:
    mean_ic: float
    mean_rank_ic: float
    rank_ic_ir: float
    positive_rank_ic_fraction: float
    mean_top_minus_bottom_return: float
    n_dates: int
    n_observations: int
    block_rank_ic: tuple[float, ...]


def _safe_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window, min_periods=window).corr(y).replace([np.inf, -np.inf], np.nan)


def _trailing_rank(x: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the latest value in a causal trailing window."""
    return x.rolling(window=window, min_periods=window).apply(
        lambda values: pd.Series(values).rank(method="average", pct=True).iloc[-1], raw=True
    )


def build_raw_ohlcv_formulae(frame: pd.DataFrame) -> pd.DataFrame:
    """Return Alpha101-style causal formula values for one contract.

    Required columns are ``date, open, high, low, close, volume``.  The VWAP
    proxy is the bar typical price because transaction-level VWAP is absent.
    """
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    out = frame.loc[:, ["date", "open", "high", "low", "close", "volume"]].copy()
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    close = out["close"].clip(lower=1e-6)
    bar_range = (out["high"] - out["low"]).abs().clip(lower=1e-6)
    intraday = (out["close"] - out["open"]) / bar_range
    typical_price = (out["high"] + out["low"] + out["close"]) / 3.0
    volume = out["volume"].clip(lower=0.0)
    log_volume = np.log1p(volume)
    ret_1 = close.pct_change()
    momentum_3 = close.pct_change(3)

    # Formulae are deliberately shallow and predeclared.  The correlation
    # operators mirror common Alpha101/GP raw price-volume building blocks.
    out["reversal_1"] = -ret_1
    out["intraday_reversal"] = -intraday
    out["open_volume_corr_10"] = -_safe_corr(out["open"], log_volume, 10)
    out["vwap_high_corr_10"] = _safe_corr(typical_price / out["high"].clip(lower=1e-6), out["high"], 10)
    out["volume_weighted_momentum_3"] = momentum_3 * _trailing_rank(log_volume, 20)
    out["forward_return_1"] = close.shift(-1) / close - 1.0
    return out


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def score_cross_sectional_factor(
    panel: pd.DataFrame,
    factor_column: str,
    *,
    target_column: str = "forward_return_1",
    min_assets: int = 10,
    quantile: float = 0.2,
    n_blocks: int = 5,
) -> CrossSectionScore:
    """Score one factor by same-timestamp IC/RankIC and an equal-weight spread."""
    if not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    rows = panel.loc[:, ["date", factor_column, target_column]].dropna().copy()
    rows = rows[np.isfinite(rows[factor_column]) & np.isfinite(rows[target_column])]
    date_stats: list[tuple[pd.Timestamp, float, float, float, int]] = []
    for date, group in rows.groupby("date", sort=True):
        if len(group) < min_assets:
            continue
        x = group[factor_column].to_numpy(dtype=float)
        y = group[target_column].to_numpy(dtype=float)
        ic = _corr(x, y)
        rank_ic = _corr(pd.Series(x).rank(method="average").to_numpy(), pd.Series(y).rank(method="average").to_numpy())
        n_side = max(1, int(np.floor(len(group) * quantile)))
        ordered = group.sort_values(factor_column, kind="mergesort")
        spread = float(ordered.tail(n_side)[target_column].mean() - ordered.head(n_side)[target_column].mean())
        if np.isfinite(ic) and np.isfinite(rank_ic):
            date_stats.append((date, ic, rank_ic, spread, len(group)))
    if not date_stats:
        raise ValueError("no usable cross-sections; lower min_assets or inspect overlap")
    stats = pd.DataFrame(date_stats, columns=["date", "ic", "rank_ic", "spread", "n_assets"])
    blocks = np.array_split(stats["rank_ic"].to_numpy(), min(n_blocks, len(stats)))
    block_rank_ic = tuple(float(block.mean()) for block in blocks if len(block))
    rank_mean = float(stats["rank_ic"].mean())
    rank_std = float(stats["rank_ic"].std(ddof=0))
    return CrossSectionScore(
        mean_ic=float(stats["ic"].mean()),
        mean_rank_ic=rank_mean,
        rank_ic_ir=rank_mean / rank_std if rank_std > 0 else 0.0,
        positive_rank_ic_fraction=float((stats["rank_ic"] > 0).mean()),
        mean_top_minus_bottom_return=float(stats["spread"].mean()),
        n_dates=len(stats),
        n_observations=int(stats["n_assets"].sum()),
        block_rank_ic=block_rank_ic,
    )
