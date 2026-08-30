from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize


EPS = 1e-8


@dataclass(frozen=True)
class GarchFitDiagnostics:
    status: str
    fallback_reason: str | None
    omega: float | None
    alpha: float | None
    beta: float | None
    persistence: float | None
    train_return_mean: float
    train_return_std: float
    train_return_variance: float
    volatility_cap: float
    fitted_forecast_count: int
    capped_forecast_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GarchForecastResult:
    prediction_raw: np.ndarray
    prediction_guarded: np.ndarray
    fallback_flag: np.ndarray
    capped_flag: np.ndarray
    next_variance: np.ndarray
    diagnostics: GarchFitDiagnostics


def close_log_returns(close_prices: np.ndarray, eps: float = EPS) -> np.ndarray:
    close = np.asarray(close_prices, dtype=np.float64).reshape(-1)
    if close.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(np.log(np.clip(close, eps, None)))


def aligned_realized_volatility_forecast(close_window: np.ndarray, next_variance: float, eps: float = EPS) -> float:
    close = np.asarray(close_window, dtype=np.float64).reshape(-1)
    if close.size < 3:
        raise ValueError("close_window must contain at least three prices")
    if not np.isfinite(next_variance) or next_variance < 0.0:
        raise ValueError("next_variance must be finite and non-negative")
    known = np.diff(np.log(np.clip(close[1:], eps, None)))
    return float(np.sqrt((float(np.sum(known**2)) + float(next_variance)) / (known.size + 1)))


class GuardedGarchForecaster:
    """Causal GARCH(1,1) volatility forecaster with deterministic EWMA fallback."""

    def __init__(
        self,
        *,
        ewma_decay: float = 0.94,
        min_std: float = 1e-8,
        scale_ratio_bounds: tuple[float, float] = (1e-4, 1e4),
        cap_quantile: float = 0.995,
        optimizer: Callable[..., object] | None = None,
    ) -> None:
        if not 0.0 < ewma_decay < 1.0:
            raise ValueError("ewma_decay must be in (0, 1)")
        if min_std <= 0.0:
            raise ValueError("min_std must be positive")
        if not 0.0 < cap_quantile <= 1.0:
            raise ValueError("cap_quantile must be in (0, 1]")
        self.ewma_decay = float(ewma_decay)
        self.min_std = float(min_std)
        self.scale_ratio_bounds = scale_ratio_bounds
        self.cap_quantile = float(cap_quantile)
        self._optimizer = optimizer or minimize

    def fit_predict(
        self,
        close_history: np.ndarray,
        row_indices: np.ndarray,
        *,
        seq_len: int,
        fit_row_limit: int | None = None,
        fit_price_count: int | None = None,
    ) -> GarchForecastResult:
        close = np.asarray(close_history, dtype=np.float64).reshape(-1)
        rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        if seq_len < 3:
            raise ValueError("seq_len must be at least 3")
        if rows.size and (rows.min() < 0 or rows.max() + seq_len > close.size):
            raise ValueError("row_indices are out of range for close_history and seq_len")
        fit_limit = int(rows.max() + 1 if fit_row_limit is None and rows.size else (fit_row_limit or 0))
        if fit_limit < 0 or fit_limit + seq_len > close.size + seq_len:
            raise ValueError("fit_row_limit is invalid")

        price_count = min(close.size, int(fit_price_count) if fit_price_count is not None else fit_limit + seq_len)
        if price_count < 2:
            raise ValueError("fit_price_count must leave at least two fitting prices")
        fit_returns = close_log_returns(close[:price_count])
        params, reason, mean, std, train_var = self._fit_params_or_fallback(fit_returns)

        fitted_rows = np.arange(max(0, fit_limit), dtype=np.int64)
        fitted_raw = self._forecast_rows(close, fitted_rows, seq_len, params, mean, std, train_var, guard=False)[0]
        finite_fitted = fitted_raw[np.isfinite(fitted_raw)]
        cap = float(np.quantile(finite_fitted, self.cap_quantile)) if finite_fitted.size else float(np.sqrt(train_var))
        if not np.isfinite(cap) or cap <= 0.0:
            cap = float(max(np.sqrt(train_var), EPS))

        raw, guarded_variance, fallback = self._forecast_rows(close, rows, seq_len, params, mean, std, train_var, guard=True)
        capped = raw > cap
        guarded = np.minimum(raw, cap)
        diagnostics = GarchFitDiagnostics(
            status="fallback" if reason else "fitted",
            fallback_reason=reason,
            omega=None if params is None else float(params[0]),
            alpha=None if params is None else float(params[1]),
            beta=None if params is None else float(params[2]),
            persistence=None if params is None else float(params[1] + params[2]),
            train_return_mean=float(mean),
            train_return_std=float(std),
            train_return_variance=float(train_var),
            volatility_cap=float(cap),
            fitted_forecast_count=int(fitted_rows.size),
            capped_forecast_count=int(np.sum(capped)),
        )
        return GarchForecastResult(
            prediction_raw=raw.astype(np.float32),
            prediction_guarded=guarded.astype(np.float32),
            fallback_flag=fallback.astype(bool),
            capped_flag=capped.astype(bool),
            next_variance=guarded_variance.astype(np.float32),
            diagnostics=diagnostics,
        )

    def _fit_params_or_fallback(
        self, returns: np.ndarray
    ) -> tuple[np.ndarray | None, str | None, float, float, float]:
        r = np.asarray(returns, dtype=np.float64).reshape(-1)
        finite = r[np.isfinite(r)]
        if finite.size < 5:
            train_var = float(np.var(finite)) if finite.size else EPS**2
            return None, "too_few_returns", float(np.mean(finite)) if finite.size else 0.0, float(np.sqrt(max(train_var, EPS**2))), max(train_var, EPS**2)
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        train_var = float(np.var(finite))
        if std < self.min_std:
            return None, "near_static", mean, max(std, self.min_std), max(train_var, EPS**2)

        z = (finite - mean) / std

        def objective(x: np.ndarray) -> float:
            omega, alpha, beta = x
            if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.9999:
                return 1e30
            h = max(float(np.var(z)), EPS**2)
            total = 0.0
            for value in z:
                h = max(omega + alpha * float(value * value) + beta * h, EPS**2)
                total += np.log(h) + float(value * value) / h
            return 0.5 * total

        result = self._optimizer(
            objective,
            np.asarray([0.05, 0.05, 0.90], dtype=np.float64),
            method="SLSQP",
            bounds=((1e-12, 10.0), (0.0, 0.999), (0.0, 0.999)),
            constraints=({"type": "ineq", "fun": lambda x: 0.9999 - x[1] - x[2]},),
            options={"maxiter": 500, "ftol": 1e-9, "disp": False},
        )
        if not getattr(result, "success", False):
            return None, "optimizer_failed", mean, std, max(train_var, EPS**2)
        params = np.asarray(getattr(result, "x", np.asarray([])), dtype=np.float64)
        if params.shape != (3,) or not np.all(np.isfinite(params)):
            return None, "nonfinite_parameters", mean, std, max(train_var, EPS**2)
        if params[0] <= 0.0 or params[1] < 0.0 or params[2] < 0.0 or params[1] + params[2] >= 0.9999:
            return None, "constraint_violation", mean, std, max(train_var, EPS**2)
        return params, None, mean, std, max(train_var, EPS**2)

    def _forecast_rows(
        self,
        close: np.ndarray,
        rows: np.ndarray,
        seq_len: int,
        params: np.ndarray | None,
        mean: float,
        std: float,
        train_var: float,
        *,
        guard: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds = np.empty(rows.size, dtype=np.float64)
        variances = np.empty(rows.size, dtype=np.float64)
        fallback = np.zeros(rows.size, dtype=bool)
        for i, row in enumerate(rows):
            window = close[int(row) : int(row) + seq_len]
            observed_returns = close_log_returns(window)
            if params is None:
                next_var = self._ewma_next_variance(observed_returns, train_var)
                fallback[i] = True
            else:
                next_var = self._garch_next_variance(observed_returns, params, mean, std)
                ratio = next_var / max(train_var, 1e-12)
                low, high = self.scale_ratio_bounds
                if guard and (not np.isfinite(next_var) or next_var < 0.0 or ratio < low or ratio > high):
                    next_var = self._ewma_next_variance(observed_returns, train_var)
                    fallback[i] = True
            variances[i] = max(float(next_var), 0.0)
            preds[i] = aligned_realized_volatility_forecast(window, variances[i])
        return preds, variances, fallback

    def _garch_next_variance(self, observed_returns: np.ndarray, params: np.ndarray, mean: float, std: float) -> float:
        omega, alpha, beta = params
        z = (np.asarray(observed_returns, dtype=np.float64) - mean) / max(std, self.min_std)
        if z.size == 0:
            return float((omega / max(1.0 - alpha - beta, 1e-6)) * std * std)
        h = max(float(np.var(z)), EPS**2)
        for value in z:
            h = max(float(omega + alpha * value * value + beta * h), EPS**2)
        return float(h * std * std)

    def _ewma_next_variance(self, observed_returns: np.ndarray, initial_variance: float) -> float:
        var = max(float(initial_variance), EPS**2)
        for value in np.asarray(observed_returns, dtype=np.float64).reshape(-1):
            var = self.ewma_decay * var + (1.0 - self.ewma_decay) * float(value * value)
        return float(max(var, EPS**2))
