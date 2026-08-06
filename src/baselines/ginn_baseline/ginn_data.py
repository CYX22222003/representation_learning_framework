from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from features.statistical import ar_coefficients

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
ARRAY_KEYS = (
    "X_train",
    "y_gt_train",
    "y_garch_train",
    "contract_id_train",
    "X_test",
    "y_gt_test",
    "y_garch_test",
    "contract_id_test",
)


@dataclass(frozen=True)
class GinnDataConfig:
    timeframe: str = "4h"
    seq_len: int = 64
    top_k: int = 50
    train_ratio: float = 0.8
    ar_order: int = 5
    garch_p: int = 1
    garch_q: int = 1

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class SplitIndices:
    n_samples: int
    n_train: int
    fit_end: int
    train_starts: np.ndarray
    test_starts: np.ndarray


@dataclass(frozen=True)
class GarchFit:
    params: np.ndarray
    residual_mean: float
    initial_variance: float
    converged: bool
    used_fallback: bool
    status: str


@dataclass(frozen=True)
class PreparedContract:
    X_train: np.ndarray
    y_gt_train: np.ndarray
    y_garch_train: np.ndarray
    contract_id_train: np.ndarray
    X_test: np.ndarray
    y_gt_test: np.ndarray
    y_garch_test: np.ndarray
    contract_id_test: np.ndarray
    train_starts: np.ndarray
    test_starts: np.ndarray
    ar_coefficients: np.ndarray
    garch: GarchFit
    raw_rows: int
    filename: str


@dataclass(frozen=True)
class GinnCache:
    X_train: np.ndarray
    y_gt_train: np.ndarray
    y_garch_train: np.ndarray
    contract_id_train: np.ndarray
    X_test: np.ndarray
    y_gt_test: np.ndarray
    y_garch_test: np.ndarray
    contract_id_test: np.ndarray
    manifest: dict[str, Any]


class ContractPreparationError(RuntimeError):
    def __init__(self, filename: str, reason: str, detail: str = "") -> None:
        message = f"{filename}: {reason}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.filename = filename
        self.reason = reason
        self.detail = detail


def compute_split_indices(
    n_rows: int, seq_len: int, train_ratio: float, ar_order: int
) -> SplitIndices:
    if seq_len <= 1:
        raise ValueError("seq_len must be greater than 1")
    if ar_order < 0:
        raise ValueError("ar_order must be non-negative")
    if n_rows <= seq_len:
        raise ValueError(f"insufficient_length: need more than {seq_len} rows, got {n_rows}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    n_samples = n_rows - seq_len
    n_train = int(np.floor(train_ratio * n_samples))
    fit_end = n_train + seq_len
    train_starts = np.arange(ar_order, n_train, dtype=np.int64)
    test_starts = np.arange(n_train, n_samples, dtype=np.int64)
    if train_starts.size == 0 or test_starts.size == 0:
        raise ValueError("insufficient_length: non-empty train and test samples required")
    return SplitIndices(n_samples, n_train, fit_end, train_starts, test_starts)


def transform_ohlcv(frame: pd.DataFrame) -> np.ndarray:
    missing = [name for name in OHLCV_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"missing_columns: {missing}")
    values = frame.loc[:, OHLCV_COLUMNS].to_numpy(dtype=np.float64)
    transformed = np.zeros_like(values, dtype=np.float64)
    transformed[1:, :4] = np.diff(np.log(np.clip(values[:, :4], 1e-8, None)), axis=0)
    volume = np.log1p(np.clip(values[:, 4], 0.0, None))
    transformed[1:, 4] = np.diff(volume)
    if not np.isfinite(transformed).all():
        raise ValueError("non_finite_transform")
    return transformed


def fit_ar_channels(series: np.ndarray, fit_end: int, ar_order: int) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(OHLCV_COLUMNS):
        raise ValueError("series must have shape [T, 5]")
    if fit_end <= ar_order:
        raise ValueError("fit_end must be greater than ar_order")
    coeffs = [
        ar_coefficients(arr[:fit_end, col], order=ar_order)[0]
        for col in range(arr.shape[1])
    ]
    return np.stack(coeffs).astype(np.float64)


def apply_ar_channels(series: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64)
    coef = np.asarray(coeffs, dtype=np.float64)
    if arr.ndim != 2 or coef.ndim != 2 or coef.shape[0] != arr.shape[1]:
        raise ValueError("series and coeffs shapes are incompatible")
    ar_order = coef.shape[1]
    residuals = np.zeros_like(arr, dtype=np.float64)
    for col in range(arr.shape[1]):
        for row in range(ar_order, len(arr)):
            lags = arr[row - ar_order : row, col][::-1]
            residuals[row, col] = arr[row, col] - float(lags @ coef[col])
    return residuals


def _garch_objective(params: np.ndarray, eps: np.ndarray, initial_variance: float) -> float:
    omega, alpha, beta = params
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
        return 1e10
    sigma2 = np.empty(len(eps), dtype=np.float64)
    sigma2[0] = initial_variance
    for index in range(1, len(eps)):
        sigma2[index] = omega + alpha * eps[index - 1] ** 2 + beta * sigma2[index - 1]
        if sigma2[index] <= 0.0 or not np.isfinite(sigma2[index]):
            return 1e10
    return 0.5 * float(np.sum(np.log(sigma2) + eps**2 / sigma2))


def fit_garch11(residuals: np.ndarray, fit_end: int, max_iter: int = 200) -> GarchFit:
    fitting = np.asarray(residuals[:fit_end], dtype=np.float64).reshape(-1)
    if fitting.size < 2:
        raise ValueError("fit_end must expose at least two residuals")
    residual_mean = float(fitting.mean())
    eps = fitting - residual_mean
    initial_variance = max(float(np.var(eps)), 1e-10)
    fallback = np.array([initial_variance * 0.05, 0.10, 0.85], dtype=np.float64)
    converged = False
    used_fallback = False
    status = "converged"
    try:
        result = minimize(
            _garch_objective,
            fallback,
            args=(eps, initial_variance),
            method="SLSQP",
            bounds=[(1e-10, None), (1e-8, 0.9998), (1e-8, 0.9998)],
            constraints={"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]},
            options={"maxiter": max_iter, "ftol": 1e-7},
        )
        converged = bool(result.success)
        params = np.asarray(result.x if converged else fallback, dtype=np.float64)
        if not converged:
            used_fallback = True
            status = f"fallback: optimizer status {result.status}"
    except Exception:
        params = fallback
        used_fallback = True
        status = "fallback: optimizer exception"
    omega = max(float(params[0]), 1e-10)
    alpha = float(np.clip(params[1], 1e-8, 0.9998))
    beta = float(np.clip(params[2], 1e-8, 0.9998))
    if alpha + beta >= 1.0:
        scale = 0.9999 / (alpha + beta)
        alpha *= scale
        beta *= scale
    return GarchFit(
        params=np.array([omega, alpha, beta], dtype=np.float64),
        residual_mean=residual_mean,
        initial_variance=initial_variance,
        converged=converged,
        used_fallback=used_fallback,
        status=status,
    )


def apply_garch11(residuals: np.ndarray, fit: GarchFit) -> np.ndarray:
    eps = np.asarray(residuals, dtype=np.float64).reshape(-1) - fit.residual_mean
    omega, alpha, beta = fit.params
    sigma2 = np.empty(len(eps), dtype=np.float64)
    sigma2[0] = fit.initial_variance
    for index in range(1, len(eps)):
        sigma2[index] = omega + alpha * eps[index - 1] ** 2 + beta * sigma2[index - 1]
    return np.clip(sigma2, 1e-10, None)


def realized_volatility_target(close: np.ndarray, start: int, seq_len: int) -> float:
    future = np.clip(np.asarray(close, dtype=np.float64)[start + 1 : start + seq_len + 1], 1e-8, None)
    if future.shape[0] != seq_len:
        raise ValueError("target window is shorter than seq_len")
    return float(np.sqrt(np.mean(np.diff(np.log(future)) ** 2)))


def garch_volatility_target(sigma2: np.ndarray, start: int, seq_len: int) -> float:
    future = np.asarray(sigma2, dtype=np.float64)[start + 1 : start + seq_len + 1]
    if future.shape[0] != seq_len:
        raise ValueError("garch target window is shorter than seq_len")
    return float(np.sqrt(np.mean(future)))


def _clean_contract(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    missing = [name for name in OHLCV_COLUMNS if name not in frame.columns]
    if missing:
        raise ContractPreparationError(filename, "missing_columns", str(missing))
    cleaned = frame.loc[:, OHLCV_COLUMNS].ffill().interpolate()
    values = cleaned.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ContractPreparationError(filename, "non_finite_cleaned_values")
    return cleaned


def _samples(
    residuals: np.ndarray,
    close: np.ndarray,
    sigma2: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    contract_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.stack([residuals[start : start + seq_len] for start in starts]).astype(np.float32)
    y_gt = np.array(
        [realized_volatility_target(close, int(start), seq_len) for start in starts],
        dtype=np.float32,
    ).reshape(-1, 1)
    y_garch = np.array(
        [garch_volatility_target(sigma2, int(start), seq_len) for start in starts],
        dtype=np.float32,
    ).reshape(-1, 1)
    ids = np.full(len(starts), contract_id, dtype=np.int32)
    return X, y_gt, y_garch, ids


def prepare_contract(
    frame: pd.DataFrame, contract_id: int, filename: str, config: GinnDataConfig
) -> PreparedContract:
    try:
        cleaned = _clean_contract(frame, filename)
        split = compute_split_indices(
            len(cleaned), config.seq_len, config.train_ratio, config.ar_order
        )
        transformed = transform_ohlcv(cleaned)
        coeffs = fit_ar_channels(transformed, split.fit_end, config.ar_order)
        residuals = apply_ar_channels(transformed, coeffs)
        garch = fit_garch11(residuals[:, 3], split.fit_end)
        sigma2 = apply_garch11(residuals[:, 3], garch)
        close = cleaned["close"].to_numpy(dtype=np.float64)
        X_train, y_gt_train, y_garch_train, contract_id_train = _samples(
            residuals, close, sigma2, split.train_starts, config.seq_len, contract_id
        )
        X_test, y_gt_test, y_garch_test, contract_id_test = _samples(
            residuals, close, sigma2, split.test_starts, config.seq_len, contract_id
        )
        _validate_arrays(
            {
                "X_train": X_train,
                "y_gt_train": y_gt_train,
                "y_garch_train": y_garch_train,
                "contract_id_train": contract_id_train,
                "X_test": X_test,
                "y_gt_test": y_gt_test,
                "y_garch_test": y_garch_test,
                "contract_id_test": contract_id_test,
            },
            config,
        )
    except ContractPreparationError:
        raise
    except ValueError as exc:
        reason = str(exc).split(":", 1)[0]
        raise ContractPreparationError(filename, reason, str(exc)) from exc
    except Exception as exc:
        raise ContractPreparationError(filename, exc.__class__.__name__, str(exc)) from exc
    return PreparedContract(
        X_train=X_train,
        y_gt_train=y_gt_train,
        y_garch_train=y_garch_train,
        contract_id_train=contract_id_train,
        X_test=X_test,
        y_gt_test=y_gt_test,
        y_garch_test=y_garch_test,
        contract_id_test=contract_id_test,
        train_starts=split.train_starts,
        test_starts=split.test_starts,
        ar_coefficients=coeffs,
        garch=garch,
        raw_rows=len(cleaned),
        filename=filename,
    )


def source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_id(config: GinnDataConfig, source_files: list[dict[str, Any]]) -> str:
    payload = {
        "config": config.to_dict(),
        "source_files": source_files,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _validate_arrays(arrays: dict[str, np.ndarray], config: GinnDataConfig) -> None:
    for key in ("X_train", "X_test"):
        arr = arrays[key]
        if arr.dtype != np.float32 or arr.ndim != 3 or arr.shape[1:] != (config.seq_len, 5):
            raise ValueError(f"{key} must be float32 [N, {config.seq_len}, 5]")
        if arr.shape[0] == 0:
            raise ValueError(f"{key} cannot be empty")
    for key in ("y_gt_train", "y_garch_train", "y_gt_test", "y_garch_test"):
        arr = arrays[key]
        if arr.dtype != np.float32 or arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(f"{key} must be float32 [N, 1]")
        if np.any(arr < 0.0):
            raise ValueError(f"{key} contains negative targets")
    for key in ("contract_id_train", "contract_id_test"):
        arr = arrays[key]
        if arr.dtype != np.int32 or arr.ndim != 1:
            raise ValueError(f"{key} must be int32 [N]")
    if arrays["X_train"].shape[0] != arrays["y_gt_train"].shape[0]:
        raise ValueError("train array lengths do not match")
    if arrays["X_test"].shape[0] != arrays["y_gt_test"].shape[0]:
        raise ValueError("test array lengths do not match")
    for key, arr in arrays.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"{key} contains non-finite values")


def _combine(prepared: list[PreparedContract]) -> dict[str, np.ndarray]:
    if not prepared:
        raise ValueError("No contracts were prepared")
    return {
        "X_train": np.concatenate([item.X_train for item in prepared]).astype(np.float32),
        "y_gt_train": np.concatenate([item.y_gt_train for item in prepared]).astype(np.float32),
        "y_garch_train": np.concatenate([item.y_garch_train for item in prepared]).astype(np.float32),
        "contract_id_train": np.concatenate([item.contract_id_train for item in prepared]).astype(np.int32),
        "X_test": np.concatenate([item.X_test for item in prepared]).astype(np.float32),
        "y_gt_test": np.concatenate([item.y_gt_test for item in prepared]).astype(np.float32),
        "y_garch_test": np.concatenate([item.y_garch_test for item in prepared]).astype(np.float32),
        "contract_id_test": np.concatenate([item.contract_id_test for item in prepared]).astype(np.int32),
    }


def write_cache(
    prepared: list[PreparedContract],
    skipped_contracts: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    config: GinnDataConfig,
    cache_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    cache_path = Path(cache_path)
    manifest_path = Path(manifest_path)
    arrays = _combine(prepared)
    _validate_arrays(arrays, config)
    manifest = {
        "dataset_id": _dataset_id(config, source_files),
        "config": config.to_dict(),
        "transforms": {
            "ohlc": "log first difference with clip min 1e-8",
            "volume": "log1p first difference with negative volume clipped to 0",
        },
        "targets": {
            "y_gt": "sqrt(mean(diff(log(close[start+1:start+seq_len+1]))**2))",
            "y_garch": "sqrt(mean(sigma2[start+1:start+seq_len+1]))",
        },
        "source_files": source_files,
        "contracts": [
            {
                "contract_id": index,
                "filename": item.filename,
                "raw_rows": item.raw_rows,
                "train_samples": int(item.X_train.shape[0]),
                "test_samples": int(item.X_test.shape[0]),
                "garch": {
                    "params": item.garch.params.tolist(),
                    "residual_mean": item.garch.residual_mean,
                    "initial_variance": item.garch.initial_variance,
                    "converged": item.garch.converged,
                    "used_fallback": item.garch.used_fallback,
                    "status": item.garch.status,
                },
            }
            for index, item in enumerate(prepared)
        ],
        "skipped_contracts": skipped_contracts,
        "totals": {
            "included_contracts": len(prepared),
            "skipped_contracts": len(skipped_contracts),
            "garch_fallbacks": sum(1 for item in prepared if item.garch.used_fallback),
            "train_samples": int(arrays["X_train"].shape[0]),
            "test_samples": int(arrays["X_test"].shape[0]),
        },
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "quality": {
            "non_finite_count": int(sum(np.size(value) - np.isfinite(value).sum() for value in arrays.values())),
            "negative_target_count": int(
                sum(np.sum(arrays[key] < 0.0) for key in ("y_gt_train", "y_garch_train", "y_gt_test", "y_garch_test"))
            ),
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_and_validate_cache(
    cache_path: str | Path, manifest_path: str | Path, expected_config: GinnDataConfig
) -> GinnCache:
    cache_path = Path(cache_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = expected_config.to_dict()
    actual = manifest.get("config", {})
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"manifest config mismatch for {key}: expected {value}, got {actual.get(key)}")
    with np.load(cache_path) as data:
        arrays = {key: data[key] for key in ARRAY_KEYS}
    _validate_arrays(arrays, expected_config)
    for key, arr in arrays.items():
        recorded = manifest.get("arrays", {}).get(key)
        if recorded != {"shape": list(arr.shape), "dtype": str(arr.dtype)}:
            raise ValueError(f"manifest array mismatch for {key}")
    return GinnCache(manifest=manifest, **arrays)
