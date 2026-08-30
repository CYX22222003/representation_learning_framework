from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - exercised when scikit-learn is installed.
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import RobustScaler
except ModuleNotFoundError:  # pragma: no cover - tested indirectly in this environment.
    class RobustScaler:  # type: ignore[no-redef]
        def fit(self, x: np.ndarray) -> "RobustScaler":
            arr = np.asarray(x, dtype=np.float64)
            self.center_ = np.median(arr, axis=0)
            q25, q75 = np.percentile(arr, [25.0, 75.0], axis=0)
            self.scale_ = np.where((q75 - q25) == 0.0, 1.0, q75 - q25)
            return self

        def transform(self, x: np.ndarray) -> np.ndarray:
            return (np.asarray(x, dtype=np.float64) - self.center_) / self.scale_

    class ElasticNet:  # type: ignore[no-redef]
        def __init__(
            self,
            *,
            alpha: float,
            l1_ratio: float,
            fit_intercept: bool,
            max_iter: int,
            selection: str,
        ) -> None:
            self.alpha = float(alpha)
            self.l1_ratio = float(l1_ratio)
            self.fit_intercept = bool(fit_intercept)
            self.max_iter = int(max_iter)
            self.selection = selection

        def fit(self, x: np.ndarray, y: np.ndarray) -> "ElasticNet":
            arr = np.asarray(x, dtype=np.float64)
            target = np.asarray(y, dtype=np.float64).reshape(-1)
            if self.fit_intercept:
                x_mean = np.mean(arr, axis=0)
                y_mean = float(np.mean(target))
                xc = arr - x_mean
                yc = target - y_mean
            else:
                x_mean = np.zeros(arr.shape[1], dtype=np.float64)
                y_mean = 0.0
                xc = arr
                yc = target
            coef = np.zeros(arr.shape[1], dtype=np.float64)
            l1 = self.alpha * self.l1_ratio
            l2 = self.alpha * (1.0 - self.l1_ratio)
            n = max(arr.shape[0], 1)
            for iteration in range(1, self.max_iter + 1):
                old = coef.copy()
                for j in range(arr.shape[1]):
                    residual = yc - xc @ coef + xc[:, j] * coef[j]
                    rho = float(np.dot(xc[:, j], residual) / n)
                    denom = float(np.dot(xc[:, j], xc[:, j]) / n + l2)
                    if rho < -l1:
                        coef[j] = (rho + l1) / denom
                    elif rho > l1:
                        coef[j] = (rho - l1) / denom
                    else:
                        coef[j] = 0.0
                if np.max(np.abs(coef - old)) < 1e-10:
                    break
            self.coef_ = coef
            self.intercept_ = float(y_mean - np.dot(x_mean, coef)) if self.fit_intercept else 0.0
            self.n_iter_ = iteration
            self.dual_gap_ = float(np.mean((yc - xc @ coef) ** 2))
            return self


FEATURE_NAMES = ("garch", "lstm", "interaction")


def build_meta_features(garch_prediction: np.ndarray, lstm_prediction: np.ndarray) -> np.ndarray:
    g = np.asarray(garch_prediction, dtype=np.float64).reshape(-1)
    l = np.asarray(lstm_prediction, dtype=np.float64).reshape(-1)
    if g.shape != l.shape:
        raise ValueError("garch and lstm predictions must have the same shape")
    z = np.column_stack([g, l, g * l])
    if not np.all(np.isfinite(z)):
        raise ValueError("meta-features contain non-finite values")
    return z


def clipped_predictions(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("predictions contain non-finite values")
    return np.maximum(values, 0.0)


@dataclass(frozen=True)
class StackingMetaModel:
    center: np.ndarray
    scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    alpha: float
    l1_ratio: float
    max_iter: int
    n_iter: int
    dual_gap: float

    @classmethod
    def fit(
        cls,
        garch_oof: np.ndarray,
        lstm_oof: np.ndarray,
        targets: np.ndarray,
        *,
        alpha: float = 1e-4,
        l1_ratio: float = 0.5,
        max_iter: int = 10000,
    ) -> "StackingMetaModel":
        z = build_meta_features(garch_oof, lstm_oof)
        y = np.asarray(targets, dtype=np.float64).reshape(-1)
        if z.shape[0] != y.shape[0]:
            raise ValueError("meta-feature and target row counts differ")
        if not np.all(np.isfinite(y)):
            raise ValueError("targets contain non-finite values")
        scaler = RobustScaler().fit(z)
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=max_iter,
            selection="cyclic",
        ).fit(scaler.transform(z), y)
        return cls(
            center=np.asarray(scaler.center_, dtype=np.float64),
            scale=np.asarray(scaler.scale_, dtype=np.float64),
            intercept=float(model.intercept_),
            coefficients=np.asarray(model.coef_, dtype=np.float64),
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            max_iter=int(max_iter),
            n_iter=int(model.n_iter_),
            dual_gap=float(model.dual_gap_),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "StackingMetaModel":
        return cls(
            center=np.asarray(payload["center"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            intercept=float(payload["intercept"]),
            coefficients=np.asarray(
                [payload["coefficients"][name] for name in FEATURE_NAMES],  # type: ignore[index]
                dtype=np.float64,
            ),
            alpha=float(payload["alpha"]),
            l1_ratio=float(payload["l1_ratio"]),
            max_iter=int(payload["max_iter"]),
            n_iter=int(payload["n_iter"]),
            dual_gap=float(payload["dual_gap"]),
        )

    def transform(self, z: np.ndarray) -> np.ndarray:
        features = np.asarray(z, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != 3:
            raise ValueError("meta-features must be shaped [N, 3]")
        if not np.all(np.isfinite(features)):
            raise ValueError("meta-features contain non-finite values")
        scale = np.where(self.scale == 0.0, 1.0, self.scale)
        return (features - self.center) / scale

    def predict_raw(self, garch_prediction: np.ndarray, lstm_prediction: np.ndarray) -> np.ndarray:
        z = build_meta_features(garch_prediction, lstm_prediction)
        return (self.transform(z) @ self.coefficients + self.intercept).astype(np.float64)

    def predict_nonnegative(self, garch_prediction: np.ndarray, lstm_prediction: np.ndarray) -> np.ndarray:
        return clipped_predictions(self.predict_raw(garch_prediction, lstm_prediction))

    def scaler_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "scaler": "RobustScaler",
            "fit_scope": "OOF training meta-features only",
        }

    def model_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "intercept": self.intercept,
            "coefficients": {name: float(value) for name, value in zip(FEATURE_NAMES, self.coefficients)},
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "selection": "cyclic",
            "n_iter": self.n_iter,
            "dual_gap": self.dual_gap,
            "coefficient_scope": "global per fitted epoch budget, not adaptive per-sample weights",
        }


def clipping_diagnostics(raw: np.ndarray, clipped: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    raw_arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    clipped_arr = np.asarray(clipped, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if not (raw_arr.shape == clipped_arr.shape == y.shape):
        raise ValueError("raw, clipped, and targets must have matching shapes")
    return {
        "raw_negative_fraction": float(np.mean(raw_arr < 0.0)),
        "clipped_row_count": int(np.sum(raw_arr < 0.0)),
        "raw_mse": float(np.mean((raw_arr - y) ** 2)),
        "nonnegative_mse": float(np.mean((clipped_arr - y) ** 2)),
        "clipping_mse_delta": float(np.mean((clipped_arr - y) ** 2) - np.mean((raw_arr - y) ** 2)),
    }
