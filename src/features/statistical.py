import numpy as np


def _ensure_2d(sequence: np.ndarray) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D [seq_len, features], got shape={arr.shape}")
    return arr


def ar_coefficients(series: np.ndarray, order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit an AR(order) model with least squares and return coefficients + residuals.
    """
    x = np.asarray(series, dtype=np.float32).reshape(-1)
    if len(x) <= order:
        raise ValueError(f"Series length must be > order ({order}), got {len(x)}")

    y = x[order:]
    lagged = [x[order - lag - 1 : -lag - 1] for lag in range(order)]
    X = np.stack(lagged, axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coeffs
    return coeffs.astype(np.float32), residuals.astype(np.float32)


def compute_statistical_features(sequence: np.ndarray, ar_order: int = 5) -> np.ndarray:
    """
    Build statistical representation from each feature column:
    [AR coeffs..., residual_mean, residual_std]
    """
    arr = _ensure_2d(sequence)
    outputs = []
    for col in range(arr.shape[1]):
        coeffs, residuals = ar_coefficients(arr[:, col], order=ar_order)
        outputs.append(np.concatenate([coeffs, [residuals.mean(), residuals.std()]], axis=0))
    return np.concatenate(outputs, axis=0).astype(np.float32)


def batch_statistical_features(sequences: np.ndarray, ar_order: int = 5) -> np.ndarray:
    arr = np.asarray(sequences, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D [N, seq_len, features], got shape={arr.shape}")
    return np.stack([compute_statistical_features(seq, ar_order=ar_order) for seq in arr], axis=0)
