import numpy as np
from scipy.optimize import minimize


# Number of scalar features GARCH(1,1) produces per series column.
_GARCH_N_FEATURES = 7  # omega, alpha, beta, persistence, uncond_var, mean_cond_var, std_cond_var


def _ensure_2d(sequence: np.ndarray) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D [seq_len, features], got shape={arr.shape}")
    return arr


# ── AR(p) ─────────────────────────────────────────────────────────────────────

def ar_coefficients(series: np.ndarray, order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit an AR(order) model with least squares and return coefficients + residuals.
    """
    x = np.asarray(series, dtype=np.float32).reshape(-1)
    if len(x) <= order:
        raise ValueError(f"Series length must be > order ({order}), got {len(x)}")

    # Build a lagged design matrix: each column is the series shifted by 1..order.
    y = x[order:]
    lagged = [x[order - lag - 1 : -lag - 1] for lag in range(order)]
    X = np.stack(lagged, axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coeffs
    return coeffs.astype(np.float32), residuals.astype(np.float32)


# ── GARCH(1,1) ────────────────────────────────────────────────────────────────

def _garch11_negloglik(
    params: np.ndarray, eps: np.ndarray, var_init: float
) -> float:
    """Negative Gaussian log-likelihood for GARCH(1,1)."""
    omega, alpha, beta = params
    # Stationarity condition: alpha + beta < 1 and all params positive.
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        return 1e10

    T = len(eps)
    sigma2 = np.empty(T)
    sigma2[0] = var_init
    for t in range(1, T):
        # σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        if sigma2[t] <= 0:
            return 1e10

    return 0.5 * float(np.sum(np.log(sigma2) + eps ** 2 / sigma2))


def _fit_garch11(
    returns: np.ndarray, max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit GARCH(1,1) to a returns series via MLE (SLSQP).

    Returns
    -------
    params : ndarray of shape (3,)  — [omega, alpha, beta]
    sigma2 : ndarray of shape (T,)  — fitted conditional variances
    """
    eps = (returns - returns.mean()).astype(np.float64)
    var_init = max(float(np.var(eps)), 1e-10)

    # Starting point: small omega, moderate alpha, high beta (typical for financial series).
    x0 = np.array([var_init * 0.05, 0.10, 0.85])
    bounds = [(1e-10, None), (1e-8, 0.9998), (1e-8, 0.9998)]
    constraints = {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}

    try:
        result = minimize(
            _garch11_negloglik,
            x0,
            args=(eps, var_init),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1e-7},
        )
        params = result.x if result.success else x0
    except Exception:
        params = x0

    omega, alpha, beta = params

    # Clamp into valid GARCH(1,1) parameter space
    omega = max(float(omega), 1e-10)
    alpha = float(np.clip(alpha, 1e-8, 0.9998))
    beta = float(np.clip(beta, 1e-8, 0.9998))
    if alpha + beta >= 1.0:
        total = alpha + beta
        alpha, beta = alpha / total * 0.9999, beta / total * 0.9999

    # Reconstruct conditional variance series with final parameters
    T = len(eps)
    sigma2 = np.empty(T)
    sigma2[0] = var_init
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    sigma2 = np.clip(sigma2, 1e-10, None)

    return (
        np.array([omega, alpha, beta], dtype=np.float32),
        sigma2.astype(np.float32),
    )


def garch_features(series: np.ndarray, min_len: int = 20) -> np.ndarray:
    """
    GARCH(1,1) features for a 1-D series (fitted on first differences).

    Returns a vector of length 7:
        [omega, alpha, beta, persistence, uncond_var, mean_cond_var, std_cond_var]

    Returns zeros when the series is too short to fit (< min_len + 1 points).

    Parameters
    ----------
    series  : 1-D array of raw values (prices, volume, …)
    min_len : minimum number of return observations required
    """
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    if len(x) < min_len + 1:
        return np.zeros(_GARCH_N_FEATURES, dtype=np.float32)

    # GARCH is fitted on first differences (returns), not raw price levels,
    # because GARCH requires a mean-stationary input.
    returns = np.diff(x)
    params, sigma2 = _fit_garch11(returns)
    omega, alpha, beta = params

    persistence = alpha + beta                          # how long volatility shocks last
    uncond_var = omega / max(1.0 - persistence, 1e-8)  # long-run (unconditional) variance

    return np.array(
        [omega, alpha, beta, persistence, uncond_var, sigma2.mean(), sigma2.std()],
        dtype=np.float32,
    )


# ── Dimension utility ─────────────────────────────────────────────────────────

def statistical_feature_dim(n_cols: int = 5, ar_order: int = 5) -> int:
    """
    Return the output dimension of ``compute_statistical_features`` without
    running it.  Use this to size the aggregator's ``branch_dims`` entry.

    >>> statistical_feature_dim(n_cols=5, ar_order=5)
    70
    """
    return n_cols * (ar_order + 2 + _GARCH_N_FEATURES)


# ── Combined statistical representation ───────────────────────────────────────

def compute_statistical_features(sequence: np.ndarray, ar_order: int = 5) -> np.ndarray:
    """
    Build statistical representation from each feature column.

    Per column: [AR(p) coefficients | residual_mean | residual_std | GARCH(1,1) features]

    Output length = n_cols * (ar_order + 2 + _GARCH_N_FEATURES)
    """
    arr = _ensure_2d(sequence)
    outputs = []
    for col in range(arr.shape[1]):
        coeffs, residuals = ar_coefficients(arr[:, col], order=ar_order)
        # AR part: p coefficients capturing linear temporal dependency,
        # plus mean/std of residuals as a summary of unexplained variation.
        ar_part = np.concatenate([coeffs, [residuals.mean(), residuals.std()]])
        garch_part = garch_features(arr[:, col])
        outputs.append(np.concatenate([ar_part, garch_part]))
    return np.concatenate(outputs, axis=0).astype(np.float32)


def batch_statistical_features(sequences: np.ndarray, ar_order: int = 5) -> np.ndarray:
    arr = np.asarray(sequences, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D [N, seq_len, features], got shape={arr.shape}")
    return np.stack(
        [compute_statistical_features(seq, ar_order=ar_order) for seq in arr], axis=0
    )