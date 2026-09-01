"""Chronological OOF utilities; the callback must fit on train_idx only."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OOFResult:
    predictions: np.ndarray
    covered: np.ndarray
    folds: tuple


def expanding_folds(n_samples: int, n_folds: int = 5, min_train_size: int | None = None,
                    embargo: int = 0):
    if n_samples < 2 or n_folds < 2 or embargo < 0:
        raise ValueError("need at least two samples/folds and non-negative embargo")
    edges = np.linspace(0, n_samples, n_folds + 1, dtype=int)
    minimum = min_train_size if min_train_size is not None else max(1, n_samples // (n_folds + 1))
    for i in range(1, len(edges)):
        predict = np.arange(edges[i - 1], edges[i])
        train_end = edges[i - 1] - embargo
        train = np.arange(0, max(0, train_end))
        if len(train) >= minimum and len(predict):
            yield train, predict


def generate_oof_predictions(n_samples, fit_predict, *, n_folds=5,
                             min_train_size=None, embargo=0):
    """Run a user-supplied fit/predict callback on expanding chronological folds."""
    folds = tuple(expanding_folds(n_samples, n_folds, min_train_size, embargo))
    if not folds:
        raise ValueError("no valid OOF folds; lower min_train_size or add data")
    pred = np.full(n_samples, np.nan, dtype=np.float64)
    covered = np.zeros(n_samples, dtype=bool)
    for train_idx, predict_idx in folds:
        if np.any(train_idx >= predict_idx[0]):
            raise AssertionError("OOF train rows must precede prediction rows")
        values = np.asarray(fit_predict(train_idx, predict_idx), dtype=np.float64).reshape(-1)
        if len(values) != len(predict_idx) or not np.all(np.isfinite(values)):
            raise ValueError("fit_predict returned invalid OOF predictions")
        if np.any(covered[predict_idx]):
            raise AssertionError("OOF prediction rows overlap")
        pred[predict_idx] = values
        covered[predict_idx] = True
    return OOFResult(pred, covered, folds)
