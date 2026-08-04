from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from features.statistical import ar_coefficients, _fit_garch11


class LSTMVariancePredictor(nn.Module):
    """
    GINN Stage 3: LSTM-based variance predictor.

    Architecture from the original GINN paper (lstm.py):
    2-layer LSTM (hidden=256) → Linear(256→128) → BatchNorm → ReLU → Linear(128→1).

    The final hidden state of the last LSTM layer is used as the sequence
    summary — no attention, no pooling.

    Parameters
    ----------
    input_size : int
        Number of input channels per timestep.
        Default 5 (all OHLCV AR-residual columns).
        Set to 1 to match the original univariate (close only) formulation.
    hidden_size : int
        LSTM hidden dimension (default 256, matching the original).
    num_layers : int
        Number of stacked LSTM layers (default 2, matching the original).
    dropout : float
        Dropout between LSTM layers (default 0.1, matching the original).
    output_transform : {"linear", "softplus"}
        Transform applied to the final scalar. ``linear`` preserves the
        original architecture. ``softplus`` constrains volatility predictions
        to be non-negative.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_transform: str = "linear",
    ) -> None:
        super().__init__()
        if output_transform not in {"linear", "softplus"}:
            raise ValueError("output_transform must be 'linear' or 'softplus'")
        self.output_transform = output_transform
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, seq_len, input_size]

        Returns
        -------
        Tensor [B, 1]  — predicted volatility
        """
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        if self.output_transform == "softplus":
            return F.softplus(out)
        return out


def garch_fused_loss(
    pred: torch.Tensor,
    y_gt: torch.Tensor,
    y_garch: torch.Tensor,
    lambda_garch: float = 0.1,
) -> dict[str, torch.Tensor]:
    """
    GINN fused loss: blends ground-truth and GARCH supervision.

        loss = (1 - λ) * MSE(pred, y_gt) + λ * MSE(pred, y_garch)

    λ_garch steers the LSTM toward the GARCH signal without fully replacing
    the ground-truth objective.  The original paper uses λ=0.1–0.3.
    """
    gt_mse = F.mse_loss(pred, y_gt)
    garch_mse = F.mse_loss(pred, y_garch)
    total = (1.0 - lambda_garch) * gt_mse + lambda_garch * garch_mse
    return {"total": total, "gt_mse": gt_mse, "garch_mse": garch_mse}


class GinnTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y_gt: np.ndarray, y_garch: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y_gt = torch.as_tensor(y_gt, dtype=torch.float32)
        self.y_garch = torch.as_tensor(y_garch, dtype=torch.float32)
        if self.X.ndim != 3:
            raise ValueError(f"X must be [N, seq_len, features], got {tuple(self.X.shape)}")
        if self.y_gt.shape != self.y_garch.shape or self.y_gt.shape != (self.X.shape[0], 1):
            raise ValueError("target arrays must both have shape [N, 1]")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.y_gt[index], self.y_garch[index]


def make_train_loader(dataset: Dataset, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) % batch_size == 1,
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_garch: float,
) -> dict[str, float]:
    model.to(device)
    model.train()
    totals = {"total": 0.0, "gt_mse": 0.0, "garch_mse": 0.0}
    count = 0
    for X, y_gt, y_garch in loader:
        X = X.to(device)
        y_gt = y_gt.to(device)
        y_garch = y_garch.to(device)
        optimizer.zero_grad()
        pred = model(X)
        losses = garch_fused_loss(pred, y_gt, y_garch, lambda_garch=lambda_garch)
        losses["total"].backward()
        optimizer.step()
        batch = int(X.shape[0])
        count += batch
        for key in totals:
            totals[key] += float(losses[key].detach().cpu().item()) * batch
    if count == 0:
        raise ValueError("training loader produced no samples")
    return {key: value / count for key, value in totals.items()}


def _safe_corr(preds: np.ndarray, targets: np.ndarray) -> float | None:
    if preds.size < 2 or targets.size < 2:
        return None
    if np.std(preds) == 0.0 or np.std(targets) == 0.0:
        return None
    return float(np.corrcoef(preds, targets)[0, 1])


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _per_contract_metrics(
    preds: np.ndarray, targets: np.ndarray, contract_ids: np.ndarray
) -> list[dict[str, float | int | None]]:
    rows = []
    for contract_id in np.unique(contract_ids):
        mask = contract_ids == contract_id
        p = preds[mask]
        y = targets[mask]
        rows.append(
            {
                "contract_id": int(contract_id),
                "n": int(mask.sum()),
                "mse": float(np.mean((p - y) ** 2)),
                "pearson_corr": _safe_corr(p, y),
            }
        )
    return rows


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y_gt: np.ndarray,
    y_garch: np.ndarray,
    contract_ids: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, object]:
    dataset = GinnTensorDataset(X, y_gt, y_garch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_X, _, _ in loader:
            preds.append(model(batch_X.to(device)).cpu().numpy())
    pred = np.concatenate(preds, axis=0).reshape(-1).astype(np.float32)
    target = np.asarray(y_gt, dtype=np.float32).reshape(-1)
    garch_target = np.asarray(y_garch, dtype=np.float32).reshape(-1)
    ids = np.asarray(contract_ids, dtype=np.int32).reshape(-1)
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("predictions and targets must be finite")
    errors = pred - target
    garch_errors = garch_target - target
    metrics = {
        "mse": float(np.mean(errors**2)),
        "pearson_corr": _safe_corr(pred, target),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "garch_mse": float(np.mean(garch_errors**2)),
        "garch_pearson_corr": _safe_corr(garch_target, target),
        "negative_prediction_count": int(np.sum(pred < 0.0)),
        "negative_prediction_fraction": float(np.mean(pred < 0.0)),
        "prediction_stats": _summary_stats(pred),
        "target_stats": _summary_stats(target),
        "error_stats": _summary_stats(errors),
        "per_contract": _per_contract_metrics(pred, target, ids),
    }
    return {
        "predictions": pred,
        "targets": target,
        "garch_targets": garch_target,
        "contract_ids": ids,
        "metrics": metrics,
    }


def build_ginn_dataset(
    feather_paths: list[str],
    seq_len: int = 64,
    ar_order: int = 5,
    train_ratio: float = 0.8,
    close_col: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build GINN train/test arrays from Polymarket OHLCV feather files.

    Pipeline per contract:
    1. Compute log-returns for all 5 OHLCV columns.
    2. Fit AR(ar_order) on each column → per-column residuals.
    3. Fit GARCH(1,1) on the close-column residuals → conditional variance series.
    4. Slide a window of length ``seq_len`` over the residuals to build X.
    5. For each window, the targets are the next window's close-column
       realised volatility (y_gt) and mean GARCH conditional variance (y_garch).
    6. Chronological 80/20 split before merging across contracts.

    Returns
    -------
    X_train, y_gt_train, y_garch_train,
    X_test,  y_gt_test,  y_garch_test  — all float32 np.ndarray

    X shape : [N, seq_len, 5]
    y shapes: [N, 1]
    """
    Xs_tr, ygt_tr, ygarch_tr = [], [], []
    Xs_te, ygt_te, ygarch_te = [], [], []

    for path in feather_paths:
        try:
            df = pd.read_feather(path).ffill().interpolate()
        except Exception:
            continue

        if len(df) < ar_order + seq_len * 2 + 2:
            continue

        prices = df[["open", "high", "low", "close", "volume"]].values.astype(np.float64)

        # Log-returns per column (length = T-1)
        log_ret = np.log(np.clip(prices[1:], 1e-10, None)) - np.log(np.clip(prices[:-1], 1e-10, None))
        log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)

        # AR residuals per column — shape [T-1-ar_order, 5]
        residuals_cols = []
        for col in range(log_ret.shape[1]):
            try:
                _, resid = ar_coefficients(log_ret[:, col], order=ar_order)
            except Exception:
                resid = log_ret[ar_order:, col]
            residuals_cols.append(resid)

        residuals = np.stack(residuals_cols, axis=1).astype(np.float32)

        # GARCH(1,1) on close-column residuals → sigma2 series
        try:
            _, sigma2 = _fit_garch11(residuals[:, close_col].astype(np.float64))
            sigma2 = sigma2.astype(np.float32)
        except Exception:
            sigma2 = (residuals[:, close_col] ** 2)

        # Slide windows: X[i] = residuals[i : i+seq_len], targets from [i+seq_len : i+2*seq_len]
        T = len(residuals)
        max_i = T - 2 * seq_len
        if max_i <= 0:
            continue

        X_all, ygt_all, ygarch_all = [], [], []
        for i in range(max_i):
            x_window = residuals[i : i + seq_len]
            next_resid = residuals[i + seq_len : i + 2 * seq_len, close_col]
            next_sigma2 = sigma2[i + seq_len : i + 2 * seq_len]

            # Realised volatility of close residuals in next window
            y_gt_val = float(np.sqrt(np.mean(next_resid ** 2)))
            # Mean GARCH conditional variance in next window
            y_garch_val = float(np.mean(next_sigma2))

            X_all.append(x_window)
            ygt_all.append(y_gt_val)
            ygarch_all.append(y_garch_val)

        if len(X_all) < 10:
            continue

        X_all = np.stack(X_all).astype(np.float32)
        ygt_all = np.array(ygt_all, dtype=np.float32).reshape(-1, 1)
        ygarch_all = np.array(ygarch_all, dtype=np.float32).reshape(-1, 1)

        split = int(len(X_all) * train_ratio)
        Xs_tr.append(X_all[:split]);     ygt_tr.append(ygt_all[:split]);     ygarch_tr.append(ygarch_all[:split])
        Xs_te.append(X_all[split:]);     ygt_te.append(ygt_all[split:]);     ygarch_te.append(ygarch_all[split:])

    if not Xs_tr:
        raise RuntimeError("No valid feather files produced data for GINN.")

    return (
        np.concatenate(Xs_tr),     np.concatenate(ygt_tr),     np.concatenate(ygarch_tr),
        np.concatenate(Xs_te),     np.concatenate(ygt_te),     np.concatenate(ygarch_te),
    )
