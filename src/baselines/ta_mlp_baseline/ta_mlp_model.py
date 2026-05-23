from __future__ import annotations

import torch
import torch.nn as nn

from baselines.ta_mlp_baseline.ta_features import N_FEATURES


class TAMLPClassifier(nn.Module):
    """
    MLP classifier trained on TA-Lib technical indicator features.

    Architecture from the FreqTrade-based MLP paper:
    in_dim → 128 → 64 → 32 → n_classes, LeakyReLU activations.

    Default input dimension matches the 36-feature vector produced by
    ``compute_ta_features()`` in ``ta_features.py``.

    Parameters
    ----------
    in_dim : int
        Number of input features (default: N_FEATURES = 36).
    n_classes : int
        Number of output classes.  Use 2 for binary trend classification
        (up / down), 3 for the original BUY / HOLD / SELL formulation.
    """

    def __init__(self, in_dim: int = N_FEATURES, n_classes: int = 2) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, in_dim]

        Returns
        -------
        logits : Tensor [B, n_classes]
        """
        return self.net(x)
