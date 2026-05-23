from __future__ import annotations

import torch
import torch.nn as nn


class RawOHLCVMLP(nn.Module):
    """
    Baseline encoder: flattens raw OHLCV sequences and maps to a fixed-dim representation.

    Accepts [B, seq_len, n_features] tensors loaded directly from processed .npz files
    (no feature extraction step).  The ``output_dim`` property mirrors
    ``RepresentationAggregator.output_dim`` so the same task heads
    (``PriceRegressor``, ``VolatilityRegressor``, ``TrendClassifier``) attach without
    modification::

        encoder   = RawOHLCVMLP(seq_len=64)
        task_head = PriceRegressor(input_dim=encoder.output_dim)

    Unlike the framework encoders this model is trained **end-to-end** with the task
    head — there is no unsupervised pretraining step.

    Parameters
    ----------
    seq_len : int
        Length of each input sequence.
    n_features : int
        Number of OHLCV columns (default 5).
    hidden_dims : list[int]
        Width of each hidden layer.  Depth = ``len(hidden_dims)``.
        Default gives 5 hidden layers: [512, 512, 256, 256, 128].
    output_dim : int
        Dimension of the produced representation vector.
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int = 5,
        hidden_dims: list[int] | None = None,
        output_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 256, 256, 128]

        self._output_dim = output_dim

        dims = [seq_len * n_features] + hidden_dims
        layers: list[nn.Module] = [nn.Flatten()]
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], output_dim), nn.ReLU()]

        self.encoder = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        """Output feature dimension — use to size the attached task head."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, seq_len, n_features]

        Returns
        -------
        Tensor [B, output_dim]
        """
        return self.encoder(x)
