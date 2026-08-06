from __future__ import annotations

import torch
import torch.nn as nn


class RepresentationAggregator(nn.Module):
    """
    Fuses an arbitrary number of named representation branches into a single embedding.

    Two modes
    ---------
    ``"concat"`` (default)
        Branches are concatenated along the feature dimension.
        Output dim = sum of all branch dims.  No learnable parameters.

    ``"gated"``
        Each branch is projected to ``out_dim``, then a gating network produces
        per-branch softmax weights from the concatenated projections.
        Output dim = ``out_dim``.

    Parameters
    ----------
    branch_dims : dict[str, int]
        Ordered mapping of branch name → input feature dimension.
        Example::

            {
                "statistical":  70,
                "transformed":  55,
                "vae":          64,
                "contrastive": 128,
            }

    out_dim : int
        Shared projection / output dimension.  Only used when ``mode="gated"``.
    mode : str
        ``"concat"`` or ``"gated"``.

    Usage
    -----
    >>> # concat mode (default)
    >>> agg = RepresentationAggregator({"statistical": 70, "transformed": 55, "vae": 64})
    >>> emb, weights = agg({"statistical": s, "transformed": t, "vae": v})
    >>> emb.shape      # [B, 189]
    >>> weights        # None

    >>> # gated mode
    >>> agg = RepresentationAggregator(
    ...     {"statistical": 70, "transformed": 55, "vae": 64},
    ...     out_dim=128, mode="gated",
    ... )
    >>> emb, weights = agg({"statistical": s, "transformed": t, "vae": v})
    >>> emb.shape      # [B, 128]
    >>> weights.shape  # [B, 3]
    """

    def __init__(
        self,
        branch_dims: dict[str, int],
        out_dim: int = 128,
        mode: str = "concat",
    ) -> None:
        super().__init__()
        if mode not in ("concat", "gated"):
            raise ValueError(f"mode must be 'concat' or 'gated', got {mode!r}")
        if len(branch_dims) < 1:
            raise ValueError("branch_dims must contain at least one branch")

        self.branch_names: list[str] = list(branch_dims.keys())
        self.mode = mode
        self._concat_dim = sum(branch_dims.values())

        if mode == "gated":
            self._out_dim = out_dim
            n = len(self.branch_names)
            self.projectors = nn.ModuleDict(
                {
                    name: nn.Sequential(nn.Linear(dim, out_dim), nn.GELU())
                    for name, dim in branch_dims.items()
                }
            )
            self.gate = nn.Sequential(
                nn.Linear(out_dim * n, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, n),
            )
        else:
            self._out_dim = self._concat_dim

    @property
    def output_dim(self) -> int:
        """Dimension of the tensor returned by ``forward``."""
        return self._out_dim

    @property
    def n_branches(self) -> int:
        return len(self.branch_names)

    def forward(
        self, branch_inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        branch_inputs : dict[str, Tensor]
            Must contain a key for every branch in ``branch_dims``.
            Each tensor has shape ``[B, branch_dim]``.

        Returns
        -------
        aggregated : Tensor  [B, output_dim]
        weights    : Tensor  [B, n_branches] in gated mode, ``None`` in concat mode
        """
        missing = [n for n in self.branch_names if n not in branch_inputs]
        if missing:
            raise ValueError(f"Missing branch inputs: {missing}")

        if self.mode == "concat":
            parts = [branch_inputs[name] for name in self.branch_names]
            return torch.cat(parts, dim=-1), None

        # gated mode
        projected = [self.projectors[name](branch_inputs[name]) for name in self.branch_names]
        combined = torch.cat(projected, dim=-1)
        weights = torch.softmax(self.gate(combined), dim=-1)
        aggregated = sum(weights[:, i : i + 1] * p for i, p in enumerate(projected))
        return aggregated, weights
