"""Lazy boundary around the pinned AGPL-3.0 ``xlstm==1.0.3`` dependency."""

from __future__ import annotations

from typing import Literal

from torch import nn

from .config import SOURCE_CONTRACT, XLSTMMixerConfig


XLSTMBackend = Literal["vanilla", "cuda"]


def build_xlstm_stack(config: XLSTMMixerConfig, backend: XLSTMBackend) -> nn.Module:
    """Construct the source-aligned sLSTM-only block stack.

    Import remains lazy so repository tooling and unit tests can inspect the
    project adapter before the separately gated dependency is installed.
    """

    if backend not in ("vanilla", "cuda"):
        raise ValueError("backend must be 'vanilla' or 'cuda'")
    try:
        import xlstm
        from xlstm import (
            sLSTMBlockConfig,
            sLSTMLayerConfig,
            xLSTMBlockStack,
            xLSTMBlockStackConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "XM-MV8 requires the separately gated xlstm==1.0.3 dependency; "
            "run the Phase 6.9 runtime audit before training"
        ) from exc

    resolved_version = getattr(xlstm, "__version__", None)
    if resolved_version != SOURCE_CONTRACT.xlstm_version:
        raise RuntimeError(
            "xLSTM dependency mismatch: expected "
            f"{SOURCE_CONTRACT.xlstm_version}, found {resolved_version!r}"
        )

    slstm_layer = sLSTMLayerConfig(
        num_heads=config.num_heads,
        conv1d_kernel_size=config.conv1d_kernel_size,
        backend=backend,
        dtype="float32",
        dtype_b="float32",
        dtype_r="float32",
        dtype_w="float32",
        dtype_g="float32",
        dtype_s="float32",
        dtype_a="float32",
        enable_automatic_mixed_precision=False,
    )
    slstm_block = sLSTMBlockConfig(slstm=slstm_layer)
    stack_config = xLSTMBlockStackConfig(
        mlstm_block=None,
        slstm_block=slstm_block,
        context_length=config.num_variates + config.num_initial_tokens,
        num_blocks=config.num_blocks,
        embedding_dim=config.embedding_dim,
        add_post_blocks_norm=True,
        bias=True,
        dropout=config.dropout,
        slstm_at="all",
    )
    return xLSTMBlockStack(stack_config)
