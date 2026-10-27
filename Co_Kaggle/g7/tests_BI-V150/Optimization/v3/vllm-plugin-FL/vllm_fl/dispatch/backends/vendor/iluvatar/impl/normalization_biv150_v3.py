# SPDX-License-Identifier: Apache-2.0
"""BI-V150 v3: fused residual addition and RMS normalization.

This is an independent, optional operator-level experiment (not a change to
model precision, hyperparameters, or the vLLM attention backend).

The kernel reads one token row, optionally adds and writes its residual, then
computes RMS statistics and scales by the learned weight in one launch.
Unlike the unoptimized PyTorch reference, variance is accumulated in FP32.
"""

from __future__ import annotations

import torch


def is_eligible(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
) -> bool:
    """Conservatively limit this kernel to inference-only dense rows."""
    if (
        not isinstance(x, torch.Tensor)
        or not isinstance(weight, torch.Tensor)
        or not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or x.ndim < 1
        or x.ndim > 3
        or x.numel() == 0
        or not x.is_contiguous()
        or x.requires_grad
        or weight.requires_grad
        or weight.ndim != 1
        or not weight.is_contiguous()
        or weight.device != x.device
        or weight.dtype != x.dtype
        or x.shape[-1] != weight.numel()
        or not 128 <= x.shape[-1] <= 8192
    ):
        return False

    if residual is not None and (
        not isinstance(residual, torch.Tensor)
        or residual.shape != x.shape
        or residual.device != x.device
        or residual.dtype != x.dtype
        or not residual.is_contiguous()
        or residual.requires_grad
    ):
        return False
    return True


def fused_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the fused kernel; caller must check is_eligible first."""
    import triton
    import triton.language as tl

    # The function is lazily imported only when v3 is enabled.  Creating the
    # JIT specialization once per dtype/hidden size is expected at first use.
    @triton.jit
    def _fused_rms_kernel(
        X, R, W, Y, RES_OUT,
        H: tl.constexpr,
        BLOCK: tl.constexpr,
        EPS: tl.constexpr,
        HAS_RES: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        valid = col < H
        offsets = row * H + col

        values = tl.load(X + offsets, mask=valid, other=0).to(tl.float32)
        if HAS_RES:
            added = tl.load(R + offsets, mask=valid, other=0).to(tl.float32)
            # Preserve the residual-add result in the input dtype, matching
            # the observable tensor returned by the vLLM/FlagGems interface.
            values = (values + added).to(X.dtype.element_ty).to(tl.float32)
            tl.store(RES_OUT + offsets, values, mask=valid)

        # Padded lanes are zero, and division uses the logical hidden size.
        mean_square = tl.sum(values * values, axis=0) / H
        inv_rms = tl.rsqrt(mean_square + EPS)
        gamma = tl.load(W + col, mask=valid, other=0).to(tl.float32)
        normalized = (values * inv_rms) * gamma
        tl.store(Y + offsets, normalized, mask=valid)

    assert is_eligible(x, residual, weight)
    hidden = x.shape[-1]
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x) if residual is not None else None
    block = triton.next_power_of_2(hidden)
    warps = 4 if block <= 4096 else 8
    _fused_rms_kernel[(x.numel() // hidden,)](
        x,
        residual if residual is not None else x,
        weight,
        out,
        residual_out if residual_out is not None else out,
        hidden,
        block,
        float(epsilon),
        residual is not None,
        num_warps=warps,
    )
    return (out, residual_out) if residual_out is not None else out
