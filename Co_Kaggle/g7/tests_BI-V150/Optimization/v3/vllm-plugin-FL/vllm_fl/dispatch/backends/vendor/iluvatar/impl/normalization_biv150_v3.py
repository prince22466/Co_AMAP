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
        or (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))
        or weight.ndim != 1
        or not weight.is_contiguous()
        or weight.device != x.device
        or weight.dtype != x.dtype
        or x.shape[-1] != weight.numel()
        or not 128 <= x.shape[-1] <= 4096
    ):
        return False

    if residual is not None and (
        not isinstance(residual, torch.Tensor)
        or residual.shape != x.shape
        or residual.device != x.device
        or residual.dtype != x.dtype
        or not residual.is_contiguous()
        or (torch.is_grad_enabled() and residual.requires_grad)
        # In-place FlagGems mode cannot represent both results when the
        # inputs share storage; disallow even offset-overlapping views.
        or x.untyped_storage().data_ptr() == residual.untyped_storage().data_ptr()
    ):
        return False
    return True


def fused_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    *,
    inplace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run v3. Preserve FlagGems' in-place contract when requested.

    FlagGems fused-add RMSNorm mutates x and residual, returning those same
    tensors; the vendor PyTorch implementation instead returns new tensors.
    A single kernel supports both semantics without changing dispatch.
    """
    # Triton imports and JIT specialization are cached at module level.
    # Keeping the JIT function stable avoids recompilation/dispatch overhead.
    from .normalization_biv150_v3_kernel import _fused_rms_kernel
    import triton

    assert is_eligible(x, residual, weight)
    hidden = x.shape[-1]
    out = x if inplace and residual is not None else torch.empty_like(x)
    residual_out = (
        residual if inplace else torch.empty_like(x)
    ) if residual is not None else None
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
