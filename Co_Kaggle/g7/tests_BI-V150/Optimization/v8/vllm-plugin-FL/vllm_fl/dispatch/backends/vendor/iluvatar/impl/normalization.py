# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference normalization operator implementations using PyTorch.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_iluvatar(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using PyTorch.

    Args:
        obj: The calling obj (e.g., RMSNorm layer)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    # The vendor backend retains its out-of-place contract.
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) == "iluvatar":
        from .normalization_biv150_v3 import fused_rms_norm, is_eligible
        if is_eligible(x, residual, obj.weight):
            return fused_rms_norm(x, residual, obj.weight, obj.variance_epsilon)

    # Get weight and epsilon from obj
    weight = obj.weight
    epsilon = obj.variance_epsilon

    if residual is not None:
        x = x + residual
        residual = x

    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    output = weight * x

    if residual is not None:
        return output, residual
    return output
