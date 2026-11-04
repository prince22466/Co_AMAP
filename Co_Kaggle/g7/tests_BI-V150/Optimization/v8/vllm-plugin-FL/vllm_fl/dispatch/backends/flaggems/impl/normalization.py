# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_flaggems(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using FlagGems.

    Args:
        obj: The calling obj (e.g., RMSNorm layer)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    # The v3 RMSNorm kernel is automatically used for eligible Iluvatar
    # inference shapes. Otherwise retain the original FlagGems operator.
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) == "iluvatar":
        from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization_biv150_v3 import (
            fused_rms_norm,
            is_eligible,
        )
        weight = obj.weight
        if is_eligible(x, residual, weight):
            return fused_rms_norm(
                x, residual, weight, obj.variance_epsilon,
                inplace=residual is not None,
            )

    from flag_gems.modules.normalization import gems_rms_forward

    # Get weight and epsilon from obj
    weight = obj.weight
    epsilon = obj.variance_epsilon

    return gems_rms_forward(x, residual, weight, epsilon)
