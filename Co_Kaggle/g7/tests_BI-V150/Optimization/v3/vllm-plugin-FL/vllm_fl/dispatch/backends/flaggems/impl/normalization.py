# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union
import os

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
    if os.environ.get("VLLM_FL_BIV150_RMS_V3", "0").lower() in ("1", "true", "yes"):
        from vllm.platforms import current_platform
        if getattr(current_platform, "vendor_name", None) == "iluvatar":
            from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization_biv150_v3 import (
                is_eligible, fused_rms_norm,
            )
            if is_eligible(x, residual, obj.weight):
                return fused_rms_norm(x, residual, obj.weight, obj.variance_epsilon)

    from flag_gems.modules.normalization import gems_rms_forward

    # Get weight and epsilon from obj
    weight = obj.weight
    epsilon = obj.variance_epsilon

    return gems_rms_forward(x, residual, weight, epsilon)
