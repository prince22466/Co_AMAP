# SPDX-License-Identifier: Apache-2.0
"""Lazy-imported Triton kernel for the independent BI-V150 v3 RMSNorm path."""

import triton
import triton.language as tl


# The kernel is imported only when v3 is enabled.  Creating the
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
        # FlagGems computes RMS from the FP32 sum. It rounds the sum
        # *only when storing* the externally observable residual tensor.
        values = values + added
        tl.store(RES_OUT + offsets, values, mask=valid)

    # Padded lanes are zero, and division uses the logical hidden size.
    mean_square = tl.sum(values * values, axis=0) / H
    inv_rms = tl.rsqrt(mean_square + EPS)
    gamma = tl.load(W + col, mask=valid, other=0).to(tl.float32)
    normalized = values * inv_rms
    if not HAS_RES:
        # Match FlagGems rms_norm: normalize, round to the input dtype,
        # then multiply by the weight (rather than rounding only at store).
        normalized = normalized.to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + offsets, normalized * gamma, mask=valid)
