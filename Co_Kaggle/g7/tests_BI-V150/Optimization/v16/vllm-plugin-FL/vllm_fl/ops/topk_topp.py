# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded row-parallel launch for the vLLM 0.24.0 Triton top-k/top-p kernel.

PR #261 measured the original kernel with 16 programs on BI-V150.
Expose up to four programs per compute unit instead, with corresponding
per-program scratch. The upstream kernel, lookup tables and sampler
selection are unchanged. Expanded-grid scratch is private to each call.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Exact source used in PR #261 (also the upstream v0.24.0 release).
_SUPPORTED_SOURCE_SHA256 = (
    "64c66699cd398c2eafe9eb91f0e189162f83a6f83a209651048e163d7d6fcee8"
)


def _build_wrapper(upstream):
    original = upstream.apply_top_k_top_p_triton

    def apply_top_k_top_p_triton(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        mask_value: float = float("-inf"),
    ) -> torch.Tensor:
        # Only change the measured GPU top-p path (including combined k/p).
        # Small batches, top-k alone, invalid/unsupported layouts and other
        # devices continue through the original wrapper.
        if (
            p is None
            or logits.ndim != 2
            or logits.dtype != torch.float32
            or logits.device.type != "cuda"
            or logits.shape[0] == 0
            or logits.shape[1] == 0
            or logits.stride(1) != 1
            or logits.stride(0) < logits.shape[1]
            or p.ndim != 1
            or p.shape[0] != logits.shape[0]
            or p.device != logits.device
            or p.stride(0) != 1
            or (k is not None and (
                k.ndim != 1 or k.shape[0] != logits.shape[0]
                or k.device != logits.device or k.stride(0) != 1
            ))
        ):
            return original(logits, k, p, mask_value)

        batch_size, vocab_size = logits.shape
        num_sm = upstream.num_compute_units(logits.device.index)
        if num_sm <= 0 or batch_size <= num_sm:
            return original(logits, k, p, mask_value)

        max_programs = 4 * num_sm
        num_programs = min(max_programs, batch_size)
        k_ptr = k.to(torch.int32) if k is not None else logits
        p_ptr = p.to(torch.float32)

        # Scratch is private to this launch. PyTorch's caching allocator
        # reuses storage in stream order and owns allocations made during
        # graph capture. Do not retain a shared mutable tensor across streams
        # or graphs, or replace an address captured by an older graph.
        # Allocate only the rows actually used; no extra persistent cache.
        buffer = logits.new_empty((num_programs, vocab_size))

        tables = upstream._TRITON_TABLE_CACHE.get(logits.device)
        if tables is None:
            normal_cdf_to_sigma_table = logits.new_tensor(
                upstream._NORMAL_CDF_TO_SIGMA_TABLE
            )
            percentile_to_std_table = logits.new_tensor(
                upstream._PERCENTILE_TO_STD_TABLE
            )
            upstream._TRITON_TABLE_CACHE[logits.device] = (
                normal_cdf_to_sigma_table,
                percentile_to_std_table,
            )
        else:
            normal_cdf_to_sigma_table, percentile_to_std_table = tables

        upstream._topk_topp_kernel[(num_programs,)](
            logits,
            logits.stride(0),
            buffer,
            percentile_to_std_table,
            normal_cdf_to_sigma_table,
            k_ptr,
            p_ptr,
            BATCH_SIZE=batch_size,
            MASK_VALUE=mask_value,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=8192,
            BLOCK_SIZE_TRUNC=4096,
            TOPK_ENABLED=k is not None,
            TOPP_ENABLED=True,
        )
        return logits

    apply_top_k_top_p_triton._fl_row_grid_original = original
    return apply_top_k_top_p_triton


def install_top_p_row_grid() -> bool:
    """Replace only the native sampler's imported Triton wrapper.

    Unknown vLLM source or an existing replacement is left untouched.
    The upstream module itself remains the original reference. No installed
    files are edited and no sampler backend/selection rules are changed.
    """
    from vllm.v1.sample.ops import topk_topp_sampler as sampler
    from vllm.v1.sample.ops import topk_topp_triton as upstream

    original = upstream.apply_top_k_top_p_triton
    selected = getattr(sampler, "apply_top_k_top_p_triton", None)
    if getattr(selected, "_fl_row_grid_original", None) is original:
        return True
    if selected is not original:
        logger.debug("Top-p row grid: sampler uses a different Triton wrapper")
        return False

    try:
        digest = hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest()
    except (OSError, TypeError, AttributeError):
        logger.warning("Top-p row grid: cannot verify upstream source")
        return False
    if digest != _SUPPORTED_SOURCE_SHA256:
        logger.warning("Top-p row grid: unsupported upstream source SHA-256 %s", digest)
        return False

    sampler.apply_top_k_top_p_triton = _build_wrapper(upstream)
    logger.info("Top-p row grid: installed bounded 4x-CU Triton wrapper")
    return True
