# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Experimental tile choices for long Iluvatar FlashAttention prefill.

This module only controls launch parameters of the existing FlagGems Triton
flash_varlen_fwd_kernel. It does not change model settings, attention dispatch,
or benchmark implementation. Baseline remains the default until profiled.
"""

import os

ENV_NAME = "FLAGGEMS_ILUVATAR_PREFILL_TILE"
MIN_PREFILL_Q = 8192

# (query rows per CTA, key/value rows per iteration).
# 128x32 is the existing Iluvatar long-prefill configuration.
TILE_CANDIDATES = {
    "64x32": (64, 32),
    "64x64": (64, 64),
    "128x32": (128, 32),
    "128x64": (128, 64),
    "128x128": (128, 128),
    "256x32": (256, 32),
}


def select_prefill_tiles(
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_size: int,
    is_dropout: bool,
    is_local: bool,
    mode: str | None = None,
) -> tuple[int, int] | None:
    """Return (BLOCK_M, BLOCK_N) for eligible long prefills, else None.

    No changes are made to decode, short prompts, local attention, dropout,
    non-Iluvatar paths, or to the existing configuration by default.

    The 'auto' policy is a *candidate hypothesis*, not a proven optimum.
    Validate it with the local BI-V150 sweep and math_500 before enabling.
    """
    if (
        max_seqlen_q < MIN_PREFILL_Q
        or max_seqlen_k < MIN_PREFILL_Q
        or head_size <= 0
        or head_size > 256
        or is_dropout
        or is_local
    ):
        return None

    chosen = (os.environ.get(ENV_NAME, "baseline") if mode is None else mode).strip().lower()
    if chosen == "baseline":
        return None
    if chosen == "auto":
        # Longer queries amortize each K/V tile across more Q rows, but
        # growing the Q accumulator can raise registers and lower occupancy.
        if head_size <= 64 and max_seqlen_q >= 16384:
            chosen = "128x64"
        elif head_size <= 128:
            chosen = "64x64"
        else:
            chosen = "64x32"

    if chosen not in TILE_CANDIDATES:
        raise ValueError(
            f"Unsupported {ENV_NAME}={chosen!r}; expected baseline, auto, "
            + ", ".join(TILE_CANDIDATES)
        )

    block_m, block_n = TILE_CANDIDATES[chosen]
    # Do not attempt extra-wide accumulators on 256-dimensional heads.
    if block_m * head_size > 32768:
        raise ValueError(
            f"{chosen} with head_size={head_size} exceeds the experimental "
            "Q accumulator guard (32768 elements per CTA)"
        )
    return block_m, block_n

def unwrap_libentry_kernel(launch_result):
    """LibEntry launches return (CompiledKernel, constexprs); plain Triton may not."""
    return launch_result[0] if isinstance(launch_result, tuple) else launch_result

# Opt-in actual kernel memory-access refinement, independent of tile tuning.
PAGED_KV_ENV = "FLAGGEMS_ILUVATAR_PAGED_KV"


def select_grouped_page_lookup(
    *,
    is_paged: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: int,
    block_n: int,
    is_dropout: bool,
    is_local: bool,
    mode: str | None = None,
) -> bool:
    """Use one page-table load per page rather than one per token.

    Only supports power-of-two pages aligned with KV tiles. In addition to a
    full 16K prompt, 2K query chunks with a long existing context are eligible.
    Baseline remains the default; this does not change backend selection.
    """
    chosen = (
        os.environ.get(PAGED_KV_ENV, "baseline") if mode is None else mode
    ).strip().lower()
    if chosen not in ("baseline", "grouped"):
        raise ValueError(
            f"Unsupported {PAGED_KV_ENV}={chosen!r}; expected baseline or grouped"
        )
    if chosen == "baseline":
        return False
    return (
        is_paged
        and max_seqlen_q >= 128
        and max_seqlen_k >= 8192
        and not is_dropout
        and not is_local
        and block_size > 0
        and block_n >= block_size
        and block_n % block_size == 0
        and (block_size & (block_size - 1)) == 0
        and (block_n & (block_n - 1)) == 0
    )


def grouped_page_indices(tile_start: int, block_n: int, block_size: int):
    """CPU reference for physical page indices inside an aligned KV tile."""
    if block_size <= 0 or block_n <= 0 or block_n % block_size:
        raise ValueError("KV tile must contain an integer number of pages")
    if tile_start % block_size:
        raise ValueError("KV tile must start at a page boundary")
    return [tile_start // block_size + j // block_size for j in range(block_n)]
