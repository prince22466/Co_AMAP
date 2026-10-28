# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in page-grouped KV-cache lookup for the existing Iluvatar varlen kernel.

This module controls an internal specialization of *the same* attention
operator. It never switches the vLLM backend, changes model hyperparameters,
or modifies the competition benchmark. The original lookup is the default.
"""

import os

ENV_NAME = "FLAGGEMS_ILUVATAR_PAGED_KV"


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
    """Return whether aligned page-grouped K/V address mapping is eligible.

    Each BLOCK_N-wide tile must contain an integral, power-of-two number of
    power-of-two-sized pages. KV tiles start at multiples of BLOCK_N, so
    every page boundary aligns with the tile. Do not touch decode/small Q,
    contiguous KV, local attention or dropout.
    """
    selected = (
        os.environ.get(ENV_NAME, "baseline") if mode is None else mode
    ).strip().lower()
    if selected not in ("baseline", "grouped"):
        raise ValueError(
            f"Unsupported {ENV_NAME}={selected!r}; expected baseline or grouped"
        )
    if selected == "baseline":
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


def grouped_virtual_page_indices(
    tile_start: int, block_n: int, block_size: int
) -> list[int]:
    """Pure-Python address oracle for tests (never called on the GPU path)."""
    if block_size <= 0 or block_n <= 0 or block_n % block_size:
        raise ValueError("KV tile must contain whole pages")
    if tile_start < 0 or tile_start % block_size:
        raise ValueError("KV tile must start at a valid page boundary")
    return [
        tile_start // block_size + lane // block_size
        for lane in range(block_n)
    ]
