# SPDX-License-Identifier: Apache-2.0
"""v8 combination of prefill (v1/v2/v4) and decode (v5) Triton kernels.

Existing vLLM TritonAttentionBackend metadata, KV-cache update, and fallback
remain unchanged. Only eligible phase/configuration uses an optimized kernel.
v4 inherits from TritonAttentionImpl so its zero-arg super() falls back to
the real vLLM kernel even when invoked through this subclass.
"""
from __future__ import annotations

import os

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

from .attention_biv150 import BIV150TritonAttentionImpl
from .attention_biv150_v2 import BIV150TritonAttentionV2Impl
from .attention_grouped_v4 import BIV150GroupedPrefillAttentionImpl
from .attention_decode_v5 import BIV150DecodeAttentionImpl


def get_modes() -> tuple[str, str]:
    """Read explicit startup choices; defaults favor measured v1 prefill."""
    prefill = os.environ.get("VLLM_FL_BIV150_V8_PREFILL", "v1").lower()
    decode = os.environ.get("VLLM_FL_BIV150_V8_DECODE", "none").lower()
    if prefill not in ("none", "v1", "v2", "v4"):
        raise ValueError("VLLM_FL_BIV150_V8_PREFILL must be none, v1, v2 or v4")
    if decode not in ("none", "v5"):
        raise ValueError("VLLM_FL_BIV150_V8_DECODE must be none or v5")
    return prefill, decode


def select_phase(max_query_len: int, prefill: str, decode: str) -> str:
    """Only pure decode batches enter v5; mixed batches use prefill routing."""
    if max_query_len == 1:
        return decode
    return prefill


class BIV150StackedAttentionImpl(BIV150GroupedPrefillAttentionImpl):
    """Compose distinct already-optimized operators without backend swapping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v8_prefill_mode, self._v8_decode_mode = get_modes()

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Preserve the baseline profiling and encoder paths.
        if (
            attn_metadata is None
            or self.attn_type != AttentionType.DECODER
        ):
            return TritonAttentionImpl.forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        mode = select_phase(
            attn_metadata.max_query_len,
            self._v8_prefill_mode,
            self._v8_decode_mode,
        )
        # v5's kernel handles aligned 2D/3D decode and falls back internally
        # for unsupported shapes. It is deliberately restricted to pure decode.
        if mode == "v5":
            return BIV150DecodeAttentionImpl.forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )
        if mode == "v1":
            return BIV150TritonAttentionImpl.forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )
        if mode == "v2":
            return BIV150TritonAttentionV2Impl.forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )
        if mode == "v4":
            # This superclass contains per-sequence mixed-batch safety guards.
            return BIV150GroupedPrefillAttentionImpl.forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )
        return TritonAttentionImpl.forward(
            self, layer, query, key, value, kv_cache, attn_metadata,
            output, output_scale, output_block_scale,
        )


class BIV150StackedAttentionBackend(TritonAttentionBackend):
    """Retain original Triton metadata and KV-cache update implementation."""

    @staticmethod
    def get_impl_cls() -> type[BIV150StackedAttentionImpl]:
        return BIV150StackedAttentionImpl
