# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-aware v1 prefill + v5 pure-decode Triton attention on BI-V150.

Retains the existing vLLM Triton backend interface, metadata, cache writes,
and operator priorities. The v1 and v5 launchers remain intact and separate.
The decision depends only on the invocation's query-length metadata; mixed
prefill/decode always takes v1, not the v5-only decode specialization.
"""

import torch

from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionMetadata,
)

from .attention_biv150 import BIV150TritonAttentionImpl
from .attention_decode_v5 import BIV150DecodeAttentionImpl


class BIV150V1V5AttentionImpl(BIV150TritonAttentionImpl):
    """Route pure decode to v5 and all other calls to unchanged v1."""

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
        if attn_metadata is not None and attn_metadata.max_query_len == 1:
            return BIV150DecodeAttentionImpl.forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )


class BIV150V1V5AttentionBackend(TritonAttentionBackend):
    """Reuse vLLM Triton attention metadata and KV-cache update behavior."""

    @staticmethod
    def get_impl_cls() -> type[BIV150V1V5AttentionImpl]:
        return BIV150V1V5AttentionImpl
