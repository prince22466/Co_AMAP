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
from .kv_cache_v6 import try_write_kv_cache_v6


class BIV150V1V5AttentionImpl(BIV150TritonAttentionImpl):
    """Route pure decode to v5, prefill to v1, and eligible KV writes to v6."""

    def do_kv_cache_update(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        # Keep vLLM's encoder, per-token-head quant, HND, FP8 and small
        # write behavior. v6 specializes only an existing Triton KV write.
        from vllm.v1.attention.backend import AttentionType

        if (
            self.attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER)
            and not self._is_per_token_head_quant
            and try_write_kv_cache_v6(
                key, value, kv_cache, slot_mapping, self.kv_cache_dtype
            )
        ):
            return
        return super().do_kv_cache_update(
            layer, key, value, kv_cache, slot_mapping
        )

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
