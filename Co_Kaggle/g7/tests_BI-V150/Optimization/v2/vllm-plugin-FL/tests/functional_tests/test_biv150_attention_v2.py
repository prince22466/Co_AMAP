# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the BI-V150 v2 GQA prefill-kernel experiment.

GPU tests compare numerical outputs from the pinned vLLM 0.24.0 baseline
launcher and the plugin-owned tuned launcher; they do NOT replace MATH-500.
"""

import pytest
import torch


BASELINE = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
V2 = (
    "vllm_fl.dispatch.backends.vendor.iluvatar.impl."
    "attention_biv150_v2.BIV150TritonAttentionV2Backend"
)


def test_opt_in_preserves_original_default(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    monkeypatch.delenv("VLLM_FL_BIV150_ATTN_V2", raising=False)
    assert FlagGemsBackend().attention_backend() == BASELINE


def test_opt_in_selects_v2_on_biv150(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    monkeypatch.setenv("VLLM_FL_BIV150_ATTN_V2", "1")
    assert FlagGemsBackend().attention_backend() == V2


def test_opt_in_rejects_conflicting_attention_flags(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "1")
    monkeypatch.setenv("VLLM_FL_BIV150_ATTN_V2", "1")
    with pytest.raises(RuntimeError, match="cannot both be enabled"):
        FlagGemsBackend().attention_backend()


@pytest.mark.parametrize(
    ("query_len", "context_len"),
    [
        (1, 512),      # Single-token decode: unchanged launch geometry.
        (64, 0),       # Short uncached prefill.
        (127, 2048),   # Short query, long cached prefix: v1 tile.
        (128, 2048),   # Boundary of the v2 wide-tile specialization.
        (256, 0),      # Long query but total KV < 2048: v1 tile.
        (2048, 0),     # First chunk.
        (2048, 4096),  # Chunked prefill after an existing prefix.
        (2048, 14336), # 16K endpoint (expensive).
    ],
)
def test_bf16_gqa_prefill_matches_baseline(query_len, context_len):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA-compatible GPU (run on BI-V150)")

    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention as baseline_unified_attention,
    )
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.triton_unified_attention_biv150_v2 import (
        unified_attention as optimized_unified_attention,
    )

    device = "cuda"
    torch.manual_seed(2026)
    head_size = 128
    num_query_heads = 16
    num_kv_heads = 2
    kv_block_size = 16
    total_len = context_len + query_len
    num_blocks = (total_len + kv_block_size - 1) // kv_block_size

    query = torch.randn(
        (query_len, num_query_heads, head_size),
        device=device,
        dtype=torch.bfloat16,
    )
    key_cache = torch.randn(
        (num_blocks, kv_block_size, num_kv_heads, head_size),
        device=device,
        dtype=torch.bfloat16,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(
        num_blocks, device=device, dtype=torch.int32
    ).reshape(1, -1)
    cu_seqlens_q = torch.tensor([0, query_len], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([total_len], device=device, dtype=torch.int32)

    arguments = dict(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=query_len,
        seqused_k=seq_lens,
        max_seqlen_k=total_len,
        softmax_scale=head_size ** -0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    expected = torch.empty_like(query)
    actual = torch.empty_like(query)
    with torch.inference_mode():
        baseline_unified_attention(**arguments, out=expected)
        optimized_unified_attention(**arguments, out=actual)
        torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    if query_len == 1:
        # No changes to single-token decode selection or computation.
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("max_seqlen_q", "max_seqlen_k", "eligible", "expected"),
    [
        (4096, 4096, True, True),      # 4K prompt
        (16384, 16384, True, True),    # 16K prompt
        (2048, 16384, True, True),     # chunked long prefill
        (127, 4096, True, False),      # short query: retain v1 tile
        (128, 2047, True, False),      # short KV sequence: retain v1 tile
        (2048, 16384, False, False),   # non-MiniCPM/features: baseline
        (1, 16384, False, False),      # single-token decode
    ],
)
def test_v2_wide_tile_gate(max_seqlen_q, max_seqlen_k, eligible, expected):
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.triton_unified_attention_biv150_v2 import (
        _use_biv150_v2_wide_kv_tile,
    )

    assert _use_biv150_v2_wide_kv_tile(
        is_supported_gqa_prefill=eligible,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    ) is expected
