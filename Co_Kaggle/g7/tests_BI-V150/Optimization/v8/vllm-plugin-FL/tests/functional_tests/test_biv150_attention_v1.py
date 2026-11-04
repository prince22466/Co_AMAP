# SPDX-License-Identifier: Apache-2.0
"""Regression checks for preserved v1 attention and automatic v1+v4 selection.

GPU tests compare vLLM 0.24.0 against the preserved standalone v1 kernel;
they do NOT replace MATH-500 accuracy evaluation.
"""

from types import SimpleNamespace

import pytest
import torch


BASELINE = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
V1_V4 = (
    "vllm_fl.dispatch.backends.vendor.iluvatar.impl."
    "attention_biv150_grouped_v4.BIV150V1GroupedV4AttentionBackend"
)
FLAGGEMS = "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"


def _mock_platform(monkeypatch, vendor_name):
    import vllm.platforms

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name=vendor_name)
    )


def test_selects_v1_v4_automatically_on_iluvatar(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    _mock_platform(monkeypatch, "iluvatar")
    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    monkeypatch.delenv("VLLM_FL_BIV150_ATTN_V1", raising=False)
    assert FlagGemsBackend().attention_backend() == V1_V4


def test_other_platform_preserves_triton_fallback(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    _mock_platform(monkeypatch, "nvidia")
    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    assert FlagGemsBackend().attention_backend() == BASELINE


def test_explicit_flaggems_override_still_takes_precedence(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    _mock_platform(monkeypatch, "iluvatar")
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "1")
    assert FlagGemsBackend().attention_backend() == FLAGGEMS


def test_no_gpu_still_raises(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        FlagGemsBackend().attention_backend()


@pytest.mark.parametrize(
    ("query_len", "context_len"),
    [
        (1, 512),      # Single-token decode: unchanged launch geometry.
        (64, 0),       # Short uncached prefill.
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
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.triton_unified_attention_biv150 import (
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
