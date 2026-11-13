# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the integrated workload-specialized Triton attention backend.

GPU comparisons cover prefill and decode against pinned vLLM 0.24.0;
they do not replace official accuracy or serving benchmarks.
"""

from types import SimpleNamespace

import pytest
import torch


BASELINE = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
OPTIMIZED_BACKEND = (
    "vllm_fl.dispatch.backends."
    "triton_attention.OptimizedTritonAttentionBackend"
)
FLAGGEMS = "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"


def _mock_platform(monkeypatch, vendor_name):
    import vllm.platforms

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name=vendor_name)
    )


@pytest.mark.parametrize("vendor_name", ["iluvatar", "nvidia", "amd", None])
def test_stock_triton_selection_is_vendor_independent(monkeypatch, vendor_name):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    _mock_platform(monkeypatch, vendor_name)
    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    monkeypatch.setattr(AttentionBackendEnum, "get_path", lambda self: BASELINE)
    assert FlagGemsBackend().attention_backend() == OPTIMIZED_BACKEND


def test_registered_triton_override_is_preserved(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    _mock_platform(monkeypatch, "nvidia")
    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    override = "custom.attention.RegisteredBackend"
    monkeypatch.setattr(AttentionBackendEnum, "get_path", lambda self: override)
    assert FlagGemsBackend().attention_backend() == override


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"use_mla": True}, NotImplementedError), ({"use_sparse": True}, ValueError)],
)
def test_unsupported_attention_modes_preserve_errors(monkeypatch, kwargs, error):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    _mock_platform(monkeypatch, "nvidia")
    with pytest.raises(error):
        FlagGemsBackend().attention_backend(**kwargs)


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


def test_unified_backend_binds_integrated_kernel():
    from vllm_fl.dispatch.backends import (
        triton_attention as adapter,
        triton_unified_attention as kernel,
    )

    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionImpl

    assert issubclass(adapter.OptimizedTritonAttentionBackend, TritonAttentionBackend)
    assert issubclass(adapter.OptimizedTritonAttentionImpl, TritonAttentionImpl)
    assert adapter.OptimizedTritonAttentionBackend.get_impl_cls() is (
        adapter.OptimizedTritonAttentionImpl
    )
    assert adapter.unified_attention is kernel.unified_attention


@pytest.mark.parametrize(
    ("query_len", "context_len"),
    [
        (1, 512),      # Single-token decode with paged-KV lookup.
        (7, 17),       # Partial query tile and non-page-aligned context.
        (9, 15),       # One row beyond the eight-query specialization tile.
        (64, 0),       # Short uncached prefill.
        (65, 33),      # Query tail and partial KV-cache page.
        (2048, 0),     # First chunk.
        (2048, 4096),  # Chunked prefill after an existing prefix.
        (2048, 14336), # 16K endpoint (expensive).
    ],
)
def test_bf16_gqa_attention_matches_baseline(query_len, context_len, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA-compatible GPU")

    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention as baseline_unified_attention,
    )
    from vllm_fl.dispatch.backends import (
        triton_unified_attention as optimized_module,
    )

    # Assert kernel selection as well as numerical equivalence. Otherwise
    # accidentally disabling prefill tiling can pass a pure output test.
    launches = []
    original_kernel = optimized_module.kernel_unified_attention

    class KernelSpy:
        def __getitem__(self, grid):
            launcher = original_kernel[grid]

            def capture(**kwargs):
                launches.append((kwargs["BLOCK_M"], kwargs["DECODE_PAGE_LOOKUP"]))
                return launcher(**kwargs)

            return capture

    monkeypatch.setattr(optimized_module, "kernel_unified_attention", KernelSpy())
    optimized_unified_attention = optimized_module.unified_attention

    device = "cuda"
    torch.manual_seed(2026)
    head_size = 128
    num_query_heads = 16
    num_kv_heads = 2
    kv_block_size = 16
    total_len = context_len + query_len
    num_used_blocks = (total_len + kv_block_size - 1) // kv_block_size
    # The pinned reference loads page indices for a whole 32-token KV tile
    # before masking token loads. With 16-token pages, its final tile may
    # read one spare page-table entry (e.g. 98 tokens use seven pages).
    # Match the padded page-table contract used by the decode/mixed tests.
    num_blocks = num_used_blocks + 1

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
    assert launches == [(64 if query_len > 1 else 16, query_len == 1)]
