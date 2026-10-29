# SPDX-License-Identifier: Apache-2.0
"""BI-V150 v5 paged-KV decode attention regression checks.

Run GPU comparisons on BI-V150. These do not replace the competition
MATH-500 accuracy test or the unchanged 4K/16K throughput benchmark.
"""
import pytest
import torch


BASELINE = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
V5 = (
    "vllm_fl.dispatch.backends.vendor.iluvatar.impl."
    "attention_decode_v5.BIV150DecodeAttentionBackend"
)


def test_default_iluvatar_uses_v5_without_enable_flag(monkeypatch):
    flag_gems = pytest.importorskip("flag_gems")
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flag_gems, "vendor_name", "iluvatar", raising=False)
    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    assert FlagGemsBackend().attention_backend() == V5


def test_other_vendors_keep_original_triton_backend(monkeypatch):
    flag_gems = pytest.importorskip("flag_gems")
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flag_gems, "vendor_name", "cuda", raising=False)
    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    assert FlagGemsBackend().attention_backend() == BASELINE


def test_existing_explicit_flaggems_override_is_respected(monkeypatch):
    flag_gems = pytest.importorskip("flag_gems")
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flag_gems, "vendor_name", "iluvatar", raising=False)
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "1")
    assert FlagGemsBackend().attention_backend() == (
        "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"
    )


@pytest.mark.parametrize(
    ("query_len", "kv_len", "num_q_heads", "num_kv_heads", "num_segments"),
    [
        (1, 4097, 16, 2, 0),  # 2D decode, unaligned KV tail
        (1, 16377, 16, 2, 0),  # 2D long decode, final partially-filled page
        (1, 16377, 16, 2, 4),  # 3D segmented long decode
        (1, 4097, 16, 4, 0),  # shape fallback: different KV head count
        (32, 4097, 16, 2, 0),  # prefill fallback: Q len > 1
    ],
)
def test_v5_matches_vllm_unified_attention(
    query_len, kv_len, num_q_heads, num_kv_heads, num_segments, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("Requires BI-V150 or another CUDA-compatible GPU")

    from vllm.v1.attention.ops import triton_unified_attention as baseline_module
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl import (
        triton_unified_attention_decode_v5 as optimized_module,
    )

    # Make the 3D test deterministic even if the surrounding test
    # environment enables vLLM batch-invariant execution (which ordinarily
    # suppresses the 3D kernel). Compare both implementations with the
    # same intended dispatch mode.
    monkeypatch.setattr(baseline_module, "is_batch_invariant", False)
    monkeypatch.setattr(optimized_module, "is_batch_invariant", False)
    baseline_attention = baseline_module.unified_attention

    # Verify the optimized path is actually selected in 2D/3D decode.
    seen_launches = []
    jit_kernel = optimized_module.kernel_unified_attention

    class KernelSpy:
        def __getitem__(self, grid):
            launch = jit_kernel[grid]

            def capture(**kwargs):
                seen_launches.append(
                    (kwargs["DECODE_PAGE_LOOKUP"], kwargs["IS_3D"])
                )
                return launch(**kwargs)

            return capture

    monkeypatch.setattr(
        optimized_module, "kernel_unified_attention", KernelSpy()
    )

    torch.manual_seed(2026)
    device = "cuda"
    head_size = 128
    kv_block_size = 16
    num_used_pages = (kv_len + kv_block_size - 1) // kv_block_size
    # One spare page avoids an invalid page-table load in baseline for a
    # partially filled 32-token tile. Physical page order is non-monotonic.
    num_pages = num_used_pages + 1
    physical_page_order = torch.randperm(num_pages, device=device, dtype=torch.int64)
    block_table = physical_page_order.to(torch.int32).unsqueeze(0).contiguous()
    key_cache = torch.randn(
        (num_pages, kv_block_size, num_kv_heads, head_size),
        device=device,
        dtype=torch.bfloat16,
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(
        (query_len, num_q_heads, head_size),
        device=device,
        dtype=torch.bfloat16,
    )
    cu_seqlens_q = torch.tensor([0, query_len], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([kv_len], device=device, dtype=torch.int32)

    common = dict(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=query_len,
        seqused_k=seq_lens,
        max_seqlen_k=kv_len,
        softmax_scale=head_size ** -0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )

    def call_attention(fn, output):
        args = dict(common, out=output)
        if num_segments:
            # Per-token, per-query-head partial outputs and statistics.
            args.update(
                seq_threshold_3D=64,
                num_par_softmax_segments=num_segments,
                softmax_segm_output=torch.empty(
                    (query_len, num_q_heads, num_segments, head_size),
                    device=device, dtype=torch.float32,
                ),
                softmax_segm_max=torch.empty(
                    (query_len, num_q_heads, num_segments),
                    device=device, dtype=torch.float32,
                ),
                softmax_segm_expsum=torch.empty(
                    (query_len, num_q_heads, num_segments),
                    device=device, dtype=torch.float32,
                ),
            )
        with torch.inference_mode():
            fn(**args)

    expected = torch.empty_like(query)
    actual = torch.empty_like(query)
    call_attention(baseline_attention, expected)
    call_attention(optimized_module.unified_attention, actual)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)

    expected_optimized = (
        query_len == 1 and num_q_heads == 16 and num_kv_heads == 2
    )
    assert seen_launches == [(expected_optimized, bool(num_segments))]


def test_decode_paged_lookup_is_not_enabled_for_prefill(monkeypatch):
    """No prefill tiling or non-decode page-lookup specialization."""
    if not torch.cuda.is_available():
        pytest.skip("Requires BI-V150 or another CUDA-compatible GPU")
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl import (
        triton_unified_attention_decode_v5 as optimized,
    )

    # Catch the *compile-time* gate as well as numerical equivalence.
    # The kernel does not expose a separate selectable operator.
    seen = []
    orig_kernel = optimized.kernel_unified_attention

    class KernelSpy:
        def __getitem__(self, grid):
            launcher = orig_kernel[grid]

            def capture(**kwargs):
                seen.append(kwargs["DECODE_PAGE_LOOKUP"])
                return launcher(**kwargs)

            return capture

    monkeypatch.setattr(optimized, "kernel_unified_attention", KernelSpy())
    device = "cuda"
    q = torch.randn((8, 16, 128), dtype=torch.bfloat16, device=device)
    k = torch.randn((40, 16, 2, 128), dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    args = dict(
        q=q, k=k, v=v, out=torch.empty_like(q),
        cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32, device=device),
        max_seqlen_q=8,
        seqused_k=torch.tensor([640], dtype=torch.int32, device=device),
        max_seqlen_k=640,
        softmax_scale=128**-0.5, causal=True,
        window_size=(-1, -1),
        block_table=torch.arange(40, dtype=torch.int32, device=device)[None, :],
        softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
    )
    with torch.inference_mode():
        optimized.unified_attention(**args)
    assert seen == [False]
