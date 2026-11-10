# SPDX-License-Identifier: Apache-2.0
"""v12 experimental block-sparse causal attention: BI-V150 GPU checks.

This is an approximate attention experiment. It does NOT prove full-model
numerical equivalence, unchanged MATH-500 accuracy, or competition eligibility.
"""
import pytest
import torch


def _case(query_len=65, context_len=95):
    device = "cuda"
    torch.manual_seed(12012026)
    total_len = query_len + context_len
    page = 16
    used_pages = (total_len + page - 1) // page
    num_pages = used_pages + 2  # spare padded page-table entries
    q = torch.randn((query_len, 16, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn((num_pages, page, 2, 128), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    # Nontrivial logical->physical mapping verifies the paged-KV address path.
    logical_to_physical = torch.randperm(
        num_pages, device=device, dtype=torch.int64
    )
    table = logical_to_physical.to(torch.int32).unsqueeze(0)
    kwargs = dict(
        q=q, k=k, v=v,
        cu_seqlens_q=torch.tensor([0, query_len], device=device, dtype=torch.int32),
        max_seqlen_q=query_len,
        seqused_k=torch.tensor([total_len], device=device, dtype=torch.int32),
        max_seqlen_k=total_len,
        softmax_scale=128**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    ordered_k = k[logical_to_physical[:used_pages]].reshape(-1, 2, 128)[:total_len]
    ordered_v = v[logical_to_physical[:used_pages]].reshape(-1, 2, 128)[:total_len]
    return kwargs, ordered_k, ordered_v


def _sparse_reference(kwargs, ordered_k, ordered_v, window, sinks):
    """Exact causal local+prefix mask, FP32 PyTorch dot/softmax reference."""
    q = kwargs["q"]
    ctx = ordered_k.shape[0] - q.shape[0]
    all_positions = torch.arange(ordered_k.shape[0], device=q.device)
    kv_idx = torch.arange(16, device=q.device) // 8
    output = torch.empty_like(q)
    for i in range(q.shape[0]):
        absolute_q = ctx + i
        allowed = (all_positions <= absolute_q) & (
            (all_positions < sinks) | (all_positions + window > absolute_q)
        )
        k_rows = ordered_k[allowed][:, kv_idx].float()
        v_rows = ordered_v[allowed][:, kv_idx].float()
        # (H,D) x (T,H,D) -> (H,T)
        scores = torch.einsum("hd,thd->ht", q[i].float(), k_rows)
        scores = scores * kwargs["softmax_scale"]
        p = torch.softmax(scores, dim=-1)
        answer = torch.einsum("ht,thd->hd", p, v_rows)
        output[i] = answer.to(q.dtype)
    return output


@pytest.mark.parametrize(
    ("qlen", "ctx", "window", "sinks"),
    [
        (65, 95, 32, 16),     # query tail + shuffled KV pages + sparse gap
        (9, 145, 48, 0),     # no prefix sink + nonzero context
        (7, 97, 17, 33),     # misaligned local and prefix boundaries
        (65, 33, 256, 8),    # window contains all causal context: dense equivalence
    ],
)
def test_v12_sparse_matches_explicit_reference(qlen, ctx, window, sinks):
    if not torch.cuda.is_available():
        pytest.skip("Requires BI-V150 GPU")
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl import (
        triton_unified_attention_biv150 as kernel,
    )
    kwargs, ordered_k, ordered_v = _case(qlen, ctx)
    actual = torch.empty_like(kwargs["q"])
    with torch.inference_mode():
        kernel.unified_attention(
            **kwargs, out=actual,
            experimental_sparse_window_tokens=window,
            experimental_sparse_sink_tokens=sinks,
        )
        torch.cuda.synchronize()
    expected = _sparse_reference(kwargs, ordered_k, ordered_v, window, sinks)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)


def test_v12_default_remains_dense_and_long_window_agrees():
    if not torch.cuda.is_available():
        pytest.skip("Requires BI-V150 GPU")
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl import (
        triton_unified_attention_biv150 as kernel,
    )
    kwargs, _, _ = _case(65, 33)
    a = torch.empty_like(kwargs["q"])
    b = torch.empty_like(kwargs["q"])
    with torch.inference_mode():
        kernel.unified_attention(**kwargs, out=a)
        kernel.unified_attention(
            **kwargs, out=b,
            experimental_sparse_window_tokens=256,
            experimental_sparse_sink_tokens=16,
        )
        torch.cuda.synchronize()
    torch.testing.assert_close(a, b, atol=3e-2, rtol=3e-2)


def test_v12_unsupported_prefill_fails_closed():
    if not torch.cuda.is_available():
        pytest.skip("Requires BI-V150 GPU")
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl import (
        triton_unified_attention_biv150 as kernel,
    )
    kwargs, _, _ = _case(65, 33)
    out = torch.empty_like(kwargs["q"])
    with pytest.raises(ValueError, match="standard BF16 GQA"):
        kernel.unified_attention(
            **{**kwargs, "window_size": (31, -1)},
            out=out,
            experimental_sparse_window_tokens=32,
            experimental_sparse_sink_tokens=8,
        )
    with pytest.raises(ValueError, match="sink requires"):
        kernel.unified_attention(
            **kwargs, out=out,
            experimental_sparse_window_tokens=0,
            experimental_sparse_sink_tokens=8,
        )


def test_v12_decode_stays_dense_when_sparse_requested():
    if not torch.cuda.is_available():
        pytest.skip("Requires BI-V150 GPU")
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl import (
        triton_unified_attention_biv150 as kernel,
    )
    kwargs, _, _ = _case(1, 95)
    a = torch.empty_like(kwargs["q"])
    b = torch.empty_like(kwargs["q"])
    with torch.inference_mode():
        kernel.unified_attention(**kwargs, out=a)
        kernel.unified_attention(
            **kwargs, out=b,
            experimental_sparse_window_tokens=32,
            experimental_sparse_sink_tokens=8,
        )
        torch.cuda.synchronize()
    torch.testing.assert_close(a, b, atol=3e-2, rtol=3e-2)
