"""BI-V150 correctness regression for v4 grouped page lookups in vLLM Triton.

Compares exactly the same paged-KV inputs against the upstream vLLM 0.24.0
unified attention operator. Includes partial last KV pages and mixed
prefill/decode batches. This is NOT a performance/competition benchmark.
"""

import math

import pytest
import torch


@pytest.mark.parametrize(
    "query_lengths,kv_lengths",
    [
        ([256], [8193]),        # incomplete final page
        ([2048], [16384]),     # long chunked prefill
        ([128, 1], [8193, 8193]),  # prefill + decode in same launch
    ],
)
def test_grouped_prefill_matches_original_triton(query_lengths, kv_lengths):
    import flag_gems

    if flag_gems.vendor_name != "iluvatar" or not torch.cuda.is_available():
        pytest.skip("requires BI-V150 GPU and Iluvatar FlagGems platform")

    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention as upstream_attention,
    )
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.triton_unified_attention_grouped_v4 import (
        unified_attention as grouped_attention,
    )

    torch.manual_seed(2026)
    device = "cuda"
    page_size, heads, kv_heads, d = 16, 16, 2, 128
    pages_per_sequence = [math.ceil(n / page_size) for n in kv_lengths]
    total_pages = sum(pages_per_sequence)
    q_count = sum(query_lengths)

    q = torch.randn((q_count, heads, d), dtype=torch.bfloat16, device=device)
    k = torch.randn(
        (total_pages, page_size, kv_heads, d), dtype=torch.bfloat16, device=device
    )
    v = torch.randn_like(k)

    # Random physical page order, plus a padded table column so that the
    # upstream unmasked page lookup stays within the allocated table.
    shuffled = torch.randperm(total_pages, device=device).to(torch.int32)
    table = torch.zeros(
        (len(query_lengths), max(pages_per_sequence) + 2),
        device=device, dtype=torch.int32,
    )
    start = 0
    for row, page_count in enumerate(pages_per_sequence):
        table[row, :page_count] = shuffled[start : start + page_count]
        start += page_count

    cu_q = torch.tensor(
        [0] + list(__import__("itertools").accumulate(query_lengths)),
        device=device, dtype=torch.int32,
    )
    seq_lens = torch.tensor(kv_lengths, device=device, dtype=torch.int32)

    def execute(operator):
        out = torch.empty_like(q)
        operator(
            q=q, k=k, v=v, out=out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=max(query_lengths),
            seqused_k=seq_lens,
            max_seqlen_k=max(kv_lengths),
            softmax_scale=1.0 / math.sqrt(d),
            causal=True,
            window_size=(-1, -1),
            block_table=table,
            softcap=0.0,
            q_descale=None, k_descale=None, v_descale=None,
        )
        return out

    expected = execute(upstream_attention)
    actual = execute(grouped_attention)
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all().item()
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
