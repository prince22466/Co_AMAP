"""BI-V150 v8 automatically selected v1+v4 grouped paged-KV lookup regression.

Compare the automatically selected combined operator against the preserved
v1 operator on identical physical KV pages; benchmark scripts
and model hyperparameters are not modified.
"""
import math
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

V1_V4 = (
    "vllm_fl.dispatch.backends.vendor.iluvatar.impl."
    "attention_biv150_grouped_v4.BIV150V1GroupedV4AttentionBackend"
)
TRITON = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
FLAGGEMS = "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"


def _backend(monkeypatch, vendor, flaggems=False):
    import vllm.platforms
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name=vendor)
    )
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "1" if flaggems else "0")
    return FlagGemsBackend().attention_backend()


def test_grouped_v4_selected_automatically_on_iluvatar(monkeypatch):
    monkeypatch.delenv("VLLM_FL_BIV150_GROUPED_V4", raising=False)
    assert _backend(monkeypatch, "iluvatar") == V1_V4


@pytest.mark.parametrize("legacy_value", ["0", "1"])
def test_legacy_grouped_flag_has_no_effect(monkeypatch, legacy_value):
    monkeypatch.setenv("VLLM_FL_BIV150_GROUPED_V4", legacy_value)
    assert _backend(monkeypatch, "iluvatar") == V1_V4


def test_grouped_v4_does_not_change_other_platforms(monkeypatch):
    assert _backend(monkeypatch, "nvidia") == TRITON


def test_explicit_flaggems_override_keeps_priority(monkeypatch):
    assert _backend(monkeypatch, "iluvatar", flaggems=True) == FLAGGEMS


@pytest.mark.parametrize(
    "query_lengths,kv_lengths",
    [
        ([256], [8193]),           # incomplete last page, grouped lookup
        ([2048], [16384]),         # long chunked prefill, grouped lookup
        ([128, 1], [8193, 8193]), # mixed prefill/decode, per-seq guard
        ([1], [16384]),            # decode-only, grouped gate disabled
        ([128], [4096]),           # short prefill, grouped gate disabled
    ],
)
def test_v1_grouped_v4_matches_v1(query_lengths, kv_lengths):
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU; run on BI-V150")

    import flag_gems
    if flag_gems.vendor_name != "iluvatar":
        pytest.skip("requires BI-V150 Iluvatar FlagGems")

    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.triton_unified_attention_biv150 import (
        unified_attention as v1_attention,
    )
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.triton_unified_attention_biv150_v4 import (
        unified_attention as v1_v4_attention,
    )

    torch.manual_seed(2026)
    page_size, query_heads, kv_heads, head_size = 16, 16, 2, 128
    device = "cuda"
    pages_per_seq = [math.ceil(n / page_size) for n in kv_lengths]
    count_pages = sum(pages_per_seq)
    q = torch.randn(
        (sum(query_lengths), query_heads, head_size),
        device=device, dtype=torch.bfloat16,
    )
    k = torch.randn(
        (count_pages, page_size, kv_heads, head_size),
        device=device, dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    # Shuffled physical allocation, padded block tables and partial tails.
    physical = torch.randperm(count_pages, device=device, dtype=torch.int64)
    table = torch.zeros(
        (len(query_lengths), max(pages_per_seq) + 2),
        device=device, dtype=torch.int32,
    )
    offset = 0
    for row, n in enumerate(pages_per_seq):
        table[row, :n] = physical[offset:offset + n].to(torch.int32)
        offset += n
    cu_q = torch.tensor(
        [0, *accumulate(query_lengths)], device=device, dtype=torch.int32
    )
    seq_lens = torch.tensor(kv_lengths, device=device, dtype=torch.int32)

    kwargs = dict(
        q=q, k=k, v=v, cu_seqlens_q=cu_q,
        max_seqlen_q=max(query_lengths), seqused_k=seq_lens,
        max_seqlen_k=max(kv_lengths), softmax_scale=head_size ** -0.5,
        causal=True, window_size=(-1, -1), block_table=table, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
    )
    baseline = torch.empty_like(q)
    grouped = torch.empty_like(q)
    with torch.inference_mode():
        v1_attention(out=baseline, **kwargs)
        v1_v4_attention(out=grouped, **kwargs)
        torch.cuda.synchronize()
    assert torch.isfinite(grouped).all().item()
    torch.testing.assert_close(grouped, baseline, atol=3e-2, rtol=3e-2)
    if max(query_lengths) == 1 or max(kv_lengths) < 8192:
        # The grouped specialization is excluded in these entire batches.
        torch.testing.assert_close(grouped, baseline, atol=0, rtol=0)
