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
    monkeypatch.delenv("VLLM_FL_BIV150_ATTN_V1", raising=False)
    assert FlagGemsBackend().attention_backend() == BASELINE


def test_opt_in_selects_v2_on_biv150(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    monkeypatch.setenv("VLLM_FL_BIV150_ATTN_V2", "1")
    monkeypatch.delenv("VLLM_FL_BIV150_ATTN_V1", raising=False)
    assert FlagGemsBackend().attention_backend() == V2


def test_opt_in_rejects_conflicting_attention_flags(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "1")
    monkeypatch.setenv("VLLM_FL_BIV150_ATTN_V2", "1")
    monkeypatch.delenv("VLLM_FL_BIV150_ATTN_V1", raising=False)
    with pytest.raises(RuntimeError, match="cannot both be enabled"):
        FlagGemsBackend().attention_backend()


def test_opt_in_rejects_v1_v2_conflict(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
    monkeypatch.setenv("VLLM_FL_BIV150_ATTN_V1", "1")
    monkeypatch.setenv("VLLM_FL_BIV150_ATTN_V2", "1")
    with pytest.raises(RuntimeError, match="V1 and VLLM_FL_BIV150_ATTN_V2"):
        FlagGemsBackend().attention_backend()


@pytest.mark.parametrize(
    ("query_len", "context_len", "permute_block_table"),
    [
        (1, 512, False),        # Single-token decode: unchanged geometry.
        (64, 0, False),         # Short uncached prefill.
        (127, 2048, False),     # Short query, long cached prefix: v1 tile.
        (128, 2048, False),     # Boundary of the v2 64-token KV tile.
        (128, 2049, True),      # 2177 KV tokens: partial final 64-token tile.
        (256, 0, False),        # Long query but KV < 2048: v1 tile.
        (2048, 0, False),       # First chunk.
        (2048, 4096, False),    # Chunked prefill after an existing prefix.
        (2048, 4097, True),     # Nonaligned chunked prefill; paged KV.
        (128, 16257, True),     # Partial tile immediately beyond 16K.
        (2048, 14336, False),   # 16K endpoint (expensive).
    ],
)
def test_bf16_gqa_prefill_matches_baseline(
    query_len, context_len, permute_block_table
):
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
    if permute_block_table:
        # Nonsequential physical block mapping ensures the wide tile uses the
        # correct paged-KV addresses, including the masked final tile.
        block_ids = torch.randperm(num_blocks, device=device).to(torch.int32)
    else:
        block_ids = torch.arange(num_blocks, device=device, dtype=torch.int32)
    # The optimized kernel is deliberately given a *tightly sized* table.
    # Pad the baseline-only table to a 64-token boundary: pinned upstream
    # vLLM also performs an unmasked table read for its final (32-token)
    # tile. Supplying the same logical pages plus harmless padding makes
    # the reference safe without hiding the v2 tail-mask regression case.
    block_table = block_ids.reshape(1, -1)
    padded_blocks = ((total_len + 63) // 64) * (64 // kv_block_size)
    padded_ids = torch.cat(
        [
            block_ids,
            torch.zeros(
                padded_blocks - num_blocks,
                device=device,
                dtype=torch.int32,
            ),
        ]
    )
    baseline_block_table = padded_ids.reshape(1, -1)
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
        baseline_unified_attention(
            **{**arguments, "block_table": baseline_block_table}, out=expected
        )
        optimized_unified_attention(**arguments, out=actual)
        torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    if query_len == 1:
        # No changes to single-token decode selection or computation.
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)



def test_v2_backend_path_imports():
    import importlib

    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    module_path, _, class_name = V2.rpartition(".")
    backend = getattr(importlib.import_module(module_path), class_name)
    assert issubclass(backend, TritonAttentionBackend)


def test_mixed_batch_non_aligned_paged_kv_matches_baseline():
    """Different Q/K lengths, shuffled KV pages, and a short final KV block."""
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA-compatible GPU (run on BI-V150)")

    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention as baseline_unified_attention,
    )
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.triton_unified_attention_biv150_v2 import (
        unified_attention as optimized_unified_attention,
    )

    device = "cuda"
    torch.manual_seed(2027)
    head_size = 128
    kv_block_size = 16
    q_lens = (128, 256)
    kv_lens = (2177, 4353)
    num_blocks_per_seq = [
        (kv_len + kv_block_size - 1) // kv_block_size
        for kv_len in kv_lens
    ]
    total_blocks = sum(num_blocks_per_seq)
    physical_ids = torch.randperm(total_blocks, device=device).to(torch.int32)

    # Optimized kernel gets no extra columns beyond the largest sequence.
    block_table = torch.zeros(
        (2, max(num_blocks_per_seq)), dtype=torch.int32, device=device
    )
    offset = 0
    for row, count in enumerate(num_blocks_per_seq):
        block_table[row, :count] = physical_ids[offset : offset + count]
        offset += count

    # The pinned upstream reference has an unmasked table load. Provide
    # extra tail columns only for that reference to avoid its OOB access.
    padded_width = ((max(kv_lens) + 63) // 64) * (64 // kv_block_size)
    baseline_block_table = torch.zeros(
        (2, padded_width), dtype=torch.int32, device=device
    )
    baseline_block_table[:, : block_table.shape[1]] = block_table

    query = torch.randn(
        (sum(q_lens), 16, head_size), device=device, dtype=torch.bfloat16
    )
    key_cache = torch.randn(
        (total_blocks, kv_block_size, 2, head_size),
        device=device,
        dtype=torch.bfloat16,
    )
    value_cache = torch.randn_like(key_cache)
    arguments = dict(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=torch.tensor(
            [0, q_lens[0], sum(q_lens)], device=device, dtype=torch.int32
        ),
        max_seqlen_q=max(q_lens),
        seqused_k=torch.tensor(kv_lens, device=device, dtype=torch.int32),
        max_seqlen_k=max(kv_lens),
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
        baseline_unified_attention(
            **{**arguments, "block_table": baseline_block_table}, out=expected
        )
        optimized_unified_attention(**arguments, out=actual)
        torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


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
