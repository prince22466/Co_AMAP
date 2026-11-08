"""Mixed paged-KV attention correctness for BI-V150 prefill and fallback."""
import pytest
import torch

@pytest.mark.parametrize(
    ("window", "expected_stages"),
    [((-1, -1), 2), ((127, 0), None)],
)
def test_mixed_prefill_decode_paging_matches_reference(window, expected_stages, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Requires BI-V150 or another CUDA-compatible GPU")

    from vllm.v1.attention.ops.triton_unified_attention import unified_attention as reference
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl import (
        triton_unified_attention_biv150 as candidate,
    )

    torch.manual_seed(261107)
    device = "cuda"
    page, head, qheads, kvheads = 16, 128, 16, 2
    qlens = [1, 63, 2048]
    contexts = [16384, 4097, 14336]
    lengths = [q + c for q, c in zip(qlens, contexts)]
    counts = [(length + page - 1) // page for length in lengths]
    pages = sum(count + 1 for count in counts)

    query = torch.randn((sum(qlens), qheads, head), device=device, dtype=torch.bfloat16)
    key = torch.randn((pages, page, kvheads, head), device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    order = torch.randperm(pages, device=device, dtype=torch.int64).to(torch.int32)
    table = torch.zeros((len(qlens), max(counts) + 1), device=device, dtype=torch.int32)
    offset = 0
    for index, count in enumerate(counts):
        table[index, :count + 1] = order[offset:offset + count + 1]
        offset += count + 1

    qstarts = torch.tensor([0, 1, 64, 2112], device=device, dtype=torch.int32)
    seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    common = dict(
        q=query, k=key, v=value, cu_seqlens_q=qstarts, max_seqlen_q=max(qlens),
        seqused_k=seq_lens, max_seqlen_k=max(lengths), softmax_scale=head ** -0.5,
        causal=True, window_size=window, block_table=table, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
    )

    launches = []
    original = candidate.kernel_unified_attention

    class KernelSpy:
        def __getitem__(self, grid):
            launch = original[grid]
            def call(**kwargs):
                launches.append((kwargs["BLOCK_M"], kwargs.get("num_stages")))
                return launch(**kwargs)
            return call

    monkeypatch.setattr(candidate, "kernel_unified_attention", KernelSpy())
    expected = torch.empty_like(query)
    actual = torch.empty_like(query)
    with torch.inference_mode():
        reference(**common, out=expected)
        candidate.unified_attention(**common, out=actual)
        torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    assert len(launches) == 1
    assert launches[0][0] == (64 if expected_stages == 2 else 16)
    assert launches[0][1] == expected_stages
