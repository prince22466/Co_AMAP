# SPDX-License-Identifier: Apache-2.0
"""v17 KV lookahead: same-input original/candidate and upstream checks."""

import statistics

import pytest
import torch


@pytest.mark.parametrize(
    "lengths,segments,layout,window,expected_pipeline",
    [
        ([1, 15, 16, 17, 31, 32, 33], 0, "interleaved", -1, True),
        ([33, 129, 513, 4097], 4, "interleaved", -1, True),
        ([1, 17, 33, 65], 16, "interleaved", -1, True),
        ([4097, 4129], 0, "separate", -1, True),
        ([16377, 16385], 16, "interleaved", -1, True),
        ([4097, 4129], 0, "interleaved", 127, False),
    ],
)
def test_decode_lookahead(lengths, segments, layout, window, expected_pipeline, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible GPU required")
    from vllm_fl.platform import PlatformFL

    PlatformFL.pre_register_and_update()
    from vllm.v1.attention.ops import triton_unified_attention as upstream
    from vllm_fl.dispatch.backends import triton_unified_attention as candidate

    monkeypatch.setattr(candidate, "is_batch_invariant", False)
    monkeypatch.setattr(upstream, "is_batch_invariant", False)

    args, q = make_inputs(lengths, segments, layout, window)
    # The unmodified v17 path is the most direct correctness control.
    original = candidate.kernel_unified_attention
    launches = []
    force_original = False

    class Spy:
        def __getitem__(self, grid):
            launch = original[grid]

            def call(**kwargs):
                launches.append((kwargs["PIPELINE_DECODE_KV"], kwargs["IS_3D"]))
                if force_original:
                    kwargs["PIPELINE_DECODE_KV"] = False
                return launch(**kwargs)

            return call

    monkeypatch.setattr(candidate, "kernel_unified_attention", Spy())

    with torch.inference_mode():
        expected = torch.empty_like(q)
        control = torch.empty_like(q)
        actual = torch.empty_like(q)
        upstream.unified_attention(**args, out=expected)
        force_original = True
        candidate.unified_attention(**args, out=control)
        force_original = False
        candidate.unified_attention(**args, out=actual)
        torch.cuda.synchronize()

    assert launches == [(expected_pipeline, bool(segments))] * 2
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    # Same mathematical operation order and same BF16 inputs; use a much
    # tighter control comparison than the upstream comparison.
    torch.testing.assert_close(actual, control, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(actual).all()


def make_inputs(lengths, segments, layout, window=-1):
    torch.manual_seed(20261119)
    batch = len(lengths)
    page_size, heads, kv_heads, dim = 16, 16, 2, 128
    counts = [(length + page_size - 1) // page_size for length in lengths]
    pages = sum(counts) + 1
    order = torch.randperm(pages, device="cuda", dtype=torch.int32)
    # Initially allocate the exact number of pages for the longest sequence.
    table = torch.zeros((batch, max(counts)), device="cuda", dtype=torch.int32)
    offset = 0
    for i, count in enumerate(counts):
        table[i, :count] = order[offset:offset+count]
        offset += count
    # Upstream's stock 2D path may read an extra table entry for an odd
    # number of 16-token pages. Add one column for reference safety;
    # the exact-width graph test below removes it.
    table = torch.cat((table, torch.zeros_like(table[:, :1])), dim=1)
    if layout == "interleaved":
        kv = torch.randn((pages, 2, page_size, kv_heads, dim),
                         device="cuda", dtype=torch.bfloat16)
        k, v = kv.unbind(1)
    else:
        k = torch.randn((pages, page_size, kv_heads, dim),
                        device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
    q = torch.randn((batch, heads, dim), device="cuda", dtype=torch.bfloat16)
    args = dict(
        q=q, k=k, v=v, cu_seqlens_q=torch.arange(batch+1, device="cuda", dtype=torch.int32),
        max_seqlen_q=1, seqused_k=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        max_seqlen_k=max(lengths), softmax_scale=dim**-0.5, causal=True,
        window_size=(window, window), block_table=table, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
    )
    if segments:
        shape = (batch, heads, segments)
        args.update(
            seq_threshold_3D=batch,
            num_par_softmax_segments=segments,
            softmax_segm_output=torch.empty((*shape, dim), device="cuda", dtype=torch.float32),
            softmax_segm_max=torch.empty(shape, device="cuda", dtype=torch.float32),
            softmax_segm_expsum=torch.empty(shape, device="cuda", dtype=torch.float32),
        )
    return args, q


@pytest.mark.parametrize("segments", [0, 16])
def test_exact_page_table_and_graph_replay(segments, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible GPU required")
    from vllm_fl.platform import PlatformFL

    PlatformFL.pre_register_and_update()
    from vllm_fl.dispatch.backends import triton_unified_attention as candidate
    monkeypatch.setattr(candidate, "is_batch_invariant", False)
    # Odd page count, partial last page, and exact table allocation.
    args, q = make_inputs([4097, 4097], segments, "interleaved")
    args["block_table"] = args["block_table"][:, :-1].contiguous()
    expected, actual = torch.empty_like(q), torch.empty_like(q)
    original = candidate.kernel_unified_attention
    force_original = False

    class Spy:
        def __getitem__(self, grid):
            launch = original[grid]

            def call(**kwargs):
                assert kwargs["PIPELINE_DECODE_KV"]
                if force_original:
                    kwargs["PIPELINE_DECODE_KV"] = False
                return launch(**kwargs)

            return call

    monkeypatch.setattr(candidate, "kernel_unified_attention", Spy())
    with torch.inference_mode():
        candidate.unified_attention(**args, out=actual)
        force_original = True
        candidate.unified_attention(**args, out=expected)
        torch.cuda.synchronize()
        force_original = False
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate.unified_attention(**args, out=actual)
        for lengths in ([4097, 4097], [17, 33], [32, 4096]):
            # Reuse the same graph addresses with different live metadata.
            args["seqused_k"].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            graph.replay()
            force_original = True
            candidate.unified_attention(**args, out=expected)
            force_original = False
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
            assert torch.isfinite(actual).all()


@pytest.mark.parametrize("batch,length,segments", [
    (64, 4608, 0), (30, 16896, 0), (30, 16896, 16),
])
def test_matched_decode_timing(batch, length, segments, monkeypatch):
    """Diagnostic A/B; no assertion of speedup or official score."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible GPU required")
    from vllm_fl.platform import PlatformFL

    PlatformFL.pre_register_and_update()
    from vllm_fl.dispatch.backends import triton_unified_attention as candidate
    monkeypatch.setattr(candidate, "is_batch_invariant", False)
    args, q = make_inputs([length] * batch, segments, "interleaved")
    original = candidate.kernel_unified_attention
    force_original = False
    control, actual = torch.empty_like(q), torch.empty_like(q)

    class Spy:
        def __getitem__(self, grid):
            launch = original[grid]

            def call(**kwargs):
                assert kwargs["PIPELINE_DECODE_KV"]
                if force_original:
                    kwargs["PIPELINE_DECODE_KV"] = False
                return launch(**kwargs)

            return call

    monkeypatch.setattr(candidate, "kernel_unified_attention", Spy())

    def invoke(control_path):
        nonlocal force_original
        force_original = control_path
        candidate.unified_attention(**args, out=control if control_path else actual)

    def timed(control_path):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invoke(control_path)
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    with torch.inference_mode():
        for _ in range(3):
            invoke(True)
            invoke(False)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, control, atol=2e-3, rtol=2e-3)
        samples = {True: [], False: []}
        for i in range(12):
            order = (True, False) if i % 2 == 0 else (False, True)
            for control_path in order:
                samples[control_path].append(timed(control_path))
    c, v = (statistics.median(samples[k]) for k in (True, False))
    print(f"V17_KV_PIPELINE batch={batch} length={length} segments={segments} "
          f"control_ms={c:.6f} candidate_ms={v:.6f} speedup={c/v:.4f}")
