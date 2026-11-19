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
        ([17, 33, 129], 0, "head_major", -1, True),
        ([17, 33, 129], 16, "head_major", -1, True),
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
                requested = kwargs["PIPELINE_DECODE_KV"]
                if force_original:
                    kwargs["PIPELINE_DECODE_KV"] = False
                launches.append((requested, kwargs["PIPELINE_DECODE_KV"], kwargs["IS_3D"]))
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

    assert launches == [
        (expected_pipeline, False, bool(segments)),
        (expected_pipeline, expected_pipeline, bool(segments)),
    ]
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    # Same mathematical operation order and same BF16 inputs; use a much
    # tighter control comparison than the upstream comparison.
    torch.testing.assert_close(actual, control, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(actual).all()
    # Independent dense FP32 reference for the small full-causal cases.
    # Gather only valid physical pages and expand each KV head to its
    # corresponding GQA query-head group. No Triton code is reused.
    if max(lengths) <= 513 and window == -1:
        dense = []
        for i, length in enumerate(lengths):
            pages = args["block_table"][i, : (length + 15) // 16].long()
            k = args["k"][pages].reshape(-1, 2, 128)[:length].float()
            v = args["v"][pages].reshape(-1, 2, 128)[:length].float()
            k = k.repeat_interleave(8, dim=1)
            v = v.repeat_interleave(8, dim=1)
            scores = torch.einsum("hd,thd->ht", q[i].float(), k) * args["softmax_scale"]
            dense.append(torch.einsum("ht,thd->hd", scores.softmax(-1), v))
        torch.testing.assert_close(actual.float(), torch.stack(dense), atol=3e-2, rtol=3e-2)


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
    elif layout == "head_major":
        # vLLM HND physical order, with the same logical NHD-shaped view.
        kv = torch.randn((pages, 2, kv_heads, page_size, dim),
                         device="cuda", dtype=torch.bfloat16).permute(0, 1, 3, 2, 4)
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
        # vLLM build_for_cudagraph_capture fills seq_lens with one,
        # not the eventual serving context length. Warm and capture that way.
        args["seqused_k"].fill_(1)
        candidate.unified_attention(**args, out=actual)
        force_original = True
        candidate.unified_attention(**args, out=expected)
        torch.cuda.synchronize()
        force_original = False
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate.unified_attention(**args, out=actual)
        original_table = args["block_table"].clone()
        for i, lengths in enumerate(([1, 1], [4097, 4097], [17, 33], [32, 4096], [4097, 4097])):
            # Reuse captured addresses with changing sequence lengths,
            # physical pages and queries. Preserve valid unique page IDs.
            args["seqused_k"].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            args["block_table"].copy_(original_table.roll(i, dims=1))
            q.neg_()
            actual.fill_(float("nan"))
            graph.replay()
            force_original = True
            expected.fill_(float("nan"))
            candidate.unified_attention(**args, out=expected)
            force_original = False
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
            assert torch.isfinite(actual).all()


@pytest.mark.parametrize("batch,length,segments", [
    (64, 4608, 0), (64, 4608, 16), (30, 16896, 0), (30, 16896, 16),
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
                assert kwargs["IS_3D"] == bool(segments)
                if force_original:
                    kwargs["PIPELINE_DECODE_KV"] = False
                return launch(**kwargs)

            return call

    monkeypatch.setattr(candidate, "kernel_unified_attention", Spy())

    def invoke(control_path):
        nonlocal force_original
        force_original = control_path
        candidate.unified_attention(**args, out=control if control_path else actual)

    def timed(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    def measure(call, mode):
        samples = {True: [], False: []}
        for i in range(12):
            order = (True, False) if i % 2 == 0 else (False, True)
            for control_path in order:
                samples[control_path].append(timed(lambda: call(control_path)))
        c, v = (statistics.median(samples[k]) for k in (True, False))
        print(f"V17_KV_PIPELINE mode={mode} batch={batch} length={length} segments={segments} "
              f"control_ms={c:.6f} candidate_ms={v:.6f} speedup={c/v:.4f} "
              f"control_samples={samples[True]} candidate_samples={samples[False]}")

    with torch.inference_mode():
        for _ in range(3):
            invoke(True)
            invoke(False)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, control, atol=2e-3, rtol=2e-3)
        measure(invoke, "eager")
        # Measure full attention calls (including reduce_segments in 3D)
        # under graph replay as well. Capture both with the startup length.
        args["seqused_k"].fill_(1)
        torch.cuda.synchronize()
        graphs = {}
        for control_path in (True, False):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                invoke(control_path)
            graphs[control_path] = graph
        args["seqused_k"].fill_(length)
        for _ in range(3):
            graphs[True].replay()
            graphs[False].replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, control, atol=2e-3, rtol=2e-3)
        measure(lambda control_path: graphs[control_path].replay(), "graph")


@pytest.mark.parametrize("mode", ["prefill", "mixed", "batch_invariant", "different_heads"])
def test_pipeline_fallback(mode, monkeypatch):
    """Verify that unsupported paths are never opted into the lookahead."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible GPU required")
    from vllm_fl.platform import PlatformFL

    PlatformFL.pre_register_and_update()
    from vllm_fl.dispatch.backends import triton_unified_attention as candidate
    monkeypatch.setattr(candidate, "is_batch_invariant", mode == "batch_invariant")
    args, q = make_inputs([33, 65], 0, "interleaved")
    if mode in ("prefill", "mixed"):
        qlens = [2, 2] if mode == "prefill" else [1, 2]
        q = torch.randn((sum(qlens), 16, 128), device="cuda", dtype=torch.bfloat16)
        args.update(q=q, max_seqlen_q=2,
                    cu_seqlens_q=torch.tensor([0, qlens[0], sum(qlens)],
                                               device="cuda", dtype=torch.int32))
    elif mode == "different_heads":
        # 8 query heads, 2 KV heads: still valid GQA but not target geometry.
        q = q[:, :8, :].contiguous()
        args.update(q=q)
    original = candidate.kernel_unified_attention
    seen = []

    class Spy:
        def __getitem__(self, grid):
            launch = original[grid]

            def call(**kwargs):
                seen.append(kwargs["PIPELINE_DECODE_KV"])
                return launch(**kwargs)

            return call

    monkeypatch.setattr(candidate, "kernel_unified_attention", Spy())
    with torch.inference_mode():
        out = torch.empty_like(q)
        candidate.unified_attention(**args, out=out)
        torch.cuda.synchronize()
    assert seen == [False]
    assert torch.isfinite(out).all()
