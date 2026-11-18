# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the v16 bounded native Triton top-p launch."""

import pytest
import torch


@pytest.fixture(scope="module")
def modules():
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA-compatible GPU")

    from vllm_fl.platform import PlatformFL

    PlatformFL.pre_register_and_update()
    from vllm.v1.sample.ops import topk_topp_sampler as sampler
    from vllm.v1.sample.ops import topk_topp_triton as upstream
    from vllm_fl.ops.topk_topp import install_top_p_row_grid

    original = upstream.apply_top_k_top_p_triton
    # Exercise the actual Iluvatar hook, not a test-only sampler replacement.
    from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
        patch_top_p_row_grid_for_iluvatar,
    )

    patch_top_p_row_grid_for_iluvatar()
    selected = sampler.apply_top_k_top_p_triton
    assert getattr(selected, "_fl_row_grid_original", None) is original
    assert install_top_p_row_grid()
    assert sampler.apply_top_k_top_p_triton is selected
    assert upstream.apply_top_k_top_p_triton is original
    return upstream, sampler, selected


def _same(a, b):
    assert torch.equal(a.contiguous().view(torch.int32),
                       b.contiguous().view(torch.int32))


@pytest.mark.parametrize("rows", [8, 16, 17, 32, 48, 64, 65])
@pytest.mark.parametrize("mode", ["top-p", "both", "top-k"])
def test_grid_and_mask(modules, monkeypatch, rows, mode):
    upstream, sampler, selected = modules
    vocab = 130560 if rows == 64 else 8193
    logits = torch.randn((rows, vocab), device="cuda") * 3
    logits[0, :20] = 1
    logits[1, :20] = -float("inf")
    logits[2, 0] = float("nan")
    logits[3, 0] = float("inf")
    k = torch.full((rows,), 50, dtype=torch.int32, device="cuda") if mode != "top-p" else None
    p = torch.linspace(.5, .99, rows, device="cuda") if mode != "top-k" else None
    expected = upstream.apply_top_k_top_p_triton(logits.clone(), k, p)

    seen = []
    kernel = upstream._topk_topp_kernel

    class KernelSpy:
        def __getitem__(self, grid):
            launch = kernel[grid]

            def run(*args, **kwargs):
                seen.append((grid, args[2].shape[0]))
                return launch(*args, **kwargs)

            return run

    monkeypatch.setattr(upstream, "_topk_topp_kernel", KernelSpy())
    # Through the actual sampler dispatch (these sizes use Triton in v0.24).
    actual = sampler.apply_top_k_top_p(logits.clone(), k, p)
    torch.cuda.synchronize()
    _same(expected, actual)
    units = upstream.num_compute_units(logits.device.index)
    programs = min(rows, units * (4 if p is not None else 1))
    assert seen == [((programs,), programs)]


def test_noncontiguous_and_no_filter(modules):
    upstream, _, selected = modules
    logits = torch.randn((64, 16386), device="cuda")[:, ::2]
    p = torch.full((64,), .95, device="cuda")
    assert selected(logits, None, None) is logits
    expected = upstream.apply_top_k_top_p_triton(logits.clone(), None, p)
    actual = selected(logits, None, p)
    torch.cuda.synchronize()
    _same(expected, actual)


def test_native_rng_and_processed_outputs(modules, monkeypatch):
    from types import SimpleNamespace

    upstream, sampler, selected = modules
    logits = torch.randn((64, 8193), device="cuda") * 3
    p = torch.full((64,), .95, device="cuda")
    for mode in ("raw_logprobs", "processed_logits", "processed_logprobs"):
        for fp64 in (False, True):
            outputs = []
            for fn in (upstream.apply_top_k_top_p_triton, selected):
                monkeypatch.setattr(sampler, "apply_top_k_top_p_triton", fn)
                torch.manual_seed(271)
                generator = torch.Generator(device="cuda").manual_seed(123)
                x = logits.clone()
                ctx = SimpleNamespace(logprobs_mode=mode, use_fp64_gumbel=fp64)
                tokens, processed = sampler.TopKTopPSampler.forward_native(
                    ctx, x, {0: generator}, None, p
                )
                torch.cuda.synchronize()
                outputs.append((tokens, processed, x, torch.cuda.get_rng_state(),
                                generator.get_state()))
            for left, right in zip(*outputs):
                if left is None:
                    assert right is None
                elif left.dtype == torch.float32:
                    _same(left, right)
                else:
                    assert torch.equal(left, right)


def test_graph_replay_across_batch_sizes(modules):
    upstream, sampler, _ = modules
    # A distinct vocab avoids reusing a large cache from earlier tests.
    vocab = 8201
    units = upstream.num_compute_units(torch.cuda.current_device())
    small, large = units + 1, 4 * units
    if large > 512:
        pytest.skip("Limit graph test workspace on large GPUs")
    graphs = []
    workspace = None
    for rows in (small, large):
        x = torch.randn((rows, vocab), device="cuda")
        p = torch.full((rows,), .95, device="cuda")
        sampler.apply_top_k_top_p(x, None, p)
        torch.cuda.synchronize()
        key = (x.device, x.dtype, vocab, "fl_row_grid")
        current = upstream._TRITON_BUFFER_CACHE[key]
        assert current.shape == (large, vocab)
        if workspace is None:
            workspace = current
        else:
            assert current is workspace
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sampler.apply_top_k_top_p(x, None, p)
        graphs.append((graph, x, p))
    # Replay the smaller graph after capture of a larger batch.
    for graph, x, p in graphs:
        for _ in range(3):
            new = torch.randn_like(x)
            expected = upstream.apply_top_k_top_p_triton(new.clone(), None, p)
            x.copy_(new)
            graph.replay()
            torch.cuda.synchronize()
            _same(expected, x)
