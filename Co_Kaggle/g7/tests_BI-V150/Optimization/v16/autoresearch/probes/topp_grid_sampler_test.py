"""Differential test of the isolated grid candidate through native sampling."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import vllm.v1.sample.ops.topk_topp_sampler as sampler

path = Path(__file__).with_name('topp_grid_candidate_scratch.py')
spec = importlib.util.spec_from_file_location('v16_topp_grid_candidate', path)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
original = sampler.apply_top_k_top_p_triton
torch.manual_seed(8012)

def generators(mode, rows):
    indices = [] if mode == 'none' else ([0, rows - 1] if mode == 'mixed' else list(range(rows)))
    return {i: torch.Generator(device='cuda').manual_seed(1000 + i) for i in indices}

def assert_same(a, b):
    if a is None or b is None:
        assert a is None and b is None
    else:
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)

cases = [
    (1, 8193, 'none', 'none', False, False, 'raw_logprobs'),
    (8, 8193, 'topp', 'none', False, False, 'raw_logits'),
    (17, 8193, 'topk', 'mixed', False, False, 'raw_logprobs'),
    (64, 8193, 'both', 'all', False, False, 'raw_logprobs'),
    (64, 130560, 'topp', 'mixed', False, False, 'raw_logprobs'),
    (17, 8193, 'topp', 'none', True, False, 'raw_logprobs'),
    (17, 8193, 'topp', 'none', False, True, 'raw_logprobs'),
    (17, 8193, 'topp', 'none', False, False, 'processed_logits'),
    (17, 8193, 'both', 'none', False, False, 'processed_logprobs'),
]
try:
    for rows, vocab, filtering, gen_mode, fp64, noncontig, mode in cases:
        source = torch.randn((rows, vocab), device='cuda') * 3
        if rows > 1:
            source[0, 0] = float('nan')
            source[1, 1] = float('inf')
        k = torch.full((rows,), min(50, vocab), device='cuda', dtype=torch.int32) if filtering in ('topk', 'both') else None
        p = torch.linspace(.5, .99, rows, device='cuda') if filtering in ('topp', 'both') else None
        def make_input():
            if not noncontig:
                return source.clone()
            out = torch.empty((rows, 2 * vocab), device='cuda')[:, ::2]
            out.copy_(source)
            assert not out.is_contiguous()
            return out
        x0, x1 = make_input(), make_input()
        g0, g1 = generators(gen_mode, rows), generators(gen_mode, rows)
        ctx = SimpleNamespace(logprobs_mode=mode, use_fp64_gumbel=fp64)
        sampler.apply_top_k_top_p_triton = original
        torch.manual_seed(831)
        out0, processed0 = sampler.TopKTopPSampler.forward_native(ctx, x0, g0, k, p)
        state0 = torch.cuda.get_rng_state()
        sampler.apply_top_k_top_p_triton = candidate.apply_top_k_top_p_triton
        torch.manual_seed(831)
        out1, processed1 = sampler.TopKTopPSampler.forward_native(ctx, x1, g1, k, p)
        state1 = torch.cuda.get_rng_state()
        torch.cuda.synchronize()
        assert torch.equal(out0, out1)
        assert torch.equal(state0, state1)
        assert all(torch.equal(g0[i].get_state(), g1[i].get_state()) for i in g0)
        assert_same(processed0, processed1)
        assert_same(x0, x1)
        print(json.dumps({'shape': [rows, vocab], 'filter': filtering, 'generators': gen_mode, 'fp64': fp64, 'noncontiguous': noncontig, 'mode': mode, 'tokens_rng_outputs_and_input_match': True}), flush=True)
finally:
    sampler.apply_top_k_top_p_triton = original

