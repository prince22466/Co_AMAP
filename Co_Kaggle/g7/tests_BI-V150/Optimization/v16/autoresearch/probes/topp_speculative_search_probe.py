"""Scratch-only speculative top-p search: compare against the installed kernel."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import torch
from vllm.v1.sample.ops import topk_topp_triton as baseline

source = Path(baseline.__file__)
code = source.read_text()
start = code.index('            #### STANDALONE TOP-P SAMPLING ####')
end = code.index('        # Sixth pass: Apply mask', start)
segment = code[start:end]
start_search = segment.index('                    found_pivot = 0\n                    while found_pivot == 0:')
end_search = segment.index('                else:\n                    # Re-populate', start_search)
replacement = '''                    found_pivot = 0
                    while found_pivot == 0:
                        mid = (max_range - min_range) * 0.5 + min_range
                        low = (mid - min_range) * 0.5 + min_range
                        high = (max_range - mid) * 0.5 + mid
                        s_mid = 0.0
                        s_low = 0.0
                        s_high = 0.0
                        m_mid = 1.0
                        m_low = 1.0
                        m_high = 1.0
                        n_mid = tl.zeros((), dtype=tl.uint32)
                        n_low = tl.zeros((), dtype=tl.uint32)
                        n_high = tl.zeros((), dtype=tl.uint32)
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                            mask_n_2 = offs_n < search_range
                            probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0)
                            above_mid = probs_blk > mid
                            above_low = probs_blk > low
                            above_high = probs_blk > high
                            s_mid += tl.sum(probs_blk * above_mid)
                            s_low += tl.sum(probs_blk * above_low)
                            s_high += tl.sum(probs_blk * above_high)
                            m_mid, n_mid = _update_min_larger_stats(probs_blk, above_mid, m_mid, n_mid, 1.0)
                            m_low, n_low = _update_min_larger_stats(probs_blk, above_low, m_low, n_low, 1.0)
                            m_high, n_high = _update_min_larger_stats(probs_blk, above_high, m_high, n_high, 1.0)
                        for step in range(0, 2):
                            if step == 0:
                                pivot = mid
                                pivot_sum = s_mid
                                minimum = m_mid
                                num_min = n_mid
                            else:
                                pivot = tl.where(s_mid > p, high, tl.where(s_mid < p, low, mid))
                                pivot_sum = tl.where(s_mid > p, s_high, tl.where(s_mid < p, s_low, s_mid))
                                minimum = tl.where(s_mid > p, m_high, tl.where(s_mid < p, m_low, m_mid))
                                num_min = tl.where(s_mid > p, n_high, tl.where(s_mid < p, n_low, n_mid))
                            if found_pivot == 0:
                                if pivot_sum >= p and pivot_sum - (minimum * num_min) < p:
                                    p_pivot = pivot
                                    min_larger_prob = minimum
                                    num_min_larger = num_min
                                    p_pivots_sum = pivot_sum
                                    found_pivot = 1
                                if pivot_sum > p:
                                    min_range = pivot
                                elif pivot_sum < p:
                                    max_range = pivot
                                num_iters += 1
                                if (max_range - min_range) < 1e-9 or num_iters >= 18:
                                    p_pivot = (max_range + min_range) / 2.0
                                    min_larger_prob = minimum
                                    num_min_larger = num_min
                                    p_pivots_sum = pivot_sum
                                    found_pivot = 1
'''
segment = segment[:start_search] + replacement + segment[end_search:]
candidate_path = Path(__file__).with_name('topp_search_candidate_scratch.py')
candidate_path.write_text(code[:start] + segment + code[end:])
spec = importlib.util.spec_from_file_location('v16_topp_candidate', candidate_path)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
print('source', source, 'sha256', hashlib.sha256(source.read_bytes()).hexdigest(), flush=True)
print('scratch_sha256', hashlib.sha256(candidate_path.read_bytes()).hexdigest(), flush=True)

torch.manual_seed(8123)
device = 'cuda'
def equivalent(x, y):
    return torch.equal(x.view(torch.int32), y.view(torch.int32))

cases = []
for batch, vocab in [(1, 8191), (8, 8193), (8, 130560), (64, 130560)]:
    logits = torch.randn((batch, vocab), device=device, dtype=torch.float32) * 3
    cases.append((f'random_{batch}_{vocab}', logits, None, torch.full((batch,), .95, device=device)))
    cases.append((f'topp_one_{batch}_{vocab}', logits, None, torch.ones((batch,), device=device)))
    cases.append((f'topk_topp_{batch}_{vocab}', logits, torch.full((batch,), 50, device=device), torch.full((batch,), .9, device=device)))
edge = torch.randn((8, 8193), device=device)
edge[0, :20] = 0
edge[1, :20] = float('-inf')
edge[2, :] = float('-inf')
edge[3, 0] = float('inf')
edge[4, 0] = float('nan')
edge[5, :64] = 1.0
edge[6, :64] = torch.linspace(1.0 - 1e-6, 1.0 + 1e-6, 64, device=device)
cases.append(('edge', edge, None, torch.tensor([.95, .5, .95, .95, .95, .2, 0.0, 1.0], device=device)))
for name, logits, k, p in cases:
    for repeat in range(2):
        ref = baseline.apply_top_k_top_p_triton(logits.clone(), k, p)
        got = candidate.apply_top_k_top_p_triton(logits.clone(), k, p)
        torch.cuda.synchronize()
        if not equivalent(ref, got):
            print('MISMATCH', name, repeat, int((ref.view(torch.int32) != got.view(torch.int32)).sum().item()), flush=True)
            raise SystemExit(1)
    print('PASS', name, flush=True)

for batch in (1, 8, 64):
    logits = torch.randn((batch, 130560), device=device) * 3
    p = torch.full((batch,), .95, device=device)
    times = [[], []]
    for i in range(16):
        for side in ((0, 1) if i % 2 == 0 else (1, 0)):
            x = logits.clone()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            (baseline if side == 0 else candidate).apply_top_k_top_p_triton(x, None, p)
            b.record()
            b.synchronize()
            if i >= 4:
                times[side].append(a.elapsed_time(b))
    print('TIMING', json.dumps({'batch': batch, 'vocab': 130560, 'baseline_ms': statistics.median(times[0]), 'candidate_ms': statistics.median(times[1]), 'samples': len(times[0])}), flush=True)

