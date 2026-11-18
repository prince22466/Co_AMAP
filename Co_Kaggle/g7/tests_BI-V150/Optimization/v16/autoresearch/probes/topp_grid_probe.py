"""Scratch-only bounded per-row program grid: compare against installed kernel."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import torch
from vllm.v1.sample.ops import topk_topp_triton as baseline

source = Path(baseline.__file__)
code = source.read_text()
old_grid = '    NUM_PROGRAMS = min(num_sm, batch_size)'
old_alloc = '        size = min(next_power_of_2(NUM_PROGRAMS), num_sm)'
assert code.count(old_grid) == 1 and code.count(old_alloc) == 1
code = code.replace(old_grid, '    NUM_PROGRAMS = min(4 * num_sm, batch_size)')
code = code.replace(old_alloc, '        size = min(next_power_of_2(NUM_PROGRAMS), 4 * num_sm)')
candidate_path = Path(__file__).with_name('topp_grid_candidate_scratch.py')
candidate_path.write_text(code)
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

for batch in (8, 64):
    graph_input = torch.randn((batch, 8193), device=device)
    graph_p = torch.full((batch,), .95, device=device)
    graph_buffer = graph_input.clone()
    candidate.apply_top_k_top_p_triton(graph_buffer, None, graph_p)
    graph_buffer.copy_(graph_input)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate.apply_top_k_top_p_triton(graph_buffer, None, graph_p)
    for replay in range(3):
        next_input = torch.randn_like(graph_input) if replay != 1 else torch.zeros_like(graph_input)
        expected = baseline.apply_top_k_top_p_triton(next_input.clone(), None, graph_p)
        graph_buffer.copy_(next_input)
        graph.replay()
        torch.cuda.synchronize()
        if not equivalent(expected, graph_buffer):
            print('GRAPH_MISMATCH', batch, replay, flush=True)
            raise SystemExit(1)
    print('GRAPH_PASS', batch, flush=True)

for batch in (1, 8, 64, 256):
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
for label, module in (('baseline', baseline), ('candidate', candidate)):
    for key, buf in module._TRITON_BUFFER_CACHE.items():
        print('BUFFER', label, str(key), list(buf.shape), buf.numel() * buf.element_size(), flush=True)

