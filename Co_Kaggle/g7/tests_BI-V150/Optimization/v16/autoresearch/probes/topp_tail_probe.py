"""Bounded data-dependence screen for the unchanged top-p kernel."""
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from vllm.v1.sample.ops import topk_topp_triton as original

torch.manual_seed(8123)
print('source_sha256', hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(), flush=True)
for batch in (1, 64):
    shape = (batch, 130560)
    random = torch.randn(shape, device='cuda') * 3
    equal = torch.zeros(shape, device='cuda')
    dominant = torch.zeros(shape, device='cuda')
    dominant[:, 0] = 20.0
    inputs = {'normal_scale3': random, 'equal': equal, 'dominant': dominant}
    p = torch.full((batch,), .95, device='cuda')
    times = {name: [] for name in inputs}
    kept = {}
    names = list(inputs)
    for iteration in range(8):
        for name in names[iteration % len(names):] + names[:iteration % len(names)]:
            x = inputs[name].clone()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            y = original.apply_top_k_top_p_triton(x, None, p)
            b.record()
            b.synchronize()
            if iteration >= 2:
                times[name].append(a.elapsed_time(b))
            if iteration == 0:
                kept[name] = int(torch.isfinite(y).sum().item())
    print(json.dumps({'batch': batch, 'vocab': shape[1], 'p': .95, 'median_ms': {k: statistics.median(v) for k,v in times.items()}, 'samples_ms': times, 'kept_counts': kept}), flush=True)

