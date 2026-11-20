"""Host-side measurement and reporting for Group9; GPU imported lazily."""
from __future__ import annotations

import csv
import json
import math
import platform
import socket
import statistics
import sys
import time
from pathlib import Path


def runtime():
    try:
        import torch
        import triton
        import triton.testing  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Requires PyTorch and Triton in the BI-V150 environment') from exc
    if not torch.cuda.is_available():
        raise SystemExit('No compatible GPU visible via torch.cuda')
    return torch, triton


def compiler_metadata(handles):
    if not isinstance(handles, (tuple, list)):
        handles = (handles,)
    output = {}
    for key, extractor in {
        'registers_per_thread': lambda h: getattr(h, 'n_regs', None),
        'spills': lambda h: getattr(h, 'n_spills', None),
        'shared_bytes': lambda h: getattr(getattr(h, 'metadata', None), 'shared', None),
        'backend_warp_size': lambda h: getattr(getattr(h, 'metadata', None), 'warp_size', None),
    }.items():
        output[key] = ';'.join('' if extractor(h) is None else str(extractor(h)) for h in handles)
    return output


def dump_asm(handles, folder, stem):
    if not folder:
        return
    if not isinstance(handles, (tuple, list)):
        handles = (handles,)
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    for i, handle in enumerate(handles):
        for name, content in (getattr(handle, 'asm', None) or {}).items():
            if isinstance(content, str):
                (path / f'{stem}_k{i}.{name}.txt').write_text(content, encoding='utf-8')


def gpu_event_timing(launches: dict, *, trials: int, warmup_ms: int, rep_ms: int):
    """GPU-event timing; not a measurement of Python dispatch overhead."""
    import triton.testing
    history = {name: [] for name in launches}
    names = list(launches)
    for trial in range(trials):
        order = names[trial % len(names):] + names[:trial % len(names)]
        for name in order:
            elapsed_ms = float(triton.testing.do_bench(
                launches[name], warmup=warmup_ms, rep=rep_ms))
            if not (0 < elapsed_ms and math.isfinite(elapsed_ms)):
                raise ValueError(f'Invalid event result for {name}: {elapsed_ms}')
            history[name].append(elapsed_ms * 1000)
    return {
        name: {
            'gpu_event_us': statistics.median(v),
            'gpu_event_trials_us': ';'.join(f'{x:.4f}' for x in v),
            'gpu_event_mad_us': statistics.median(abs(x - statistics.median(v)) for x in v),
        }
        for name, v in history.items()
    }


def sync_wall_timing(torch, launch, *, batch: int, trials: int):
    """Synchronized end-to-end wall time per invocation; includes Python enqueue.

    A synchronization at the end of each batch prevents backlog. This measures
    host + GPU throughput together and must NOT be interpreted as pure launch
    latency. Each callback must be independent across invocations.
    """
    records = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for __ in range(batch):
            launch()
        torch.cuda.synchronize()
        records.append((time.perf_counter_ns() - start) / 1000.0 / batch)
    return {'sync_wall_us_per_call': statistics.median(records),
            'sync_wall_trials_us': ';'.join(f'{x:.4f}' for x in records),
            'host_batch': batch}


def check_cpu(torch, gpu, cpu_reference, *, atol=6e-5, rtol=5e-4):
    import numpy as np
    observed = gpu.detach().cpu().numpy()
    expected = np.asarray(cpu_reference, dtype=np.float32)
    if observed.shape != expected.shape or not np.all(np.isfinite(observed)):
        raise AssertionError('Output shape mismatch or nonfinite data')
    diff = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
    max_abs = float(diff.max())
    if not np.allclose(observed, expected, atol=atol, rtol=rtol):
        mismatch = int(np.argmax(diff))
        raise AssertionError(
            f'CPU reference mismatch at flattened index {mismatch}: '
            f'GPU={observed.flat[mismatch]:.8g}, CPU={expected.flat[mismatch]:.8g}, '
            f'max_abs={max_abs:.6g}; first_gpu={observed.flat[:6].tolist()}, '
            f'first_cpu={expected.flat[:6].tolist()}')
    return {'max_abs_error': max_abs}


def write_rows(path, records):
    if not records:
        raise ValueError('No output records')
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(name for rec in records for name in rec))
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f'CSV: {output} ({len(records)} rows)', flush=True)


def write_environment(path, torch, triton, args):
    dest = Path(path).with_suffix('.env.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'host': socket.gethostname(), 'system': platform.platform(),
        'python': sys.version.split()[0], 'torch': torch.__version__,
        'triton': triton.__version__, 'gpu': torch.cuda.get_device_name(0),
        'options': vars(args),
        'methodology': 'GPU event and synchronized wall metrics are distinct; '
                       'graph capture is optional; references computed on CPU.',
    }
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
