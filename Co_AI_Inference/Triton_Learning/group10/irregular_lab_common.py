"""Group10 timing/reporting helpers; GPU imports happen only on invocation."""
from __future__ import annotations

import csv
import json
import math
import platform
import socket
import statistics
import sys
from pathlib import Path


def require_gpu():
    try:
        import torch
        import triton
        import triton.testing  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Group10 requires torch + Triton in the BI-V150 environment') from exc
    if not torch.cuda.is_available():
        raise SystemExit('No compatible GPU is visible via torch.cuda')
    return torch, triton


def check_gpu_result(gpu, reference, *, atol: float = 2e-4, rtol: float = 2e-4) -> dict:
    import numpy as np
    got = gpu.detach().to('cpu').numpy()
    ref = np.asarray(reference, dtype=np.float32)
    if got.shape != ref.shape:
        raise AssertionError(f'Shape mismatch GPU {got.shape}, reference {ref.shape}')
    if not np.all(np.isfinite(got)):
        raise AssertionError(f'Nonfinite/unwritten GPU output: {got.reshape(-1)[:12].tolist()}')
    difference = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    max_diff = float(difference.max()) if difference.size else 0.0
    if not np.allclose(got, ref, atol=atol, rtol=rtol):
        index = int(np.argmax(difference))
        raise AssertionError(
            f'CPU reference mismatch at flat index {index}: '
            f'GPU={got.reshape(-1)[index]:.9g} CPU={ref.reshape(-1)[index]:.9g}; '
            f'max_abs={max_diff:.6g}; first_GPU={got.reshape(-1)[:8].tolist()}; '
            f'first_CPU={ref.reshape(-1)[:8].tolist()}')
    return {'max_abs_error': max_diff}


def compiled_metadata(handles) -> dict:
    if not isinstance(handles, (list, tuple)):
        handles = (handles,)
    fields = {'registers_per_thread': [], 'spills': [], 'shared_bytes': [], 'backend_warp_size': []}
    for h in handles:
        metadata = getattr(h, 'metadata', None)
        for key, val in (
            ('registers_per_thread', getattr(h, 'n_regs', None)),
            ('spills', getattr(h, 'n_spills', None)),
            ('shared_bytes', getattr(metadata, 'shared', None)),
            ('backend_warp_size', getattr(metadata, 'warp_size', None)),
        ):
            fields[key].append('' if val is None else str(val))
    return {k: ';'.join(v) for k, v in fields.items()}


def dump_compiled(handles, folder: str | None, stem: str):
    if not folder:
        return
    if not isinstance(handles, (list, tuple)):
        handles = (handles,)
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    for i, handle in enumerate(handles):
        for key, value in (getattr(handle, 'asm', None) or {}).items():
            if isinstance(value, str):
                (path / f'{stem}_k{i}.{key}.txt').write_text(value, encoding='utf-8')


def paired_bench(launches: dict, *, trials: int, warmup_ms: int, rep_ms: int) -> dict:
    """Alternates variant order. GPU-event times exclude setup and preprocessing."""
    import triton.testing
    results = {name: [] for name in launches}
    names = list(launches)
    for trial in range(trials):
        order = names[trial % len(names):] + names[:trial % len(names)]
        for name in order:
            ms = float(triton.testing.do_bench(launches[name], warmup=warmup_ms, rep=rep_ms))
            if not math.isfinite(ms) or ms <= 0:
                raise RuntimeError(f'Invalid benchmark time: {name}={ms}')
            results[name].append(ms * 1000)
    return {
        key: {
            'latency_us': statistics.median(vals),
            'trial_times_us': ';'.join(f'{v:.4f}' for v in vals),
            'trial_mad_us': statistics.median(abs(v - statistics.median(vals)) for v in vals),
        }
        for key, vals in results.items()
    }


def write_csv(path: str, rows: list[dict]):
    if not rows:
        raise ValueError('No result records')
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'CSV: {out} ({len(rows)} rows)', flush=True)


def write_environment(path: str, torch, triton, args):
    out = Path(path).with_suffix('.env.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    info = {
        'host': socket.gethostname(), 'system': platform.platform(),
        'python': sys.version.split()[0],
        'torch': torch.__version__, 'triton': triton.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'arguments': vars(args),
        'methodology': 'CPU FP64 references; event-based GPU timing; '
                       'preprocessing excluded and must be amortized in applications.',
    }
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
