"""Host-side helpers for Group7 Triton GEMM experiments (torch+triton only)."""
from __future__ import annotations

import csv
import json
import math
import platform
import socket
import statistics
import sys
from pathlib import Path


def runtime():
    try:
        import torch
        import triton
        import triton.testing  # noqa: F401
    except ImportError as ex:
        raise SystemExit('Requires torch and Triton in the target GPU environment') from ex
    if not torch.cuda.is_available():
        raise SystemExit('No GPU visible to torch.cuda (run on BI-V150)')
    return torch, triton


def positive_list(spec: str, name: str, *, powers_of_two=False) -> list[int]:
    vals = []
    for token in spec.split(','):
        try:
            v = int(token.strip())
        except ValueError as ex:
            raise ValueError(f'Invalid {name}: {token!r}') from ex
        if v < 1 or (powers_of_two and v & (v - 1)):
            raise ValueError(f'{name} must contain positive {"powers of two" if powers_of_two else "integers"}: {v}')
        if v not in vals:
            vals.append(v)
    if not vals:
        raise ValueError(f'Empty {name}')
    return vals


def parse_tiles(spec: str) -> list[tuple[int, int, int]]:
    res = []
    for part in spec.split(','):
        xyz = part.lower().strip().split('x')
        if len(xyz) != 3:
            raise ValueError(f'Expected BMxBNxBK, got {part!r}')
        values = tuple(int(x) for x in xyz)
        if any(x < 16 or x & (x - 1) for x in values):
            raise ValueError(f'Tile dimensions must be powers of two >= 16: {part!r}')
        if values not in res:
            res.append(values)
    if not res:
        raise ValueError('No tile configurations supplied')
    return res


def group_tile_coords(pid: int, m_tiles: int, n_tiles: int, group_m: int) -> tuple[int, int]:
    """CPU scheduling reference; also testable without a GPU."""
    if not (0 <= pid < m_tiles * n_tiles and min(m_tiles, n_tiles, group_m) > 0):
        raise ValueError('Invalid tile ID or geometry')
    group_span = group_m * n_tiles
    group_id = pid // group_span
    first_m = group_id * group_m
    group_actual = min(group_m, m_tiles - first_m)
    local = pid % group_span
    return first_m + (local % group_actual), local // group_actual


def split_k_intervals(k: int, bk: int, splits: int):
    """Equal-size, BK-aligned disjoint intervals; final interval(s) may be empty."""
    if min(k, bk, splits) < 1:
        raise ValueError('k/bk/splits must be positive')
    length = math.ceil(k / (bk * splits)) * bk
    return [(min(k, i * length), min(k, (i + 1) * length)) for i in range(splits)]


def persistent_tiles(total: int, workers: int):
    if total < 1 or workers < 1:
        raise ValueError('positive total and workers required')
    return [[i for i in range(worker, total, workers)] for worker in range(min(total, workers))]


def compiled_info(handle):
    md = getattr(handle, 'metadata', None)
    return dict(registers_per_thread=getattr(handle, 'n_regs', ''),
                spills=getattr(handle, 'n_spills', ''),
                shared_bytes=getattr(md, 'shared', ''),
                backend_warp_size=getattr(md, 'warp_size', ''))


def dump_kernel(handle, directory: str | None, stem: str):
    if not directory:
        return
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for name, data in (getattr(handle, 'asm', None) or {}).items():
        if isinstance(data, str):
            (target / f'{stem}.{name}.txt').write_text(data, encoding='utf-8')


def bench_alternating(launches: dict, *, trials: int, warmup_ms: int, rep_ms: int):
    """Alternate order; return median GPU-event timing in microseconds."""
    import triton.testing
    keys = list(launches)
    history = {key: [] for key in keys}
    for trial in range(trials):
        order = keys[trial % len(keys):] + keys[:trial % len(keys)]
        for key in order:
            value = float(triton.testing.do_bench(
                launches[key], warmup=warmup_ms, rep=rep_ms))
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError(f'Nonpositive or invalid GPU time: {key}={value}')
            history[key].append(value * 1000.0)
    return {
        key: {
            'latency_us': statistics.median(vals),
            'trial_latencies_us': ';'.join(f'{v:.3f}' for v in vals),
            'timing_mad_us': statistics.median(abs(v - statistics.median(vals)) for v in vals),
        }
        for key, vals in history.items()
    }


def check_gemm(torch, got, expected, *, atol=0.02, rtol=0.006):
    torch.cuda.synchronize()
    if not torch.isfinite(got).all().item():
        raise AssertionError('GEMM output contains NaN/Inf')
    deviation = (got - expected).abs()
    max_abs = float(deviation.max().item())
    denom = expected.abs().clamp_min(1.0)
    max_scaled = float((deviation / denom).max().item())
    torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
    return {'max_abs_error': max_abs, 'max_scaled_error': max_scaled}


def save_csv(path, records):
    if not records:
        raise ValueError('No result rows')
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(col for item in records for col in item))
    with p.open('w', newline='', encoding='utf-8') as f:
        out = csv.DictWriter(f, cols)
        out.writeheader()
        out.writerows(records)
    print(f'Wrote {p} ({len(records)} rows)', flush=True)


def save_environment(path, torch, triton, params):
    p = Path(path).with_suffix('.env.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    env = dict(hostname=socket.gethostname(), system=platform.platform(),
               python=sys.version.split()[0], torch=torch.__version__,
               triton=triton.__version__, gpu=torch.cuda.get_device_name(0),
               params=vars(params),
               notes='Logical TFLOPS includes full two-launch cost for split-K; not measured hardware FLOPs.')
    p.write_text(json.dumps(env, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
