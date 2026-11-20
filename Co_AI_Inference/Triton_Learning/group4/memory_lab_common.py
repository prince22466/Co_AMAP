"""Helpers for standalone Triton memory-latency A/B experiments.

This module is also usable without a GPU for CPU reference tests.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import socket
import statistics
import sys
from pathlib import Path

import numpy as np


def powers_of_two(value: str, name: str) -> list[int]:
    vals = []
    for s in value.split(','):
        try:
            n = int(s.strip())
        except ValueError as exc:
            raise ValueError(f'{name}: invalid integer {s!r}') from exc
        if n <= 0 or (n & (n - 1)):
            raise ValueError(f'{name}: values must be positive powers of two (got {n})')
        if n not in vals:
            vals.append(n)
    if not vals:
        raise ValueError(f'{name} must not be empty')
    return vals


def positive_ints(value: str, name: str) -> list[int]:
    vals = []
    for s in value.split(','):
        try:
            n = int(s.strip())
        except ValueError as exc:
            raise ValueError(f'{name}: invalid integer {s!r}') from exc
        if n <= 0:
            raise ValueError(f'{name}: values must be positive (got {n})')
        if n not in vals:
            vals.append(n)
    if not vals:
        raise ValueError(f'{name} must not be empty')
    return vals


def working_sets_mib(value: str) -> list[tuple[float, int]]:
    """Return (MiB, FP32 elements). Require a power-of-two number of elements."""
    result = []
    for item in value.split(','):
        try:
            mib = float(item.strip())
        except ValueError as exc:
            raise ValueError(f'Invalid working-set MiB: {item!r}') from exc
        if not math.isfinite(mib) or mib <= 0:
            raise ValueError('Working-set MiB must be positive and finite')
        count_float = mib * 1048576 / 4
        count = round(count_float)
        if not math.isclose(count_float, count, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f'Working-set size {mib} MiB is not an integer FP32 count')
        if count <= 0 or count & (count - 1):
            raise ValueError(f'Working-set size {mib} MiB must contain a power-of-two FP32 element count')
        if all(prev != count for _, prev in result):
            result.append((count * 4 / 1048576, count))
    if not result:
        raise ValueError('Working-set list is empty')
    return result


def stream_reference(pool: np.ndarray, n: int, reps: int, stride: int,
                     positions: np.ndarray, ws_entries: int) -> np.ndarray:
    """Float32 sequential sum reference for selected output positions."""
    assert ws_entries > 0 and (ws_entries & (ws_entries - 1)) == 0
    acc = np.zeros(len(positions), dtype=np.float32)
    for t in range(reps):
        ix = (positions.astype(np.int64) * stride + t * n) & (ws_entries - 1)
        acc = acc + pool[ix].astype(np.float32)
    return acc


def chase_reference(table: np.ndarray, chains: int, total_lookups: int,
                    positions: np.ndarray) -> np.ndarray:
    """Match the Triton pointer-chase starts/loads/reduction exactly."""
    assert total_lookups % chains == 0
    mask = len(table) - 1
    assert len(table) > 0 and (len(table) & mask) == 0
    out = np.zeros(len(positions), dtype=np.int64)
    for row in range(chains):
        ix = ((positions.astype(np.int64) + row * 65537) & mask).astype(np.int64)
        for _ in range(total_lookups // chains):
            ix = table[ix].astype(np.int64)
        out += ix
    return out.astype(np.int32)


def sample_positions(n: int) -> np.ndarray:
    s = list(range(min(n, 24)))
    for p in (n // 4, n // 2, (3 * n) // 4, n - 1):
        if p not in s:
            s.append(p)
    return np.asarray(s, dtype=np.int64)


def verify_selected(gpu_output, positions: np.ndarray, expected: np.ndarray, *, exact: bool):
    # Copy the small output vector to host first. Avoid creating an extra GPU
    # advanced-index kernel on backends with incomplete PyTorch indexing ops.
    observed = gpu_output.detach().cpu().numpy()[positions]
    if exact:
        if not np.array_equal(observed.astype(np.int64), expected.astype(np.int64)):
            different = np.flatnonzero(observed != expected)
            j = int(different[0])
            raise AssertionError(f'Index mismatch at output {positions[j]}: GPU={observed[j]}, CPU={expected[j]}')
    else:
        if not np.all(np.isfinite(observed)) or not np.allclose(observed, expected, rtol=2e-5, atol=2e-3):
            errors = np.abs(observed - expected)
            j = int(np.argmax(errors))
            raise AssertionError(f'FP32 mismatch at output {positions[j]}: GPU={observed[j]}, CPU={expected[j]}')


def gpu_info(torch, triton, params: dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = {
        'hostname': socket.gethostname(),
        'os': platform.platform(),
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'triton': triton.__version__,
        'device': torch.cuda.get_device_name(0),
        'params': params,
    }
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def metadata(kernel) -> dict:
    md = getattr(kernel, 'metadata', None)
    return {
        'registers_per_thread': getattr(kernel, 'n_regs', ''),
        'spills': getattr(kernel, 'n_spills', ''),
        'shared_bytes': getattr(md, 'shared', ''),
        'backend_warp_size': getattr(md, 'warp_size', ''),
    }


def dump_asm(kernel, folder: str | None, stem: str) -> None:
    if not folder:
        return
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    for fmt, code in getattr(kernel, 'asm', {}).items():
        if isinstance(code, str):
            (root / f'{stem}.{fmt}.txt').write_text(code, encoding='utf-8')


def benchmark_round_robin(cases: list[dict], *, warmup_ms: int, rep_ms: int, trials: int) -> None:
    """Interleave configs across trials to reduce fixed-order clock/thermal bias.

    CUDA event timing through Triton do_bench. Compilation and correctness are
    performed before this function; eviction is NOT done between timed calls.
    """
    import triton.testing
    histories: list[list[float]] = [[] for _ in cases]
    for trial in range(trials):
        order = range(len(cases)) if trial % 2 == 0 else range(len(cases) - 1, -1, -1)
        for j in order:
            case = cases[j]
            if case.get('status') != 'ready':
                continue
            try:
                p20, p50, p80 = triton.testing.do_bench(
                    case['launch'], warmup=warmup_ms, rep=rep_ms, quantiles=[0.2, 0.5, 0.8],
                )
                if not all(math.isfinite(float(t)) and float(t) > 0 for t in (p20, p50, p80)):
                    raise RuntimeError(f'Invalid event timing for {case["name"]}')
                histories[j].append(float(p50) * 1000)
            except Exception as exc:
                case.update(status='error', error=f'Event timing: {type(exc).__name__}: {exc}')
                print(f'FAILED timing {case["name"]}: {case["error"]}')
    for j, case in enumerate(cases):
        if histories[j]:
            us = statistics.median(histories[j])
            case.update(latency_us=us, trial_medians_us=';'.join(f'{x:.3f}' for x in histories[j]),
                        status='ok')
            read_bytes = case['logical_read_bytes']
            case['logical_read_gbps'] = read_bytes / (us * 1000.0)


def write_csv(path: str, cases: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in case.items() if k not in ('launch', 'name')}
            for case in cases]
    if not rows:
        raise RuntimeError('No results')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with out.open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f'CSV: {out} ({len(rows)} records)', flush=True)
