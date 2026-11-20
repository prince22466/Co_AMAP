"""Host-only helpers for Group8; safe to import without a GPU or Triton."""
from __future__ import annotations

import csv
import json
import math
import platform
import socket
import statistics
import sys
from pathlib import Path


def parse_list(value: str, name: str, *, power2: bool = True) -> list[int]:
    result = []
    for s in value.split(','):
        try:
            n = int(s.strip())
        except ValueError as exc:
            raise ValueError(f'Invalid {name} entry: {s!r}') from exc
        if n < 1 or (power2 and (n & (n - 1))):
            raise ValueError(f'{name} must be positive {"powers of two" if power2 else "integers"}, got {n}')
        if n not in result:
            result.append(n)
    if not result:
        raise ValueError(f'No {name} values')
    return result


def load_runtime():
    try:
        import torch
        import triton
        import triton.testing  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Requires torch and Triton on the target GPU runtime') from exc
    if not torch.cuda.is_available():
        raise SystemExit('No GPU reported by torch.cuda; run on BI-V150')
    return torch, triton


def metadata(handles) -> dict:
    """Return per-kernel metadata; blank fields mean unsupported/unreported."""
    if not isinstance(handles, (tuple, list)):
        handles = (handles,)
    fields = ('registers_per_thread', 'spills', 'shared_bytes', 'backend_warp_size')
    results = {key: [] for key in fields}
    for kernel in handles:
        md = getattr(kernel, 'metadata', None)
        items = (getattr(kernel, 'n_regs', ''), getattr(kernel, 'n_spills', ''),
                 getattr(md, 'shared', ''), getattr(md, 'warp_size', ''))
        for key, val in zip(fields, items):
            results[key].append('' if val is None else str(val))
    return {key: ';'.join(vals) for key, vals in results.items()}


def dump_asm(handles, folder: str | None, stem: str):
    if not folder:
        return
    if not isinstance(handles, (tuple, list)):
        handles = (handles,)
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    for index, handle in enumerate(handles):
        for name, val in (getattr(handle, 'asm', None) or {}).items():
            if isinstance(val, str):
                (out / f'{stem}_k{index}.{name}.txt').write_text(val, encoding='utf8')


def benchmark_alternating(launches, warmup_ms: int, rep_ms: int, trials: int):
    """GPU-event timing, alternating A/B order between trials."""
    import triton.testing
    keys = list(launches)
    samples = {key: [] for key in keys}
    for t in range(trials):
        order = keys[t % len(keys):] + keys[:t % len(keys)]
        for key in order:
            ms = float(triton.testing.do_bench(
                launches[key], warmup=warmup_ms, rep=rep_ms))
            if not (math.isfinite(ms) and ms > 0):
                raise RuntimeError(f'Invalid timing for {key}: {ms}')
            samples[key].append(ms * 1000)
    return {
        key: dict(latency_us=statistics.median(us),
                  trials_us=';'.join(f'{v:.3f}' for v in us),
                  mad_us=statistics.median(abs(v - statistics.median(us)) for v in us))
        for key, us in samples.items()
    }



def cpu_reference(torch, x, lab: str):
    """Independent CPU-FP64 reference; never invoke FP64 kernels on the GPU.

    Some Triton-supported accelerators have incomplete device-side FP64
    support.  Calling x.double().sum(...) on a device tensor can therefore
    produce an invalid benchmark reference even when the Triton result is fine.
    Move FP32 input to the host FIRST, then promote and compute in CPU FP64.
    Reference construction is outside GPU timing measurements.
    """
    if lab not in ('R01', 'R02', 'R03', 'R04'):
        raise ValueError(f'Unknown lab for reference: {lab}')
    host = x.detach().to(device='cpu', dtype=torch.float32)
    host64 = host.to(dtype=torch.float64)
    if lab in ('R01', 'R02'):
        expected_host = host64.sum(dim=1).to(dtype=torch.float32)
    else:
        expected_host = torch.softmax(host64, dim=-1).to(dtype=torch.float32)
    if not torch.isfinite(expected_host).all().item():
        raise AssertionError(f'Non-finite CPU reference for {lab}')
    return expected_host.to(device=x.device)


def check_result(torch, got, expected, *, atol: float, rtol: float) -> dict:
    torch.cuda.synchronize()
    if not torch.isfinite(got).all().item():
        raise AssertionError('Output contains NaN/Inf')
    err = (got - expected).abs()
    try:
        torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        # Include concrete output samples so a backend write/lowering failure
        # can be distinguished from ordinary summation error on the GPU host.
        actual_sample = got.reshape(-1)[:8].detach().cpu().tolist()
        reference_sample = expected.reshape(-1)[:8].detach().cpu().tolist()
        raise AssertionError(
            f'{exc}\nFirst output values: {actual_sample}\n'
            f'First reference values: {reference_sample}'
        ) from exc
    return {'max_abs_error': float(err.max().item()),
            'max_relative_error': float((err / expected.abs().clamp_min(atol)).max().item())}


def write_csv(path: str, rows: list[dict]):
    if not rows:
        raise ValueError('No result rows')
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with p.open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {p} ({len(rows)} rows)', flush=True)


def write_environment(path: str, torch, triton, args):
    p = Path(path).with_suffix('.env.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    info = dict(hostname=socket.gethostname(), system=platform.platform(),
                python=sys.version.split()[0], torch=torch.__version__,
                triton=triton.__version__, gpu=torch.cuda.get_device_name(0),
                options=vars(args),
                note='Logical requested bytes are not measured DRAM traffic; '
                     'warp placement and shared memory require compiler/counter verification.')
    p.write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
