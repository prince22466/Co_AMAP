#!/usr/bin/env python3
"""Group9: launch scaling, fusion, optional graph replay, persistent work grid.

This is a standalone educational microbenchmark, NOT a vLLM/competition
benchmark. CPU references avoid unsupported device-side FP64 paths on BI-V150.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from execution_lab_common import (
    check_cpu, compiler_metadata, dump_asm, gpu_event_timing, runtime,
    sync_wall_timing, write_environment, write_rows,
)
from execution_reference import (
    cpu_affine_sequence, cpu_fusion, cpu_launch, cpu_persistent,
    parse_positive_list,
)

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise SystemExit('Run this GPU lab in an environment with Triton installed') from exc


@triton.jit
def tiny_launch_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(X + off, off < N, other=0)
    tl.store(Y + off, a + 0.25, off < N)


@triton.jit
def fusion_stage_a(X, TEMP, N: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + off, off < N, other=0)
    tl.store(TEMP + off, x * 1.125 + 0.25, off < N)


@triton.jit
def fusion_stage_b(X, TEMP, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + off, off < N, other=0)
    t = tl.load(TEMP + off, off < N, other=0)
    tl.store(Y + off, t * t + x, off < N)


@triton.jit
def fusion_combined(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + off, off < N, other=0)
    t = x * 1.125 + 0.25
    tl.store(Y + off, t * t + x, off < N)


@triton.jit
def affine_step_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + off, off < N, other=0)
    tl.store(Y + off, x * 0.987 + 0.002, off < N)


@triton.jit
def grid_work_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(X + off, off < N, other=0)
    for _ in range(ITERS):
        acc = tl.fma(acc, 0.999, 0.0001)
    tl.store(Y + off, acc, off < N)


@triton.jit
def persistent_work_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr,
                           ITERS: tl.constexpr, NUM_TILES: tl.constexpr):
    """Bounded worker grid; each worker handles multiple independent tiles."""
    tile = tl.program_id(0)
    while tile < NUM_TILES:
        off = tile * BLOCK + tl.arange(0, BLOCK)
        acc = tl.load(X + off, off < N, other=0)
        for _ in range(ITERS):
            acc = tl.fma(acc, 0.999, 0.0001)
        tl.store(Y + off, acc, off < N)
        tile += tl.num_programs(0)


@dataclass
class Case:
    lab: str
    variant: str
    launch: Callable
    output: object
    reference: object
    handles: tuple = ()
    extra: dict = field(default_factory=dict)


def make_data(torch, n: int, seed: int):
    rng = np.random.default_rng(seed)
    cpu = rng.uniform(-0.7, 0.7, size=n).astype(np.float32)
    return cpu, torch.from_numpy(cpu.copy()).to('cuda')


def l01(torch, a):
    cpu, x = make_data(torch, a.tiny_n, a.seed)
    y = torch.empty_like(x)
    grid = (triton.cdiv(a.tiny_n, a.block),)
    result = []
    for count in a.launches:
        def run(cnt=count):
            handles = []
            for _ in range(cnt):
                handles.append(tiny_launch_kernel[grid](
                    x, y, a.tiny_n, a.block, num_warps=a.warps))
            return tuple(handles)
        result.append(Case('L01', f'launches_{count}', run, y, cpu_launch(cpu),
                           extra={'n': a.tiny_n, 'launches': count,
                                  'logical_bytes': count * 2 * a.tiny_n * 4,
                                  'note': 'Work and memory traffic increase with launch count'}))
    return result


def l02(torch, a):
    cpu, x = make_data(torch, a.n, a.seed)
    tmp, out = torch.empty_like(x), torch.empty_like(x)
    grid = (triton.cdiv(a.n, a.block),)

    def separate():
        first = fusion_stage_a[grid](x, tmp, a.n, a.block,
                                     num_warps=a.warps, enable_fp_fusion=False)
        second = fusion_stage_b[grid](x, tmp, out, a.n, a.block,
                                      num_warps=a.warps, enable_fp_fusion=False)
        return (first, second)

    def fused():
        handle = fusion_combined[grid](x, out, a.n, a.block,
                                       num_warps=a.warps, enable_fp_fusion=False)
        return (handle,)

    # Separate: 3 reads and 2 writes; fused: 1 read and 1 write.
    return [Case('L02', 'unfused', separate, out, cpu_fusion(cpu),
                 extra={'n': a.n, 'launches': 2, 'logical_bytes': a.n * 5 * 4}),
            Case('L02', 'fused', fused, out, cpu_fusion(cpu),
                 extra={'n': a.n, 'launches': 1, 'logical_bytes': a.n * 2 * 4})]


def l03(torch, a):
    cases = []
    for steps in a.graph_steps:
        cpu, x = make_data(torch, a.graph_n, a.seed)
        output_a, output_b = torch.empty_like(x), torch.empty_like(x)
        grid = (triton.cdiv(a.graph_n, a.block),)

        def eager(steps=steps, x=x, first=output_a, second=output_b,
                  grid=grid):
            source = x
            handles = []
            for i in range(steps):
                dest = first if i % 2 == 0 else second
                handles.append(affine_step_kernel[grid](
                    source, dest, a.graph_n, a.block, num_warps=a.warps,
                    enable_fp_fusion=False))
                source = dest
            return tuple(handles)

        result = output_a if steps % 2 else output_b
        ref = cpu_affine_sequence(cpu, steps)
        cases.append(Case('L03', f'eager_{steps}', eager, result, ref,
                          extra={'n': a.graph_n, 'steps': steps,
                                 'launches': steps,
                                 'logical_bytes': steps * a.graph_n * 8,
                                 'execution': 'eager'}))

        # Warmup and compile OUTSIDE graph capture. The graph keeps the same
        # tensor storage addresses across capture and replay.
        try:
            if not hasattr(torch.cuda, 'CUDAGraph') or not hasattr(torch.cuda, 'graph'):
                raise RuntimeError('torch.cuda.CUDAGraph / torch.cuda.graph is unavailable')
            eager()
            eager()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                eager()
            torch.cuda.synchronize()
        except Exception as exc:
            cases.append(Case('L03', f'graph_{steps}', None, None, None,
                              extra={'n': a.graph_n, 'steps': steps,
                                     'status': 'unsupported',
                                     'error': f'{type(exc).__name__}: {exc}',
                                     'execution': 'graph'}))
            continue

        # Important: closure defaults hold graph and buffers alive for replay.
        def replay(g=graph, _keep=(x, output_a, output_b)):
            g.replay()
            return ()

        # Verify replay actually *writes* the output rather than passing using
        # data left over from eager warmup/capture (empty graph is not a pass).
        try:
            result.fill_(float('nan'))
            replay()
            torch.cuda.synchronize()
        except Exception as exc:
            cases.append(Case('L03', f'graph_{steps}', None, None, None,
                              extra={'n': a.graph_n, 'steps': steps,
                                     'status': 'unsupported',
                                     'error': f'Graph replay unavailable: {type(exc).__name__}: {exc}',
                                     'execution': 'graph'}))
            continue
        # A successfully replayed graph with incorrect output is a real error,
        # not evidence of "unsupported" graph capture.
        check_cpu(torch, result, ref)
        cases.append(Case('L03', f'graph_{steps}', replay, result, ref,
                          extra={'n': a.graph_n, 'steps': steps,
                                 'launches': steps,
                                 'logical_bytes': steps * a.graph_n * 8,
                                 'execution': 'graph'}))
    return cases


def l04(torch, a):
    cpu, x = make_data(torch, a.n, a.seed)
    y = torch.empty_like(x)
    grid_tiles = triton.cdiv(a.n, a.block)
    reference = cpu_persistent(cpu, a.iters)
    def conventional():
        return (grid_work_kernel[(grid_tiles,)](
            x, y, a.n, a.block, a.iters, num_warps=a.warps),)
    cases = [Case('L04', 'grid', conventional, y, reference,
                  extra={'n': a.n, 'tiles': grid_tiles, 'workers': grid_tiles,
                         'iterations': a.iters, 'logical_bytes': 8 * a.n})]
    for workers in a.workers:
        worker_ctas = min(workers, grid_tiles)
        def persistent(w=worker_ctas):
            return (persistent_work_kernel[(w,)](
                x, y, a.n, a.block, a.iters, grid_tiles, num_warps=a.warps),)
        cases.append(Case('L04', f'persistent_{workers}', persistent, y, reference,
                          extra={'n': a.n, 'tiles': grid_tiles,
                                 'workers': worker_ctas, 'iterations': a.iters,
                                 'logical_bytes': 8 * a.n}))
    return cases


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lab', choices=('all', 'L01', 'L02', 'L03', 'L04'), default='all')
    p.add_argument('--variant', help='Run one variant, e.g. fused, graph_4, persistent_64')
    p.add_argument('--n', type=int, default=262144, help='Number of FP32 elements for L02/L04')
    p.add_argument('--tiny-n', type=int, default=512, help='Tiny tensor size for L01')
    p.add_argument('--graph-n', type=int, default=4096, help='FP32 elements for L03')
    p.add_argument('--block', type=int, default=256)
    p.add_argument('--warps', type=int, default=4)
    p.add_argument('--launches', default='1,2,4,8')
    p.add_argument('--graph-steps', default='1,4,8')
    p.add_argument('--workers', default='16,32,64')
    p.add_argument('--iters', type=int, default=64, help='FMA steps per output in L04')
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--warmup-ms', type=int, default=40)
    p.add_argument('--rep-ms', type=int, default=120)
    p.add_argument('--host-batch', type=int, default=32,
                   help='Number of calls between syncs for wall-time measurement')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--single-run', action='store_true', help='GPU compile + correctness, no timing')
    p.add_argument('--strict', action='store_true', help='Stop on kernel/correctness failure')
    p.add_argument('--require-graph', action='store_true', help='Fail if requested graph unsupported')
    p.add_argument('--dump-asm', metavar='DIR', default=None)
    p.add_argument('--out', default='results/group9.csv')
    a = p.parse_args()
    for name in ('n', 'tiny_n', 'graph_n', 'block', 'warps', 'iters', 'trials',
                 'warmup_ms', 'rep_ms', 'host_batch'):
        if getattr(a, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be > 0')
    if a.block & (a.block - 1) or a.warps & (a.warps - 1):
        p.error('--block and --warps must be powers of two')
    try:
        a.launches = parse_positive_list(a.launches, 'launches')
        a.graph_steps = parse_positive_list(a.graph_steps, 'graph-steps')
        a.workers = parse_positive_list(a.workers, 'workers')
    except ValueError as exc:
        p.error(str(exc))
    if a.lab == 'all' and a.variant:
        p.error('--variant requires a specific --lab')
    return a


def execute(a, torch):
    makers = {'L01': l01, 'L02': l02, 'L03': l03, 'L04': l04}
    records = []
    failures = []
    for lab in (makers if a.lab == 'all' else [a.lab]):
        print(f'-- {lab} --', flush=True)
        # A supported experiment failure should not silently become a skip.
        try:
            cases = makers[lab](torch, a)
        except Exception as exc:
            print(f'FAIL {lab} setup: {type(exc).__name__}: {exc}', flush=True)
            if a.strict:
                write_rows(a.out, records or [{'lab': lab, 'status': 'error', 'error': str(exc)}])
                raise
            records.append({'lab': lab, 'variant': 'setup', 'status': 'error',
                            'error': f'{type(exc).__name__}: {exc}'})
            failures.append(lab)
            continue
        if a.variant:
            cases = [case for case in cases if case.variant == a.variant]
            if not cases:
                raise SystemExit(f'Unknown variant {a.variant!r} for {lab}')
        timed = {}
        pending = []
        for case in cases:
            row = {'lab': case.lab, 'variant': case.variant, 'block': a.block,
                   'warps': a.warps, **case.extra}
            if case.launch is None:
                row['status'] = 'unsupported'
                print(f'SKIP {lab} {case.variant}: {row.get("error", "unsupported")}', flush=True)
                records.append(row)
                if a.require_graph:
                    failures.append(case.variant)
                continue
            try:
                # Reset buffers so that missing writes never appear valid.
                case.output.fill_(float('nan'))
                handles = case.launch() or ()
                torch.cuda.synchronize()
                row.update(check_cpu(torch, case.output, case.reference))
                row.update(compiler_metadata(handles if handles else case.handles))
                if handles:
                    dump_asm(handles, a.dump_asm, f'{lab}_{case.variant}')
                row['status'] = 'ok'
                pending.append((case, row))
                timed[case.variant] = case.launch
                print(f'PASS {lab} {case.variant}: max_abs_error={row["max_abs_error"]:.5g}', flush=True)
            except Exception as exc:
                row.update(status='error', error=f'{type(exc).__name__}: {exc}')
                print(f'FAIL {lab} {case.variant}: {row["error"]}', flush=True)
                failures.append(case.variant)
                records.append(row)
                if a.strict:
                    write_rows(a.out, records)
                    raise
        if pending and not a.single_run:
            try:
                times = gpu_event_timing(timed, trials=a.trials,
                                         warmup_ms=a.warmup_ms, rep_ms=a.rep_ms)
                for case, row in pending:
                    row.update(times[case.variant])
                    row.update(sync_wall_timing(torch, case.launch,
                                               batch=a.host_batch, trials=a.trials))
                    if 'launches' in row:
                        row['gpu_event_us_per_launch'] = row['gpu_event_us'] / row['launches']
                    print(f'TIMING {lab} {case.variant}: event={row["gpu_event_us"]:.3f}us '
                          f'wall={row["sync_wall_us_per_call"]:.3f}us/call', flush=True)
            except Exception as exc:
                # Preserve correctness rows and the timing failure in the CSV.
                for _, row in pending:
                    if 'gpu_event_us' not in row or 'sync_wall_us_per_call' not in row:
                        row.update(status='error', error=f'Timing: {type(exc).__name__}: {exc}')
                        failures.append(row['variant'])
                print(f'FAIL {lab} timing: {type(exc).__name__}: {exc}', flush=True)
                records.extend(row for _, row in pending)
                if a.strict:
                    write_rows(a.out, records)
                    raise
                continue
            by_variant = {row['variant']: row for _, row in pending}
            if lab == 'L02' and {'unfused', 'fused'} <= by_variant.keys():
                b = by_variant['unfused']['gpu_event_us']
                o = by_variant['fused']['gpu_event_us']
                for _, row in pending:
                    row['pair_speedup_event'] = b / o
                print(f'AB L02 fused speedup: {b/o:.3f}x', flush=True)
            if lab == 'L03':
                for _, row in pending:
                    if row['variant'].startswith('graph_'):
                        counterpart = 'eager_' + row['variant'].split('_')[1]
                        if counterpart in by_variant:
                            speedup = by_variant[counterpart]['sync_wall_us_per_call'] / row['sync_wall_us_per_call']
                            row['pair_speedup_wall'] = speedup
                            print(f'AB L03 {row["variant"]} synced-wall speedup: {speedup:.3f}x', flush=True)
            if lab == 'L04' and 'grid' in by_variant:
                baseline = by_variant['grid']['gpu_event_us']
                for _, row in pending:
                    row['speedup_vs_grid_event'] = baseline / row['gpu_event_us']
        records.extend(row for _, row in pending)
    write_rows(a.out, records)
    if failures:
        raise SystemExit(f'Failed variants: {", ".join(failures)}')
    if not any(r.get('status') == 'ok' for r in records):
        raise SystemExit('No successful cases; see CSV for unsupported/errors')


def main():
    a = build_parser()
    torch, installed_triton = runtime()
    torch.cuda.set_device(0)
    print(f'GPU={torch.cuda.get_device_name(0)} '
          f'torch={torch.__version__} triton={installed_triton.__version__}', flush=True)
    write_environment(a.out, torch, installed_triton, a)
    execute(a, torch)


if __name__ == '__main__':
    main()
