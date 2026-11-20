#!/usr/bin/env python3
"""Group8: reductions and stable softmax Triton A/B microbenchmarks.

R01: serial scalar scan versus parallel tree sum (also changes memory coalescing).
R02: flat tree sum versus a logical chunk-first, cross-chunk sum.
R03: three-kernel staged softmax versus one-kernel fused softmax.
R04: full-row fused softmax versus streaming online-statistics softmax.

Standalone teaching benchmarks only. No changes to MiniCPM, vLLM, FlagGems,
or competition benchmark scripts. Run in a PyTorch + Triton GPU environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from reduction_lab_common import (
    benchmark_alternating, check_result, cpu_reference, dump_asm, load_runtime,
    metadata, parse_list, write_csv, write_environment,
)

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as ex:
    raise SystemExit('Run on a GPU host with torch and Triton installed') from ex


# Execute independent row-wise serial chains in one CTA.  Each lane keeps a
# dependent accumulator for *one row*; there is no parallel reduction across
# a row.  Avoid the scalar tl.load/tl.store path that produced incorrect results
# in the BI-V150 Triton 3.6 run reported on 2026-11-20.
SERIAL_ROWS_PER_CTA = 64


@triton.jit
def reduce_serial_kernel(X, Y, ROWS: tl.constexpr, COLS: tl.constexpr,
                         ROW_BLOCK: tl.constexpr):
    """Sequential reduction per row, vectorized across independent rows.

    Each output lane maintains a *serial* dependency chain for one row.  The
    competing tree kernel reduces lanes within a row, so the comparison still
    differs in grid size, memory coalescing and occupancy; this is deliberate.
    """
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    valid = row < ROWS
    acc = tl.full((ROW_BLOCK,), 0.0, tl.float32)
    for c in range(COLS):
        v = tl.load(X + row * COLS + c, mask=valid, other=0.0)
        acc = acc + v
    tl.store(Y + row, acc, mask=valid)


@triton.jit
def reduce_tree_kernel(X, Y, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    vals = tl.load(X + row * COLS + c, mask=c < COLS, other=0)
    total = tl.sum(vals, axis=0)
    tl.store(Y + row, total)


@triton.jit
def reduce_chunked_kernel(X, Y, COLS: tl.constexpr,
                          CHUNK: tl.constexpr, GROUPS: tl.constexpr):
    """First reduce chunks, then their partials (logical two-level tree)."""
    row = tl.program_id(0)
    g = tl.arange(0, GROUPS)
    lane = tl.arange(0, CHUNK)
    c = g[:, None] * CHUNK + lane[None, :]
    vals = tl.load(X + row * COLS + c, mask=c < COLS, other=0)
    partial = tl.sum(vals, axis=1)
    result = tl.sum(partial, axis=0)
    tl.store(Y + row, result)


@triton.jit
def row_max_kernel(X, ROW_MAX, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + row * COLS + c, c < COLS, other=-float('inf'))
    tl.store(ROW_MAX + row, tl.max(x, axis=0))


@triton.jit
def row_sumexp_kernel(X, ROW_MAX, ROW_SUM,
                      COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = tl.load(ROW_MAX + row)
    x = tl.load(X + row * COLS + c, c < COLS, other=-float('inf'))
    z = tl.exp(x - m)
    tl.store(ROW_SUM + row, tl.sum(z, axis=0))


@triton.jit
def row_normalize_kernel(X, ROW_MAX, ROW_SUM, Y,
                         COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = tl.load(ROW_MAX + row)
    s = tl.load(ROW_SUM + row)
    x = tl.load(X + row * COLS + c, c < COLS, other=-float('inf'))
    prob = tl.exp(x - m) / s
    tl.store(Y + row * COLS + c, prob, c < COLS)


@triton.jit
def softmax_fused_kernel(X, Y, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + row * COLS + c, c < COLS, other=-float('inf'))
    z = tl.exp(x - tl.max(x, axis=0))
    s = tl.sum(z, axis=0)
    tl.store(Y + row * COLS + c, z / s, c < COLS)


@triton.jit
def softmax_online_kernel(X, Y, COLS: tl.constexpr,
                          CHUNK: tl.constexpr, NUM_CHUNKS: tl.constexpr):
    """Streaming stable (max, exp-sum), then reread X for normalized output.

    This is a deliberately simple two-pass online algorithm, not FlashAttention
    or an async pipeline. It trades additional global reads for lower live tile
    state on long rows.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, CHUNK)
    m = tl.full((), -float('inf'), tl.float32)
    denom = tl.full((), 0.0, tl.float32)
    for block in range(NUM_CHUNKS):
        c = block * CHUNK + lane
        x = tl.load(X + row * COLS + c, c < COLS, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, block_max)
        block_sum = tl.sum(tl.exp(x - new_m), axis=0)
        denom = denom * tl.exp(m - new_m) + block_sum
        m = new_m
    for block in range(NUM_CHUNKS):
        c = block * CHUNK + lane
        x = tl.load(X + row * COLS + c, c < COLS, other=-float('inf'))
        result = tl.exp(x - m) / denom
        tl.store(Y + row * COLS + c, result, mask=c < COLS)


DEFAULT_SHAPES = {
    'R01': (128, 256),
    'R02': (512, 1024),
    'R03': (512, 1024),
    'R04': (128, 4096),
}
VARIANTS = {
    'R01': ('serial', 'tree'),
    'R02': ('flat', 'chunked'),
    'R03': ('staged', 'fused'),
    'R04': ('fused', 'online'),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lab', choices=['all', *DEFAULT_SHAPES], default='all')
    p.add_argument('--variant', help='Run only one variant (useful for profiler captures)')
    p.add_argument('--rows', type=int, default=0, help='0 selects each lab default')
    p.add_argument('--cols', type=int, default=0, help='0 selects each lab default')
    p.add_argument('--warps', default='4', help='Comma-separated power-of-two warp counts')
    p.add_argument('--chunks', default='32,64,128', help='R02 logical chunk sizes')
    p.add_argument('--stream-chunks', default='128,256,512', help='R04 streaming chunks')
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--warmup-ms', type=int, default=30)
    p.add_argument('--rep-ms', type=int, default=100)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--single-run', action='store_true',
                   help='Compile + correctness only; no timing')
    p.add_argument('--strict', action='store_true',
                   help='Abort immediately if a configuration errors')
    p.add_argument('--dump-asm', metavar='DIR', help='Write textual compiler output')
    p.add_argument('--out', default='results/reduction.csv')
    a = p.parse_args()
    if a.rows < 0 or a.cols < 0 or a.trials < 1 or a.warmup_ms < 1 or a.rep_ms < 1:
        p.error('rows/cols must be >=0, trials/warmup-ms/rep-ms must be >0')
    a.warp_options = parse_list(a.warps, 'warps')
    a.chunk_options = parse_list(a.chunks, 'chunks')
    a.stream_options = parse_list(a.stream_chunks, 'stream-chunks')
    if a.variant and a.lab == 'all':
        p.error('--variant requires a single --lab')
    if a.variant and a.variant not in VARIANTS[a.lab]:
        p.error(f'--variant for {a.lab} must be one of {VARIANTS[a.lab]}')
    if (a.rows and a.rows > 65536) or (a.cols and a.cols > 65536):
        p.error('For this teaching lab, rows and cols must be <= 65536')
    return a


def make_case(torch_mod, rows: int, cols: int, seed: int):
    g = torch_mod.Generator(device='cpu').manual_seed(seed)
    host = torch_mod.randn((rows, cols), generator=g, dtype=torch_mod.float32) * 1.25
    # Stress numerically stable softmax, without relying on implicit random extremes.
    host[::11, 0] = 12.0
    if cols > 1:
        host[::13, 1] = -12.0
    return host.to('cuda')


def launch_factory(lab, variant, x, y, rows, cols, block, warp, chunk, scratch):
    grid = (rows,)
    if lab == 'R01':
        if variant == 'serial':
            serial_grid = (triton.cdiv(rows, SERIAL_ROWS_PER_CTA),)
            return lambda: reduce_serial_kernel[serial_grid](
                x, y, rows, cols, SERIAL_ROWS_PER_CTA, num_warps=warp)
        return lambda: reduce_tree_kernel[grid](x, y, cols, block, num_warps=warp)
    if lab == 'R02':
        if variant == 'flat':
            return lambda: reduce_tree_kernel[grid](x, y, cols, block, num_warps=warp)
        return lambda: reduce_chunked_kernel[grid](
            x, y, cols, chunk, block // chunk, num_warps=warp)
    if lab == 'R03':
        if variant == 'fused':
            return lambda: softmax_fused_kernel[grid](x, y, cols, block, num_warps=warp)
        row_max, row_sum = scratch
        def staged():
            h1 = row_max_kernel[grid](x, row_max, cols, block, num_warps=warp)
            h2 = row_sumexp_kernel[grid](x, row_max, row_sum, cols, block,
                                        num_warps=warp)
            h3 = row_normalize_kernel[grid](x, row_max, row_sum, y, cols, block,
                                           num_warps=warp)
            return (h1, h2, h3)
        return staged
    if variant == 'fused':
        return lambda: softmax_fused_kernel[grid](x, y, cols, block, num_warps=warp)
    return lambda: softmax_online_kernel[grid](
        x, y, cols, chunk, triton.cdiv(cols, chunk), num_warps=warp)


def logical_bytes(lab, variant, rows, cols):
    input_bytes = rows * cols * 4
    output_bytes = rows * (4 if lab in ('R01', 'R02') else cols * 4)
    if lab == 'R03' and variant == 'staged':
        # Three X reads, one Y write, and row-max/row-sum scratch IO.
        return 3 * input_bytes + output_bytes + 6 * rows * 4
    if lab == 'R04' and variant == 'online':
        return 2 * input_bytes + output_bytes
    return input_bytes + output_bytes


def run_lab(lab, a, torch_mod):
    default_rows, default_cols = DEFAULT_SHAPES[lab]
    rows = a.rows or default_rows
    cols = a.cols or default_cols
    block = triton.next_power_of_2(cols)
    x = make_case(torch_mod, rows, cols, a.seed)
    # Compute independent reference on CPU FP64.  Never rely on BI-V150
    # device-side .double() conversion / FP64 reductions for correctness.
    expect = cpu_reference(torch_mod, x, lab)
    if lab in ('R03', 'R04'):
        atol, rtol = 3e-5, 3e-5
    else:
        atol, rtol = 5e-4, 1e-4
    scratch = (torch_mod.empty((rows,), device='cuda', dtype=torch_mod.float32),
               torch_mod.empty((rows,), device='cuda', dtype=torch_mod.float32))
    cases = []
    for warp in a.warp_options:
        options = ([(None, '')] if lab in ('R01', 'R03') else
                   [(v, f'_chunk{v}') for v in (a.chunk_options if lab == 'R02'
                                                else a.stream_options) if v <= block])
        if not options:
            raise ValueError(f'{lab}: all requested chunk sizes exceed padded cols={block}')
        for chunk, suffix in options:
            cases.append((warp, chunk, suffix))

    records = []
    success = 0
    for warp, chunk, suffix in cases:
        launches = {}
        local = {}
        active = [v for v in VARIANTS[lab] if not a.variant or v == a.variant]
        for variant in active:
            row = dict(lab=lab, variant=variant, rows=rows, cols=cols,
                       num_warps=warp, chunk=chunk or '',
                       padded_cols=block, single_run=a.single_run,
                       serial_rows_per_cta=(SERIAL_ROWS_PER_CTA if lab == 'R01'
                                            and variant == 'serial' else ''),
                       requested_bytes=logical_bytes(lab, variant, rows, cols))
            local[variant] = row
            y = torch_mod.empty((rows,) if lab in ('R01', 'R02') else (rows, cols),
                                device='cuda', dtype=torch_mod.float32)
            # A skipped/out-of-bounds tl.store must never appear as a zero-filled
            # successful output merely because the caching allocator reused memory.
            y.fill_(float('nan'))
            launch = launch_factory(lab, variant, x, y, rows, cols, block,
                                    warp, chunk, scratch)
            try:
                compiled = launch()
                row.update(check_result(torch_mod, y, expect, atol=atol, rtol=rtol))
                row.update(metadata(compiled))
                dump_asm(compiled, a.dump_asm,
                         f'{lab}_{variant}_w{warp}{suffix}')
                row['status'] = 'PASS'
                launches[variant] = launch
                success += 1
                print(f'PASS {lab} {variant}: rows={rows} cols={cols} '
                      f'warps={warp} chunk={chunk}', flush=True)
            except Exception as exc:
                row['status'] = 'ERROR'
                row['error'] = f'{type(exc).__name__}: {exc}'
                print(f'FAIL {lab} {variant} warp={warp} chunk={chunk}: '
                      f'{row["error"]}', flush=True)
                if a.strict:
                    records.extend(local.values())
                    write_csv(a.out, records)
                    raise
        if not a.single_run and launches:
            try:
                measured = benchmark_alternating(
                    launches, a.warmup_ms, a.rep_ms, a.trials)
                for variant, times in measured.items():
                    local[variant].update(times)
                    local[variant]['logical_traffic_gbps'] = round(
                        logical_bytes(lab, variant, rows, cols) /
                        (times['latency_us'] * 1e-6) / 1e9, 3)
                if len(measured) == 2:
                    base, candidate = VARIANTS[lab]
                    speedup = (measured[base]['latency_us'] /
                               measured[candidate]['latency_us'])
                    for variant in measured:
                        local[variant]['candidate_speedup'] = round(speedup, 5)
                    print(f'  {lab} candidate/base speedup: {speedup:.3f}x', flush=True)
                for variant in measured:
                    print(f'  {variant}: {measured[variant]["latency_us"]:.3f} us',
                          flush=True)
            except Exception as exc:
                for variant in launches:
                    local[variant]['status'] = 'ERROR'
                    local[variant]['error'] = f'Timing: {type(exc).__name__}: {exc}'
                if a.strict:
                    records.extend(local.values())
                    write_csv(a.out, records)
                    raise
        records.extend(local.values())
    del x, scratch
    return records, success


def main():
    a = parse_args()
    torch_mod, installed_triton = load_runtime()
    torch_mod.cuda.set_device(0)
    print(f'GPU={torch_mod.cuda.get_device_name(0)} torch={torch_mod.__version__} '
          f'triton={installed_triton.__version__}', flush=True)
    write_environment(a.out, torch_mod, installed_triton, a)
    labs = list(DEFAULT_SHAPES) if a.lab == 'all' else [a.lab]
    all_rows = []
    for lab in labs:
        try:
            rows, _ = run_lab(lab, a, torch_mod)
            all_rows.extend(rows)
        except Exception as exc:
            if a.strict:
                raise
            print(f'FAILED LAB {lab}: {type(exc).__name__}: {exc}', flush=True)
            all_rows.append(dict(lab=lab, status='ERROR', error=str(exc)))
    write_csv(a.out, all_rows)
    errors = [r for r in all_rows if r.get('status') != 'PASS']
    if errors:
        raise SystemExit(f'{len(errors)} failed configurations; inspect {a.out}')
    print('Note: R02 chunk layout does not guarantee warp-local execution. '
          'Inspect IR and hardware counters before attributing a speedup.', flush=True)


if __name__ == '__main__':
    main()
