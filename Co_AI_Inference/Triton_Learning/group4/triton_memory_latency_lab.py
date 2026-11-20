#!/usr/bin/env python3
"""Group4: controlled Triton memory latency and cache experiments.

M01: working-set sweep, identical stream kernel and logical load count
M02: dependent pointer chain (1) versus four independent pointer chains
M03: memory-level-parallelism (MLP) sweep with fixed total pointer loads
M04: memory locality (stride) x block size, fixed logical load count

This is a *standalone teaching microbenchmark*, not a competition benchmark.
Run in the same PyTorch/Triton GPU environment as the existing BI-V150 labs.
Outputs include correctness status, event-based timing, compiler metadata, and
logical read rates (NOT measured device/HBM bandwidth).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from memory_lab_common import (
    benchmark_round_robin, chase_reference, dump_asm, gpu_info, metadata,
    positive_ints, powers_of_two, sample_positions, stream_reference,
    verify_selected, working_sets_mib, write_csv,
)

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as exc:
    raise SystemExit('PyTorch and Triton are required. Use your BI-V150 GPU environment.') from exc


@triton.jit
def stream_read_kernel(X, Y, N: tl.constexpr, WS_ENTRIES: tl.constexpr,
                       REPS: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    """Repeated global loads; address changes every iteration.

    Small WS reuses data heavily; large WS scans more unique data.
    Contiguous or strided lane addresses are selected with STRIDE.
    Volatile prevents the compiler from deleting/combining repeated loads; it
    does not mandate bypassing the GPU's hardware cache hierarchy.
    """
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    good = p < N
    accum = tl.full((BLOCK,), 0, dtype=tl.float32)
    for i in range(REPS):
        addr = (p * STRIDE + i * N) & (WS_ENTRIES - 1)
        values = tl.load(X + addr, mask=good, other=0.0, volatile=True)
        accum += values
    tl.store(Y + p, accum, mask=good)


@triton.jit
def pointer_chase_kernel(NEXT, OUT, N: tl.constexpr, POOL_ENTRIES: tl.constexpr,
                         CHAINS: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    """[CHAINS, BLOCK] independently evolving pointer chains.

    Every load address depends on the preceding load result in the same chain.
    Increasing CHAINS exposes more outstanding requests but also changes the
    register footprint, reduction work and compiler scheduling. Compare with
    hardware counters before attributing a speedup to pure memory latency.
    """
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = tl.arange(0, CHAINS)
    valid = p < N
    ix = (p[None, :] + rows[:, None] * 65537) & (POOL_ENTRIES - 1)
    for _ in range(STEPS):
        ix = tl.load(NEXT + ix, mask=valid[None, :], other=0, volatile=True)
    val = tl.sum(ix, axis=0)
    tl.store(OUT + p, val, mask=valid)


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--lab', choices=('all', 'M01', 'M02', 'M03', 'M04'), default='all')
    p.add_argument('--out', default='results/memory_latency.csv')
    p.add_argument('--dump-asm', default=None, metavar='DIR', help='Write available Triton IR/ISA text')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--warmup-ms', type=int, default=30)
    p.add_argument('--rep-ms', type=int, default=90)
    p.add_argument('--strict', action='store_true', help='Stop at first failure')
    p.add_argument('--native', action='store_true',
                   help='Launch just one selected config, without timing; for ixkn-cli profiling')
    p.add_argument('--native-index', type=int, default=0, help='Zero-based case index within --lab')
    p.add_argument('--cache-n', type=int, default=262144, help='Output elements for M01 and M04')
    p.add_argument('--cache-reps', type=int, default=64, help='Loads per output for M01 and M04')
    p.add_argument('--cache-sizes-mib', default='0.25,2,16,64',
                   help='Working-set sizes for M01, powers of two in FP32 elements')
    p.add_argument('--cache-size-mib', default='64', help='M04 working-set size in MiB')
    p.add_argument('--locality-strides', default='1,17,257', help='M04 lane address strides')
    p.add_argument('--block', type=int, default=256, help='M01/M02/M03 Triton block elements')
    p.add_argument('--blocks', default='128,256,512', help='M04 Triton block sweep')
    p.add_argument('--warps', type=int, default=4, help='Warps per Triton program')
    p.add_argument('--chase-n', type=int, default=65536, help='Outputs for M02/M03 (many CTAs)')
    p.add_argument('--pool-entries', type=int, default=1048576,
                   help='Random pointer-table entries (power of two), int32')
    p.add_argument('--total-loads', type=int, default=128,
                   help='Total pointer loads per output, constant across chain counts')
    p.add_argument('--chains', default='1,2,4,8,16', help='M03 independent-chain counts')
    return p


def validate_args(a, p):
    for key in ('trials', 'warmup_ms', 'rep_ms', 'cache_n', 'cache_reps',
                'chase_n', 'pool_entries', 'total_loads', 'block', 'warps'):
        if getattr(a, key) <= 0:
            p.error(f'--{key.replace("_", "-")} must be > 0')
    for key in ('pool_entries', 'block', 'warps'):
        v = getattr(a, key)
        if v & (v - 1):
            p.error(f'--{key.replace("_", "-")} must be a power of two')
    if a.native_index < 0:
        p.error('--native-index must be >= 0')
    if a.native and a.lab == 'all':
        p.error('--native requires a single --lab')
    try:
        a.cache_ws = working_sets_mib(a.cache_sizes_mib)
        one = working_sets_mib(a.cache_size_mib)
        if len(one) != 1:
            p.error('--cache-size-mib requires exactly one size')
        a.local_ws = one[0]
        a.strides = positive_ints(a.locality_strides, 'locality strides')
        a.block_sweep = powers_of_two(a.blocks, 'blocks')
        a.chain_sweep = powers_of_two(a.chains, 'chains')
    except ValueError as exc:
        p.error(str(exc))
    for c in [1, 4, *a.chain_sweep]:
        if a.total_loads % c:
            p.error(f'--total-loads ({a.total_loads}) must be divisible by chains={c}')
    if any(c > 32 for c in a.chain_sweep):
        p.error('This lab restricts --chains to <= 32 to limit resource usage')
    if a.cache_n * (max(a.strides) + a.cache_reps + 1) >= 2**31:
        p.error('Stream index arithmetic may overflow int32; reduce n/reps/strides')
    if a.pool_entries > 2**24:
        p.error('Limit --pool-entries to 2**24 for reasonable host setup and references')
    if a.chase_n > 2**27:
        p.error('--chase-n too large for a learning microbenchmark')


def define_cases(a):
    """Create CPU-side descriptions without allocating GPU memory."""
    cases = []
    if a.lab in ('all', 'M01'):
        for ws_mib, ws in a.cache_ws:
            cases.append({
                'lab': 'M01', 'name': f'M01_ws{ws}_b{a.block}',
                'variant': f'working_set_{ws_mib:g}_MiB',
                'kernel': 'stream_read_kernel', 'n': a.cache_n, 'block': a.block,
                'warps': a.warps, 'ws_mib': ws_mib, 'ws_entries': ws,
                'stride': 1, 'reps': a.cache_reps,
                'touched_mib_upper_bound': min(ws_mib, a.cache_n * a.cache_reps * 4 / 1048576),
                'logical_read_bytes': a.cache_n * a.cache_reps * 4,
            })
    if a.lab in ('all', 'M02'):
        for c in (1, 4):
            cases.append({
                'lab': 'M02', 'name': f'M02_c{c}_b{a.block}',
                'variant': 'dependent_one_chain' if c == 1 else 'four_independent_chains',
                'kernel': 'pointer_chase_kernel', 'n': a.chase_n, 'block': a.block,
                'warps': a.warps, 'chains': c, 'steps_per_chain': a.total_loads // c,
                'total_loads': a.total_loads, 'pool_entries': a.pool_entries,
                'logical_read_bytes': a.chase_n * a.total_loads * 4,
            })
    if a.lab in ('all', 'M03'):
        for c in a.chain_sweep:
            cases.append({
                'lab': 'M03', 'name': f'M03_c{c}_b{a.block}',
                'variant': f'{c}_chains',
                'kernel': 'pointer_chase_kernel', 'n': a.chase_n, 'block': a.block,
                'warps': a.warps, 'chains': c, 'steps_per_chain': a.total_loads // c,
                'total_loads': a.total_loads, 'pool_entries': a.pool_entries,
                'logical_read_bytes': a.chase_n * a.total_loads * 4,
            })
    if a.lab in ('all', 'M04'):
        ws_mib, ws = a.local_ws
        for block in a.block_sweep:
            for stride in a.strides:
                cases.append({
                    'lab': 'M04', 'name': f'M04_s{stride}_b{block}',
                    'variant': 'contiguous' if stride == 1 else f'lane_stride_{stride}',
                    'kernel': 'stream_read_kernel', 'n': a.cache_n, 'block': block,
                    'warps': a.warps, 'ws_mib': ws_mib, 'ws_entries': ws,
                    'stride': stride, 'reps': a.cache_reps,
                    'logical_read_bytes': a.cache_n * a.cache_reps * 4,
                })
    return cases


def main():
    parser = make_parser()
    a = parser.parse_args()
    validate_args(a, parser)
    if not torch.cuda.is_available():
        parser.error('No supported GPU runtime visible via torch.cuda')
    torch.cuda.set_device(0)
    cases = define_cases(a)
    if a.native:
        if a.native_index >= len(cases):
            parser.error(f'--native-index out of range: 0..{len(cases)-1}')
        cases = [cases[a.native_index]]
    print(f'Device={torch.cuda.get_device_name(0)} torch={torch.__version__} '
          f'triton={triton.__version__} cases={len(cases)}', flush=True)

    # Allocate one deterministic stream input and one pointer permutation.
    # M01 and M04 use the SAME GPU stream buffer, so only the address pattern
    # and configured resident working-set size change; buffer is not reallocated.
    stream_cases = [c for c in cases if c['kernel'] == 'stream_read_kernel']
    chase_cases = [c for c in cases if c['kernel'] == 'pointer_chase_kernel']
    pool_cpu = pool_gpu = stream_y = None
    table_cpu = table_gpu = chase_y = None
    if stream_cases:
        max_ws = max(c['ws_entries'] for c in stream_cases)
        rng = np.random.default_rng(a.seed)
        pool_cpu = rng.uniform(0.1, 0.9, size=max_ws).astype(np.float32)
        pool_gpu = torch.from_numpy(pool_cpu).to('cuda')
        stream_y = torch.empty(a.cache_n, device='cuda', dtype=torch.float32)
    if chase_cases:
        rng = np.random.default_rng(a.seed + 1)
        table_cpu = rng.permutation(a.pool_entries).astype(np.int32)
        table_gpu = torch.from_numpy(table_cpu).to('cuda')
        chase_y = torch.empty(a.chase_n, device='cuda', dtype=torch.int32)

    out_path = Path(a.out)
    env_path = out_path.with_suffix('.env.json')
    gpu_info(torch, triton, {k: v for k, v in vars(a).items() if k not in
                            ('cache_ws', 'local_ws', 'strides', 'block_sweep', 'chain_sweep')}, str(env_path))

    for case in cases:
        case['status'] = 'ready'
        try:
            n, b, w = case['n'], case['block'], case['warps']
            if case['kernel'] == 'stream_read_kernel':
                ws, stride, reps = case['ws_entries'], case['stride'], case['reps']
                def launch(x=pool_gpu, y=stream_y, n=n, ws=ws, stride=stride, reps=reps, b=b, w=w):
                    return stream_read_kernel[(triton.cdiv(n, b),)](
                        x, y, n, ws, reps, stride, b, num_warps=w,
                    )
                positions = sample_positions(n)
                expected = stream_reference(pool_cpu, n, reps, stride, positions, ws)
                output = stream_y
                exact = False
            else:
                chains, steps = case['chains'], case['steps_per_chain']
                def launch(x=table_gpu, y=chase_y, n=n, chains=chains, steps=steps, b=b, w=w):
                    return pointer_chase_kernel[(triton.cdiv(n, b),)](
                        x, y, n, a.pool_entries, chains, steps, b, num_warps=w,
                    )
                positions = sample_positions(n)
                expected = chase_reference(table_cpu, chains, case['total_loads'], positions)
                output = chase_y
                exact = True
            case['launch'] = launch
            kernel = launch()
            torch.cuda.synchronize()
            verify_selected(output, positions, expected, exact=exact)
            case.update(metadata(kernel))
            dump_asm(kernel, a.dump_asm, case['name'])
            print(f'PASS {case["name"]} correctness ({len(positions)} positions); '
                  f'regs={case["registers_per_thread"]} spills={case["spills"]}', flush=True)
        except Exception as exc:
            case.update(status='error', error=f'{type(exc).__name__}: {exc}')
            print(f'FAIL {case["name"]}: {case["error"]}', file=sys.stderr, flush=True)
            if a.strict:
                write_csv(a.out, cases)
                raise

    if a.native:
        c = cases[0]
        if c['status'] != 'ready':
            raise SystemExit('Native profile kernel failed correctness')
        print(f'PROFILE_OK {c["lab"]} {c["name"]}', flush=True)
        return

    try:
        benchmark_round_robin(cases, warmup_ms=a.warmup_ms, rep_ms=a.rep_ms, trials=a.trials)
    except Exception as exc:
        print(f'BENCHMARK ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        for c in cases:
            if c['status'] == 'ready':
                c.update(status='error', error=f'Benchmark aborted: {type(exc).__name__}: {exc}')
    for c in cases:
        if c.get('status') == 'ok':
            print(f'{c["lab"]} {c["variant"]:>26s} block={c["block"]:3d} '
                  f'{c["latency_us"]:10.3f} us  logical_read={c["logical_read_gbps"]:8.2f} GB/s', flush=True)
    write_csv(a.out, cases)
    pairs = [c for c in cases if c['lab'] == 'M02' and c.get('status') == 'ok']
    if len(pairs) == 2:
        seq = next(x for x in pairs if x['chains'] == 1)
        par = next(x for x in pairs if x['chains'] == 4)
        print(f'M02 observed latency ratio (1 chain / 4 chains): '
              f'{seq["latency_us"] / par["latency_us"]:.3f}x', flush=True)
    if any(c['status'] != 'ok' for c in cases):
        raise SystemExit('At least one configuration failed; inspect CSV status/error fields')
    print('NOTE: GB/s is LOGICAL load volume / time, not measured HBM bandwidth.', flush=True)
    print('NOTE: cache residency, actual MLP and warp stalls need BI-V150 hardware counters.', flush=True)


if __name__ == '__main__':
    main()
