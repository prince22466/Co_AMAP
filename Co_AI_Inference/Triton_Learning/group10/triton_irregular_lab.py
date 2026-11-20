#!/usr/bin/env python3
"""Group10: irregular-memory GPU optimization A/B labs (BI-V150 / Triton).

I01: direct gather vs index-sorted gather with scatter-back.
I02: unordered unique-destination scatter vs destination-sorted scatter.
I03: masked full grid vs mask-free full blocks plus a masked tail kernel.
I04: padded row storage vs packed ragged storage for row-wise sum.

Preprocessing and representation conversion are OUTSIDE GPU kernel timings.
All GPU outputs are independently validated against CPU references.
No changes to vLLM/FlagGems or competition benchmark scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from irregular_reference import (
    gather_inputs, scatter_inputs, tail_reference, ragged_inputs,
    parse_int_csv, parse_floats,
)
from irregular_lab_common import (
    require_gpu, check_gpu_result, compiled_metadata, dump_compiled,
    paired_bench, write_csv, write_environment,
)

try:
    import triton
    import triton.language as tl
except ImportError:
    raise SystemExit('Run this script in the BI-V150 PyTorch + Triton environment')


@triton.jit
def gather_direct_kernel(SRC, IDX, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < N
    idx = tl.load(IDX + pos, mask=valid, other=0)
    value = tl.load(SRC + idx, mask=valid, other=0)
    tl.store(OUT + pos, value, mask=valid)


@triton.jit
def gather_sorted_kernel(SRC, SORTED_IDX, ORIGINAL_POS, OUT,
                         N: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < N
    idx = tl.load(SORTED_IDX + pos, mask=valid, other=0)
    original = tl.load(ORIGINAL_POS + pos, mask=valid, other=0)
    value = tl.load(SRC + idx, mask=valid, other=0)
    # Rearrange results to the original output ordering (extra index traffic).
    tl.store(OUT + original, value, mask=valid)


@triton.jit
def scatter_direct_kernel(VALUES, DEST_IDX, OUT,
                          N: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < N
    destination = tl.load(DEST_IDX + pos, mask=valid, other=0)
    value = tl.load(VALUES + pos, mask=valid, other=0)
    # Every destination is unique, so the output has no write races.
    tl.store(OUT + destination, value, mask=valid)


@triton.jit
def scatter_sorted_kernel(VALUES, SORTED_DEST, SOURCE_POS, OUT,
                          N: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < N
    destination = tl.load(SORTED_DEST + pos, mask=valid, other=0)
    source_pos = tl.load(SOURCE_POS + pos, mask=valid, other=0)
    value = tl.load(VALUES + source_pos, mask=valid, other=0)
    tl.store(OUT + destination, value, mask=valid)


@triton.jit
def affine_masked_kernel(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < N
    x = tl.load(X + pos, mask=valid, other=0.0)
    y = tl.fma(x, 1.125, 0.375)
    tl.store(OUT + pos, y, mask=valid)


@triton.jit
def affine_full_kernel(X, OUT, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + pos)
    tl.store(OUT + pos, tl.fma(x, 1.125, 0.375))


@triton.jit
def affine_tail_kernel(X, OUT, START: tl.constexpr,
                       N: tl.constexpr, BLOCK: tl.constexpr):
    pos = START + tl.arange(0, BLOCK)
    valid = pos < N
    x = tl.load(X + pos, mask=valid, other=0.0)
    tl.store(OUT + pos, tl.fma(x, 1.125, 0.375), mask=valid)


@triton.jit
def ragged_padded_kernel(PADDED, LENGTHS, OUT,
                         ROWS: tl.constexpr, MAX_LEN: tl.constexpr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    length = tl.load(LENGTHS + row)
    cols = tl.arange(0, BLOCK)
    data = tl.load(PADDED + row * MAX_LEN + cols,
                   mask=cols < length, other=0.0)
    value = tl.sum(data, axis=0)
    tl.store(OUT + row, value)


@triton.jit
def ragged_packed_kernel(PACKED, OFFSETS, LENGTHS, OUT,
                         ROWS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(OFFSETS + row)
    length = tl.load(LENGTHS + row)
    cols = tl.arange(0, BLOCK)
    data = tl.load(PACKED + start + cols,
                   mask=cols < length, other=0.0)
    value = tl.sum(data, axis=0)
    tl.store(OUT + row, value)


def parser_for_lab():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--lab', choices=['all', 'I01', 'I02', 'I03', 'I04'], default='all')
    p.add_argument('--variant', help='Select one variant (for profiler or isolated tests)')
    p.add_argument('--n', type=int, default=262144, help='Number of gather/scatter elements')
    p.add_argument('--source-size', type=int, default=1048576, help='Gather input array size')
    p.add_argument('--gather-patterns', default='uniform,hotset',
                   help='uniform,hotset')
    p.add_argument('--scatter-patterns', default='random,block_local',
                   help='random,block_local')
    p.add_argument('--elements', default='4093,65537,262147',
                   help='Comma-separated non-power-of-two problem sizes for I03')
    p.add_argument('--blocks', default='128,256,512', help='Power-of-two block sizes')
    p.add_argument('--warps', type=int, default=4, help='num_warps (power of two)')
    p.add_argument('--rows', type=int, default=512, help='Ragged row count')
    p.add_argument('--max-len', type=int, default=1023, help='Ragged maximum row length')
    p.add_argument('--fill-rates', default='0.25,0.75', help='Ragged target fill rates')
    p.add_argument('--warmup-ms', type=int, default=30)
    p.add_argument('--rep-ms', type=int, default=100)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--single-run', action='store_true',
                   help='Correctness/compilation only; no timed benchmarking')
    p.add_argument('--strict', action='store_true',
                   help='Raise immediately on first failure (still writes partial CSV)')
    p.add_argument('--dump-asm', metavar='DIR', default=None,
                   help='Write backend IR/assembly text when exposed')
    p.add_argument('--out', default='results/group10.csv')
    return p


def check_args(args, parser):
    if min(args.n, args.source_size, args.rows, args.max_len,
           args.warps, args.warmup_ms, args.rep_ms, args.trials) <= 0:
        parser.error('All integer sizes/timing counts must be >0')
    if args.n > 2**30 or args.source_size > 2**30 or args.rows * args.max_len > 2**29:
        parser.error('Size exceeds conservative int32 indexing / allocation safety cap')
    if args.warps & (args.warps-1):
        parser.error('--warps must be a power of two')
    if args.max_len > 8192:
        parser.error('--max-len > 8192 not supported by this teaching kernel')
    try:
        blocks = parse_int_csv(args.blocks)
        elements = parse_int_csv(args.elements)
        rates = parse_floats(args.fill_rates)
    except ValueError as exc:
        parser.error(str(exc))
    if any(v & (v-1) for v in blocks):
        parser.error('--blocks must all be powers of two')
    if max(blocks) > 2048:
        parser.error('--blocks > 2048 intentionally disallowed')
    if any(v > 2**30 for v in elements):
        parser.error('--elements too large for int32 addressing')
    gather_patterns = args.gather_patterns.split(',')
    scatter_patterns = args.scatter_patterns.split(',')
    if any(p not in ('uniform', 'hotset') for p in gather_patterns):
        parser.error('--gather-patterns must use uniform,hotset')
    if any(p not in ('random', 'block_local') for p in scatter_patterns):
        parser.error('--scatter-patterns must use random,block_local')
    if args.variant and args.lab == 'all':
        parser.error('--variant requires a specific --lab')
    return blocks, elements, rates, list(dict.fromkeys(gather_patterns)), list(dict.fromkeys(scatter_patterns))


def run_pair(torch, args, *, lab, condition, base, variants, reference, output_size):
    """Launch each variant, verify all output elements, then time valid variants."""
    chosen = {k: v for k, v in variants.items() if args.variant is None or args.variant == k}
    if not chosen:
        raise ValueError(f'{lab}: unknown variant {args.variant!r}; options: {list(variants)}')
    rows = []
    launches = {}
    for name, factory in chosen.items():
        entry = {'lab': lab, 'condition': condition, 'variant': name, **base}
        y = torch.full((output_size,), float('nan'), device='cuda', dtype=torch.float32)
        try:
            launch = factory(y)
            handles = launch()
            torch.cuda.synchronize()
            entry.update(check_gpu_result(y, reference))
            entry.update(compiled_metadata(handles))
            dump_compiled(handles, args.dump_asm, f'{lab}_{condition}_{name}'.replace('.', '_'))
            entry['status'] = 'ok'
            print(f'PASS {lab} {name}: {condition}; max_abs={entry["max_abs_error"]:.6g}', flush=True)
            launches[name] = launch
        except Exception as exc:
            entry['status'] = 'error'
            entry['error'] = f'{type(exc).__name__}: {exc}'
            print(f'FAIL {lab} {name}: {condition}: {entry["error"]}', flush=True)
            if args.strict:
                rows.append(entry)
                raise RunFailure(rows, entry['error']) from exc
        rows.append(entry)
    if not args.single_run and launches:
        try:
            measurements = paired_bench(
                launches, trials=args.trials, warmup_ms=args.warmup_ms, rep_ms=args.rep_ms)
            for r in rows:
                if r['status'] == 'ok':
                    r.update(measurements[r['variant']])
        except Exception as exc:
            for r in rows:
                if r['status'] == 'ok':
                    r['status'] = 'error'
                    r['error'] = f'Timing failed: {type(exc).__name__}: {exc}'
            if args.strict:
                raise RunFailure(rows, str(exc)) from exc
    # Baseline first by insertion order; speedup = baseline / measured latency.
    timed = [r for r in rows if r.get('status') == 'ok' and 'latency_us' in r]
    if timed:
        baseline_name = next(iter(variants))
        baseline = next((r for r in timed if r['variant'] == baseline_name), None)
        if baseline:
            for r in timed:
                r['speedup_vs_baseline'] = baseline['latency_us'] / r['latency_us']
                print(f'TIME {lab} {condition} {r["variant"]}: '
                      f'{r["latency_us"]:.3f} us, speedup={r["speedup_vs_baseline"]:.3f}x',
                      flush=True)
    return rows


class RunFailure(Exception):
    def __init__(self, rows, reason):
        super().__init__(reason)
        self.rows = rows


def main():
    parser = parser_for_lab()
    args = parser.parse_args()
    blocks, elements, rates, gather_patterns, scatter_patterns = check_args(args, parser)
    torch, runtime_triton = require_gpu()
    torch.cuda.set_device(0)
    print(f'GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} '
          f'triton={runtime_triton.__version__}', flush=True)
    write_environment(args.out, torch, runtime_triton, args)
    records = []
    labs = ('I01', 'I02', 'I03', 'I04') if args.lab == 'all' else (args.lab,)

    try:
        for lab in labs:
            if lab == 'I01':
                block = blocks[0]
                for pattern in gather_patterns:
                    data = gather_inputs(args.n, args.source_size, args.seed, pattern)
                    x = torch.from_numpy(data['source']).to('cuda')
                    idx = torch.from_numpy(data['indices']).to('cuda')
                    sorted_idx = torch.from_numpy(data['sorted_indices']).to('cuda')
                    orig = torch.from_numpy(data['original_positions']).to('cuda')
                    n = args.n
                    grid = (runtime_triton.cdiv(n, block),)
                    variants = {
                        'direct': lambda y: lambda: gather_direct_kernel[grid](
                            x, idx, y, n, block, num_warps=args.warps),
                        'sorted': lambda y: lambda: gather_sorted_kernel[grid](
                            x, sorted_idx, orig, y, n, block, num_warps=args.warps),
                    }
                    records.extend(run_pair(
                        torch, args, lab=lab, condition=pattern,
                        base={'n': n, 'source_size': args.source_size, 'block': block,
                              'num_warps': args.warps,
                              'extra_preprocessing': 'sorted_indices+original_positions (excluded)',
                              'note': 'sorted variant trades source read locality for scattered writes and extra index reads'},
                        variants=variants, reference=data['reference'], output_size=n))

            elif lab == 'I02':
                block = blocks[0]
                for pattern in scatter_patterns:
                    data = scatter_inputs(args.n, args.seed, pattern)
                    values = torch.from_numpy(data['values']).to('cuda')
                    destinations = torch.from_numpy(data['destinations']).to('cuda')
                    sorted_dest = torch.from_numpy(data['sorted_destinations']).to('cuda')
                    source_pos = torch.from_numpy(data['source_positions']).to('cuda')
                    n = args.n
                    grid = (runtime_triton.cdiv(n, block),)
                    variants = {
                        'direct': lambda y: lambda: scatter_direct_kernel[grid](
                            values, destinations, y, n, block, num_warps=args.warps),
                        'sorted': lambda y: lambda: scatter_sorted_kernel[grid](
                            values, sorted_dest, source_pos, y, n, block, num_warps=args.warps),
                    }
                    records.extend(run_pair(
                        torch, args, lab=lab, condition=pattern,
                        base={'n': n, 'block': block, 'num_warps': args.warps,
                              'unique_destinations': True,
                              'extra_preprocessing': 'destination argsort (excluded)',
                              'note': 'sorted variant changes source locality and adds index traffic'},
                        variants=variants, reference=data['reference'], output_size=n))

            elif lab == 'I03':
                for n in elements:
                    data = tail_reference(n, args.seed)
                    x = torch.from_numpy(data['x']).to('cuda')
                    for block in blocks:
                        full = n // block
                        start = full * block
                        remainder = n - start
                        def make_tail_launch(y, *, n=n, block=block, full=full, start=start,
                                             remainder=remainder):
                            def go():
                                handles = []
                                if full:
                                    handles.append(affine_full_kernel[(full,)](
                                        x, y, block, num_warps=args.warps))
                                if remainder:
                                    handles.append(affine_tail_kernel[(1,)](
                                        x, y, start, n, block, num_warps=args.warps))
                                return tuple(handles)
                            return go
                        variants = {
                            'masked': lambda y, n=n, block=block: lambda: affine_masked_kernel[
                                (runtime_triton.cdiv(n, block),)](
                                x, y, n, block, num_warps=args.warps),
                            'tail_split': make_tail_launch,
                        }
                        records.extend(run_pair(
                            torch, args, lab=lab, condition=f'n{n}_b{block}',
                            base={'n': n, 'block': block, 'num_warps': args.warps,
                                  'full_tiles': full, 'tail_items': remainder,
                                  'kernel_launches_masked': 1,
                                  'kernel_launches_tail_split': int(full > 0) + int(remainder > 0),
                                  'note': 'tail_split may require two launches; neither variant pads allocations'},
                            variants=variants, reference=data['reference'], output_size=n))

            elif lab == 'I04':
                block = runtime_triton.next_power_of_2(args.max_len)
                for rate in rates:
                    data = ragged_inputs(args.rows, args.max_len, rate, args.seed)
                    packed = torch.from_numpy(data['packed']).to('cuda')
                    padded = torch.from_numpy(data['padded']).to('cuda')
                    lengths = torch.from_numpy(data['lengths']).to('cuda')
                    offsets = torch.from_numpy(data['offsets']).to('cuda')
                    rows = args.rows
                    max_len = args.max_len
                    variants = {
                        'padded': lambda y: lambda: ragged_padded_kernel[(rows,)](
                            padded, lengths, y, rows, max_len, block, num_warps=args.warps),
                        'packed': lambda y: lambda: ragged_packed_kernel[(rows,)](
                            packed, offsets, lengths, y, rows, block, num_warps=args.warps),
                    }
                    records.extend(run_pair(
                        torch, args, lab=lab, condition=f'fill{rate:g}',
                        base={'rows': rows, 'max_len': max_len, 'block': block,
                              'num_warps': args.warps, 'target_fill_rate': rate,
                              'actual_fill_rate': data['actual_fill'],
                              'packed_bytes': data['packed'].nbytes,
                              'padded_bytes': data['padded'].nbytes,
                              'extra_preprocessing': 'packing/padding (excluded)',
                              'note': 'both variants mask unused elements; layouts and footprint differ'},
                        variants=variants, reference=data['reference'], output_size=rows))
    except RunFailure as exc:
        records.extend(exc.rows)
        write_csv(args.out, records)
        raise SystemExit(f'Group10 failed: {exc}') from exc
    except Exception:
        if records:
            write_csv(args.out, records)
        raise

    write_csv(args.out, records)
    bad = [r for r in records if r.get('status') != 'ok']
    if bad:
        raise SystemExit(f'{len(bad)} Group10 variant(s) failed; see CSV errors')
    print('Group10 complete: all tested variants passed CPU-reference correctness checks.', flush=True)


if __name__ == '__main__':
    main()
