#!/usr/bin/env python3
"""Group7: Advanced GEMM A/B microbenchmarks for Triton on BI-V150.

G01: GEMM tile sizes. G02: row-major vs grouped tile ordering.
G03: regular GEMM vs split-K plus full reduction cost.
G04: ordinary grid vs a bounded, persistent work grid.

These are standalone educational kernels; not vLLM or competition benchmarks.
Only bf16/fp16 inputs and fp32 output are used; no quantization is enabled.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from gemm_lab_common import (
    bench_alternating, check_gemm, compiled_info, dump_kernel, parse_tiles,
    positive_list, runtime, save_csv, save_environment,
)

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise SystemExit('Triton is required on the GPU device') from exc


@triton.jit
def gemm_grid_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     GROUP_M: tl.constexpr):
    """Same GEMM for all tile and grouped-order comparisons."""
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    group_width = GROUP_M * tiles_n
    group_id = pid // group_width
    first_m = group_id * GROUP_M
    actual_m = tl.minimum(tiles_m - first_m, GROUP_M)
    local_pid = pid % group_width
    pid_m = first_m + (local_pid % actual_m)
    pid_n = local_pid // actual_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for ik in range(tl.cdiv(K, BK)):
        kk = ik * BK + rk
        a = tl.load(A + rm[:, None] * K + kk[None, :],
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(B + kk[:, None] * N + rn[None, :],
                    mask=(kk[:, None] < K) & (rn[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    tl.store(C + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def gemm_split_k_kernel(A, B, PART, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        SPLITS: tl.constexpr):
    """Each split writes a distinct FP32 partial tile; no atomics/zero-fill needed."""
    tile = tl.program_id(0)
    sid = tl.program_id(1)
    tiles_n = tl.cdiv(N, BN)
    pid_m = tile // tiles_n
    pid_n = tile % tiles_n
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    iters_per_split = tl.cdiv(K, SPLITS * BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(iters_per_split):
        kk = sid * (iters_per_split * BK) + i * BK + rk
        a = tl.load(A + rm[:, None] * K + kk[None, :],
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(B + kk[:, None] * N + rn[None, :],
                    mask=(kk[:, None] < K) & (rn[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    tl.store(PART + sid * M * N + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def split_k_reduce_kernel(PART, C, M: tl.constexpr, N: tl.constexpr,
                          SPLITS: tl.constexpr, SPAD: tl.constexpr,
                          REDUCE_BLOCK: tl.constexpr):
    idx = tl.program_id(0) * REDUCE_BLOCK + tl.arange(0, REDUCE_BLOCK)
    ss = tl.arange(0, SPAD)
    parts = tl.load(PART + ss[:, None] * M * N + idx[None, :],
                    mask=(ss[:, None] < SPLITS) & (idx[None, :] < M * N), other=0)
    result = tl.sum(parts, axis=0)
    tl.store(C + idx, result, mask=idx < M * N)


@triton.jit
def gemm_persistent_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           WORKERS: tl.constexpr):
    """Fixed number of worker CTAs; each CTA processes multiple output tiles."""
    tiles_n = tl.cdiv(N, BN)
    num_tiles = tl.cdiv(M, BM) * tiles_n
    for tile in range(tl.program_id(0), num_tiles, WORKERS):
        pid_m = tile // tiles_n
        pid_n = tile % tiles_n
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for ik in range(tl.cdiv(K, BK)):
            kk = ik * BK + rk
            a = tl.load(A + rm[:, None] * K + kk[None, :],
                        mask=(rm[:, None] < M) & (kk[None, :] < K), other=0)
            b = tl.load(B + kk[:, None] * N + rn[None, :],
                        mask=(kk[:, None] < K) & (rn[None, :] < N), other=0)
            acc = tl.dot(a, b, acc)
        tl.store(C + rm[:, None] * N + rn[None, :], acc,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lab', choices=['all', 'G01', 'G02', 'G03', 'G04'], default='all')
    p.add_argument('--variant', help='Run only a named variant (useful for profiler captures)')
    p.add_argument('--m', type=int, default=512)
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--k', type=int, default=1024)
    p.add_argument('--dtype', choices=['bf16', 'fp16'], default='bf16')
    p.add_argument('--tiles', default='32x64x32,64x64x32,64x128x32,128x64x32',
                   help='G01 tile sweep BMxBNxBK (comma separated)')
    p.add_argument('--bm', type=int, default=64)
    p.add_argument('--bn', type=int, default=64)
    p.add_argument('--bk', type=int, default=32)
    p.add_argument('--groups', default='1,8,16', help='G02 GROUP_M sweep; 1 is row-major')
    p.add_argument('--splits', default='2,4,8', help='G03 split-K factors (baseline 1 added)')
    p.add_argument('--workers', default='32,64', help='G04 persistent CTA count')
    p.add_argument('--warps', type=int, default=4)
    p.add_argument('--stages', type=int, default=2)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--warmup-ms', type=int, default=30)
    p.add_argument('--rep-ms', type=int, default=100)
    p.add_argument('--single-run', action='store_true', help='Correctness + metadata; skip timing')
    p.add_argument('--strict', action='store_true', help='Abort on first failed variant')
    p.add_argument('--dump-asm', help='Write backend textual IR/ISA if supplied')
    p.add_argument('--out', default='results/group7.csv')
    a = p.parse_args()
    if min(a.m, a.n, a.k, a.bm, a.bn, a.bk, a.warps, a.stages,
           a.trials, a.warmup_ms, a.rep_ms) < 1:
        p.error('All dimensions and benchmark counts must be positive')
    for s in (a.bm, a.bn, a.bk):
        if s < 16 or s & (s - 1):
            p.error('BM/BN/BK must be powers of two >=16')
    if a.warps & (a.warps - 1):
        p.error('--warps must be a power of two')
    try:
        tiles = parse_tiles(a.tiles)
        groups = positive_list(a.groups, 'group sizes')
        splits = positive_list(a.splits, 'split-K counts', powers_of_two=True)
        workers = positive_list(a.workers, 'persistent worker counts')
    except ValueError as ex:
        p.error(str(ex))
    return a, tiles, groups, splits, workers


def main():
    a, tiles, groups, splits, workers = parse_args()
    torch, installed_triton = runtime()
    torch.manual_seed(2026)
    torch.cuda.set_device(0)
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, 'bfloat16' if a.dtype == 'bf16' else 'float16')
    # torch CPU-generated inputs are deterministic; only the transfer is on GPU.
    cpu_a = torch.randn((a.m, a.k), dtype=torch.float32).to(dtype)
    cpu_b = torch.randn((a.k, a.n), dtype=torch.float32).to(dtype)
    mat_a = cpu_a.to('cuda')
    mat_b = cpu_b.to('cuda')
    reference = torch.matmul(mat_a.float(), mat_b.float())
    c = torch.empty((a.m, a.n), dtype=torch.float32, device='cuda')
    save_environment(a.out, torch, installed_triton, a)
    print(f'GPU={torch.cuda.get_device_name(0)} shape={a.m}x{a.n}x{a.k} dtype={a.dtype} '
          f'warps={a.warps} stages={a.stages}', flush=True)
    tiles_total = installed_triton.cdiv(a.m, a.bm) * installed_triton.cdiv(a.n, a.bn)
    labs = ['G01', 'G02', 'G03', 'G04'] if a.lab == 'all' else [a.lab]
    records = []

    for lab in labs:
        plans = []
        if lab == 'G01':
            for bm, bn, bk in tiles:
                name = f'tile_{bm}x{bn}x{bk}'
                grid = (installed_triton.cdiv(a.m, bm) * installed_triton.cdiv(a.n, bn),)
                def launch_grid(bm=bm, bn=bn, bk=bk, grid=grid):
                    return gemm_grid_kernel[grid](mat_a, mat_b, c, a.m, a.n, a.k,
                                                  bm, bn, bk, 1,
                                                  num_warps=a.warps, num_stages=a.stages)
                plans.append((name, launch_grid, dict(bm=bm, bn=bn, bk=bk, group_m=1,
                                                       grid_programs=grid[0], launch_count=1)))
        elif lab == 'G02':
            for group in dict.fromkeys([1] + groups):
                name = f'group_m_{group}'
                grid = (tiles_total,)
                def launch_group(group=group):
                    return gemm_grid_kernel[grid](mat_a, mat_b, c, a.m, a.n, a.k,
                                                  a.bm, a.bn, a.bk, group,
                                                  num_warps=a.warps, num_stages=a.stages)
                plans.append((name, launch_group, dict(bm=a.bm, bn=a.bn, bk=a.bk,
                                                        group_m=group, grid_programs=grid[0], launch_count=1)))
        elif lab == 'G03':
            def baseline():
                return gemm_grid_kernel[(tiles_total,)](mat_a, mat_b, c,
                    a.m, a.n, a.k, a.bm, a.bn, a.bk, 1,
                    num_warps=a.warps, num_stages=a.stages)
            plans.append(('split_1', baseline, dict(bm=a.bm, bn=a.bn, bk=a.bk,
                split_k=1, grid_programs=tiles_total, launch_count=1,
                workspace_bytes=0)))
            for split in splits:
                if split == 1:
                    continue
                workspace = torch.empty((split, a.m, a.n), dtype=torch.float32, device='cuda')
                def launch_split(split=split, workspace=workspace):
                    h0 = gemm_split_k_kernel[(tiles_total, split)](
                        mat_a, mat_b, workspace, a.m, a.n, a.k,
                        a.bm, a.bn, a.bk, split,
                        num_warps=a.warps, num_stages=a.stages)
                    h1 = split_k_reduce_kernel[(installed_triton.cdiv(a.m * a.n, 128),)](
                        workspace, c, a.m, a.n, split,
                        installed_triton.next_power_of_2(split), 128,
                        num_warps=a.warps)
                    return (h0, h1)
                plans.append((f'split_{split}', launch_split, dict(
                    bm=a.bm, bn=a.bn, bk=a.bk, split_k=split,
                    grid_programs=tiles_total * split, launch_count=2,
                    workspace_bytes=workspace.numel() * workspace.element_size())))
        else:  # G04
            def baseline():
                return gemm_grid_kernel[(tiles_total,)](mat_a, mat_b, c,
                    a.m, a.n, a.k, a.bm, a.bn, a.bk, 1,
                    num_warps=a.warps, num_stages=a.stages)
            plans.append(('ordinary_grid', baseline, dict(bm=a.bm, bn=a.bn, bk=a.bk,
                grid_programs=tiles_total, total_tiles=tiles_total, launch_count=1)))
            for worker_count in workers:
                active = min(worker_count, tiles_total)
                def persistent(workers=active):
                    return gemm_persistent_kernel[(workers,)](mat_a, mat_b, c,
                        a.m, a.n, a.k, a.bm, a.bn, a.bk, workers,
                        num_warps=a.warps, num_stages=a.stages)
                plans.append((f'persistent_{worker_count}', persistent,
                    dict(bm=a.bm, bn=a.bn, bk=a.bk, requested_workers=worker_count,
                         grid_programs=active, total_tiles=tiles_total, launch_count=1)))

        if a.variant:
            plans = [plan for plan in plans if plan[0] == a.variant]
            if not plans:
                raise SystemExit(f'{lab}: no configuration named {a.variant!r}')
        launches = {}
        staged = {}
        for name, launch, properties in plans:
            rec = dict(lab=lab, variant=name, m=a.m, n=a.n, k=a.k, dtype=a.dtype,
                       warps=a.warps, stages=a.stages, logical_gflop=2 * a.m * a.n * a.k / 1e9,
                       **properties)
            try:
                compiled = launch()  # compilation is excluded from timings
                kernels = compiled if isinstance(compiled, tuple) else (compiled,)
                check = check_gemm(torch, c, reference)
                rec.update(check)
                for ix, handle in enumerate(kernels):
                    dump_kernel(handle, a.dump_asm, f'{lab}_{name}_kernel{ix}')
                    for key, value in compiled_info(handle).items():
                        rec[f'kernel{ix}_{key}'] = value
                rec['status'] = 'PASS'
                staged[name] = rec
                launches[name] = launch
                print(f'PASS {lab} {name}: max_abs={check["max_abs_error"]:.5g} '
                      f'launches={len(kernels)}', flush=True)
            except Exception as exc:
                rec.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
                staged[name] = rec
                print(f'ERROR {lab} {name}: {rec["error"]}', flush=True)
                if a.strict:
                    records.extend(staged.values())
                    save_csv(a.out, records)
                    raise
        if not a.single_run and launches:
            timed = bench_alternating(launches, trials=a.trials,
                                      warmup_ms=a.warmup_ms, rep_ms=a.rep_ms)
            for name, metrics in timed.items():
                staged[name].update(metrics)
                staged[name]['effective_tflops'] = staged[name]['logical_gflop'] / (
                    staged[name]['latency_us'] / 1000.0)
                print(f'TIMING {lab} {name}: {staged[name]["latency_us"]:.3f} us '
                      f'{staged[name]["effective_tflops"]:.3f} effective TFLOP/s', flush=True)
        records.extend(staged.values())
        successful = [r for r in staged.values() if r['status'] == 'PASS' and 'latency_us' in r]
        if successful:
            base_name = {'G01': f'tile_{tiles[0][0]}x{tiles[0][1]}x{tiles[0][2]}',
                         'G02': 'group_m_1', 'G03': 'split_1', 'G04': 'ordinary_grid'}[lab]
            base = next((r for r in successful if r['variant'] == base_name), None)
            if base is not None:
                for row in successful:
                    row['speedup_vs_baseline'] = base['latency_us'] / row['latency_us']
        save_csv(a.out, records)
    if not any(r['status'] == 'PASS' for r in records):
        raise SystemExit('No successful configurations. Inspect CSV.')
    if any(r['status'] != 'PASS' for r in records):
        raise SystemExit('At least one variant failed; see CSV. Re-run with --strict to fail fast.')
    print('NOTE: G03 timings include both split-K GEMM and reduction kernel launches.', flush=True)


if __name__ == '__main__':
    main()
