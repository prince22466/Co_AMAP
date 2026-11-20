#!/usr/bin/env python3
"""Group5: shared-memory-associated Triton A/B experiments on BI-V150.

S01: transpose: untiled global-memory layout vs tiled conversion.
S02: GEMM: different output tiles, K reuse and compiler staging.
S03: debug_barrier: no explicit CTA barrier vs one per FMA iteration.
S04: optional Gluon shared-memory layout: un-swizzled vs swizzled.

S01/S02 are compiler-managed; shared memory is NOT guaranteed to appear.
S04 explicitly allocates shared memory but requires backend Gluon support.
Bank conflicts must be confirmed with generated code/hardware counters;
Triton source-level layouts alone do not prove bank conflicts occurred.

This lab never changes vLLM, FlagGems, or the competition benchmark.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from smem_lab_common import (
    compiled_info, dump_asm, environment, measure_pair, parse_int_list,
    runtime, save_csv, theoretical_gemm_operand_bytes,
)

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise SystemExit("Run in the BI-V150 environment with Triton installed") from exc

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    HAVE_GLUON = True
    GLUON_IMPORT_ERROR = ""
except (ImportError, AttributeError) as exc:
    HAVE_GLUON = False
    GLUON_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


@triton.jit
def transpose_untiled(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    # Output coalesced / input strided. No explicit shared memory.
    out_idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row_out = out_idx // M
    col_out = out_idx % M
    src_idx = col_out * N + row_out
    value = tl.load(X + src_idx, mask=out_idx < M * N, other=0)
    tl.store(Y + out_idx, value, mask=out_idx < M * N)


@triton.jit
def transpose_tiled(X, Y, M: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    r = tl.program_id(0) * TILE + tl.arange(0, TILE)
    c = tl.program_id(1) * TILE + tl.arange(0, TILE)
    x = tl.load(X + r[:, None] * N + c[None, :],
                mask=(r[:, None] < M) & (c[None, :] < N), other=0)
    xt = tl.trans(x)
    tl.store(Y + c[:, None] * M + r[None, :], xt,
             mask=(c[:, None] < N) & (r[None, :] < M))


@triton.jit
def gemm_tile_reuse(A, B, C,
                    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    accum = tl.full((BM, BN), 0, tl.float32)
    for ki in range(tl.cdiv(K, BK)):
        k = ki * BK + rk
        at = tl.load(A + rm[:, None] * K + k[None, :],
                     mask=(rm[:, None] < M) & (k[None, :] < K), other=0)
        bt = tl.load(B + k[:, None] * N + rn[None, :],
                     mask=(k[:, None] < K) & (rn[None, :] < N), other=0)
        accum = tl.dot(at, bt, accum)
    tl.store(C + rm[:, None] * N + rn[None, :], accum,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def barrier_cost(X, Y, N: tl.constexpr, BLOCK: tl.constexpr,
                 STEPS: tl.constexpr, WITH_BARRIER: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + idx, mask=idx < N, other=0)
    for _ in range(STEPS):
        value = tl.fma(value, 1.00001, 0.0001)
        if WITH_BARRIER:
            # Explicit CTA barrier; compiler/backend may choose implementation.
            tl.debug_barrier()
    tl.store(Y + idx, value, mask=idx < N)


if HAVE_GLUON:
    @gluon.jit
    def gluon_shared_layout_copy(X, Y,
                                 M: gl.constexpr, N: gl.constexpr,
                                 TILE: gl.constexpr, WARP: gl.constexpr,
                                 SWIZZLED: gl.constexpr):
        """Different *shared* layouts, identical global load/store layouts.

        All lanes in all participating warps reach the same CTA barrier.
        No NVIDIA-specific inline assembly or async-copy mechanism is used.
        """
        # Load coalesced across columns; export to a row-distributed layout.
        in_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1], [1, WARP], [4, 1], [1, 0])
        out_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1], [WARP, 1], [1, 4], [0, 1])
        r = gl.program_id(0) * TILE + gl.arange(0, TILE, gl.SliceLayout(1, in_layout))
        c = gl.program_id(1) * TILE + gl.arange(0, TILE, gl.SliceLayout(0, in_layout))
        value = gl.load(X + r[:, None] * N + c[None, :],
                        mask=(r[:, None] < M) & (c[None, :] < N), other=0)
        if SWIZZLED:
            smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                vec=1, per_phase=1, max_phase=4, order=[1, 0])
        else:
            smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                vec=1, per_phase=1, max_phase=1, order=[1, 0])
        smem = gl.allocate_shared_memory(gl.float32, [TILE, TILE], layout=smem_layout)
        smem.store(value)
        gl.barrier()
        out = smem.load(out_layout)
        rr = gl.program_id(0) * TILE + gl.arange(0, TILE, gl.SliceLayout(1, out_layout))
        cc = gl.program_id(1) * TILE + gl.arange(0, TILE, gl.SliceLayout(0, out_layout))
        gl.store(Y + rr[:, None] * N + cc[None, :], out,
                 mask=(rr[:, None] < M) & (cc[None, :] < N))


def detect_warp_size(torch, manual: int) -> tuple[int, str]:
    if manual:
        return manual, "user override"
    props = torch.cuda.get_device_properties(0)
    for attr in ("warp_size", "warpSize"):
        value = getattr(props, attr, None)
        if isinstance(value, int) and value > 0:
            return value, f"torch device property {attr}"
    name = torch.cuda.get_device_name(0).lower()
    if "iluvatar" in name or "bi-v150" in name:
        return 64, "BI-V150 fallback (verify with backend metadata)"
    return 32, "fallback (verify with backend metadata)"


def read_pair_tiles(spec: str) -> list[tuple[int, int]]:
    tiles = []
    for tok in spec.split(","):
        parts = tok.strip().lower().split("x")
        if len(parts) != 2:
            raise ValueError("--gemm-tiles expects BMxBN,BMxBN (e.g. 32x32,64x64)")
        bm, bn = int(parts[0]), int(parts[1])
        if min(bm, bn) < 16 or any(v & (v - 1) for v in (bm, bn)):
            raise ValueError("GEMM tile BM and BN must be powers of two >= 16")
        tiles.append((bm, bn))
    if len(tiles) != 2 or tiles[0] == tiles[1]:
        raise ValueError("Exactly two distinct GEMM tiles are required")
    return tiles


def run_one_pair(args, torch, name: str, shape_note: str, launch: dict,
                 check_output, extra: dict | None = None) -> list[dict]:
    """Compile, validate, alternate A/B measurements, and record metadata."""
    names = [args.variant] if args.variant else ["A", "B"]
    rows = []
    successful = {}
    for variant in names:
        row = {"lab": name, "variant": variant, "configuration": shape_note,
               **(extra or {})}
        try:
            handle = launch[variant]()
            torch.cuda.synchronize()
            check_output(variant)
            row.update(compiled_info(handle))
            dump_asm(handle, args.dump_asm, f"{name}_{variant}")
            row["status"] = "ok"
            successful[variant] = launch[variant]
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{name}/{variant} ERROR: {row['error']}", flush=True)
            rows.append(row)
            if args.strict:
                raise
            continue
        rows.append(row)
    if not args.single_run:
        if len(successful) == 2:
            try:
                timing = measure_pair(successful["A"], successful["B"],
                                      trials=args.trials, warmup_ms=args.warmup_ms,
                                      rep_ms=args.rep_ms)
                for row in rows:
                    if row["status"] == "ok":
                        row.update(timing[row["variant"]])
                print(f"{name}: A={timing['A']['latency_us']:.3f} us, "
                      f"B={timing['B']['latency_us']:.3f} us, "
                      f"B speedup={timing['A']['latency_us']/timing['B']['latency_us']:.3f}x", flush=True)
            except Exception as exc:
                for row in rows:
                    if row["status"] == "ok":
                        row["status"] = "error"
                        row["error"] = f"Benchmark error: {exc}"
                print(f"{name} TIMING ERROR: {exc}", flush=True)
                if args.strict:
                    raise
        else:
            from smem_lab_common import median_trial_us
            for row in rows:
                if row["status"] != "ok":
                    continue
                try:
                    data = median_trial_us(successful[row["variant"]],
                                           args.trials, args.warmup_ms, args.rep_ms)
                    row["latency_us"] = sorted(data)[len(data)//2]
                    row["trial_medians_us"] = ";".join(f"{v:.3f}" for v in data)
                    print(f"{name}/{row['variant']}: {row['latency_us']:.3f} us", flush=True)
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = f"Benchmark error: {exc}"
                    if args.strict:
                        raise
    else:
        print(f"{name}: single-run correctness/compile only", flush=True)
    return rows


def s01(args, torch):
    m, n, tile, block = args.m, args.n, args.transpose_tile, args.transpose_block
    x = torch.randn((m, n), device="cuda", dtype=torch.float32)
    y = torch.empty((n, m), device="cuda", dtype=torch.float32)
    expected = x.t().contiguous()
    launches = {
        "A": lambda: transpose_untiled[(triton.cdiv(m * n, block),)](
            x, y, m, n, block, num_warps=args.warps),
        "B": lambda: transpose_tiled[(triton.cdiv(m, tile), triton.cdiv(n, tile))](
            x, y, m, n, tile, num_warps=args.warps),
    }
    return run_one_pair(args, torch, "S01_transpose", f"M={m},N={n},tile={tile}",
                        launches, lambda v: torch.testing.assert_close(y, expected, atol=0, rtol=0),
                        {"logical_bytes": 2 * m * n * 4,
                         "interpretation": "Compiler-managed transpose; shared memory must be verified in IR"})


def s02(args, torch):
    m, n, k = args.m, args.n, args.k
    tile_a, tile_b = read_pair_tiles(args.gemm_tiles)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32).to(torch.bfloat16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)
    reference = a.float() @ b.float()
    tiles = {"A": tile_a, "B": tile_b}

    def make_launch(variant):
        bm, bn = tiles[variant]
        return lambda: gemm_tile_reuse[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            a, b, c, m, n, k, bm, bn, args.bk,
            num_warps=args.warps, num_stages=args.num_stages)

    result = run_one_pair(args, torch, "S02_gemm_tile_reuse",
                          f"M={m},N={n},K={k},BK={args.bk}",
                          {v: make_launch(v) for v in ("A", "B")},
                          lambda v: torch.testing.assert_close(c, reference, atol=0.03, rtol=0.02),
                          {"logical_flops": 2 * m * n * k,
                           "interpretation": "GEMM tiling changes reuse, programs, occupancy; does not isolate shared memory"})
    for rec in result:
        bm, bn = tiles[rec["variant"]]
        rec.update(tile_m=bm, tile_n=bn,
                   modeled_operand_request_bytes=theoretical_gemm_operand_bytes(m,n,k,bm,bn))
    return result


def s03(args, torch):
    n = args.elements
    x_cpu = torch.linspace(0.2, 0.8, n, dtype=torch.float32)
    x = x_cpu.to("cuda")
    y = torch.empty_like(x)
    ref = x_cpu.clone()
    for _ in range(args.barrier_steps):
        ref = ref * 1.00001 + 0.0001
    launches = {
        "A": lambda: barrier_cost[(triton.cdiv(n, args.barrier_block),)](
            x, y, n, args.barrier_block, args.barrier_steps, False, num_warps=args.warps),
        "B": lambda: barrier_cost[(triton.cdiv(n, args.barrier_block),)](
            x, y, n, args.barrier_block, args.barrier_steps, True, num_warps=args.warps),
    }
    return run_one_pair(args, torch, "S03_CTA_barrier",
                        f"elements={n},steps={args.barrier_steps},block={args.barrier_block}",
                        launches, lambda v: torch.testing.assert_close(y.cpu(), ref, atol=2e-4, rtol=3e-4),
                        {"barriers_per_program_A": 0,
                         "barriers_per_program_B_requested": args.barrier_steps,
                         "interpretation": "Explicit barrier overhead; not a data-sharing correctness test"})


def s04(args, torch, warp_size: int):
    if not HAVE_GLUON:
        print("S04 SKIP: Gluon not importable: " + GLUON_IMPORT_ERROR, flush=True)
        return [{"lab": "S04_gluon_shared_layout", "status": "skip",
                 "error": "Gluon import unavailable: " + GLUON_IMPORT_ERROR}]
    m, n, tile = args.m, args.n, args.smem_tile
    x = torch.randn((m, n), device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    launches = {
        "A": lambda: gluon_shared_layout_copy[(triton.cdiv(m, tile), triton.cdiv(n, tile))](
            x, y, m, n, tile, warp_size, False, num_warps=4),
        "B": lambda: gluon_shared_layout_copy[(triton.cdiv(m, tile), triton.cdiv(n, tile))](
            x, y, m, n, tile, warp_size, True, num_warps=4),
    }
    return run_one_pair(args, torch, "S04_gluon_shared_layout",
                        f"M={m},N={n},tile={tile},warp_size={warp_size}",
                        launches, lambda v: torch.testing.assert_close(y, x, atol=0, rtol=0),
                        {"layout_A": "SwizzledSharedLayout(vec=1,per_phase=1,max_phase=1)",
                         "layout_B": "SwizzledSharedLayout(vec=1,per_phase=1,max_phase=4)",
                         "requested_shared_bytes": tile * tile * 4,
                         "interpretation": "Explicit shared memory: bank conflicts require device counters/IR"})


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lab", choices=["all","S01","S02","S03","S04"], default="all")
    p.add_argument("--include-gluon", action="store_true",
                   help="Include optional S04 when --lab all (default: S01–S03 only)")
    p.add_argument("--variant", choices=["A","B"], default=None,
                   help="Only run A or B (useful for profiler capture)")
    p.add_argument("--single-run", action="store_true",
                   help="Compile, execute once, validate, and exit; don't time")
    p.add_argument("--strict", action="store_true", help="Raise on errors (S04 unavailable is still a skip)")
    p.add_argument("--m", type=int, default=512, help="Transpose/GEMM/shared-layout rows")
    p.add_argument("--n", type=int, default=512, help="Transpose/GEMM/shared-layout cols")
    p.add_argument("--k", type=int, default=512, help="GEMM reduction dimension")
    p.add_argument("--warps", type=int, default=4, help="CTA warps for S01–S03")
    p.add_argument("--warp-size", type=int, choices=[0, 32, 64], default=0,
                   help="S04: 0 automatically detects/guesses BI-V150=64, others=32")
    p.add_argument("--transpose-tile", type=int, default=32)
    p.add_argument("--transpose-block", type=int, default=256)
    p.add_argument("--gemm-tiles", default="32x32,64x64")
    p.add_argument("--bk", type=int, default=32)
    p.add_argument("--num-stages", type=int, default=2)
    p.add_argument("--elements", type=int, default=262144)
    p.add_argument("--barrier-block", type=int, default=256)
    p.add_argument("--barrier-steps", type=int, default=64)
    p.add_argument("--smem-tile", type=int, default=64)
    p.add_argument("--warmup-ms", type=int, default=30)
    p.add_argument("--rep-ms", type=int, default=100)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dump-asm", default=None)
    p.add_argument("--out", default="results/group5.csv")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if min(args.m,args.n,args.k,args.elements,args.barrier_steps,
           args.trials,args.warmup_ms,args.rep_ms,args.num_stages) <= 0:
        p.error("Dimensions, step counts and timing settings must be > 0")
    for name in ("warps","transpose_tile","transpose_block","bk","barrier_block","smem_tile"):
        value = getattr(args, name)
        if value < 1 or value & (value - 1):
            p.error(f"--{name.replace('_','-')} must be a positive power of two")
    if args.bk < 16:
        p.error("--bk must be >= 16 for BF16 tl.dot")
    if args.smem_tile < 32:
        p.error("--smem-tile must be at least 32")
    try:
        read_pair_tiles(args.gemm_tiles)
    except ValueError as e:
        p.error(str(e))
    if args.variant and args.lab == "all":
        p.error("--variant requires one selected --lab")
    torch, installed_triton = runtime()
    torch.manual_seed(args.seed)
    warp_size, origin = detect_warp_size(torch, args.warp_size)
    print(f"GPU={torch.cuda.get_device_name(0)}, torch={torch.__version__}, "
          f"triton={installed_triton.__version__}, warp_size={warp_size} ({origin})", flush=True)
    environment(args.out, torch, installed_triton, args, warp_size)
    runners = {"S01": lambda: s01(args, torch), "S02": lambda: s02(args, torch),
               "S03": lambda: s03(args, torch), "S04": lambda: s04(args, torch, warp_size)}
    results = []
    selected = (["S01", "S02", "S03"] + (["S04"] if args.include_gluon else [])
                if args.lab == "all" else [args.lab])
    for lab in selected:
        print(f"\n== {lab} ==", flush=True)
        try:
            results.extend(runners[lab]())
        except Exception as e:
            results.append({"lab": lab, "status": "error", "error": f"{type(e).__name__}: {e}"})
            save_csv(args.out, results)
            if args.strict:
                raise
            print(f"{lab} ERROR: {e}", flush=True)
    save_csv(args.out, results)
    failures = [r for r in results if r.get("status") == "error"]
    if failures or not any(r.get("status") == "ok" for r in results):
        raise SystemExit(f"Failed/unsupported configurations: errors={len(failures)}, "
                         f"successful={sum(r.get('status') == 'ok' for r in results)}")
    print("Note: compiler shared_bytes and bank-conflict counters must be checked before claiming shared-memory effects.")


if __name__ == "__main__":
    main()
