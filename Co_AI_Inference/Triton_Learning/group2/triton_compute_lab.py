#!/usr/bin/env python3
"""Compute-oriented Triton A/B lab for BI-V150 and other compatible GPUs.

The two variants of each experiment perform the same mathematical task, with
identical external tensor addresses, shapes, dtypes, and logical memory IO.
Large arithmetic repetition amortizes memory access; no kernel is literally
memory-free. Do not use this file as a competition scoring benchmark.

Examples:
  python3 triton_compute_lab.py --lab C01
  python3 triton_compute_lab.py --lab all --csv compute.csv
  python3 triton_compute_lab.py --lab C06 --variant candidate --native
  python3 triton_compute_lab.py --lab C03 --dump-ir-dir ir_output
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl


@triton.jit
def fma_chain_kernel(X, G, B, Y, N: tl.constexpr, REPS: tl.constexpr,
                     BLOCK: tl.constexpr, FUSED: tl.constexpr,
                     UNROLL: tl.constexpr):
    """Purpose: compare arithmetic contraction/unrolling at fixed IO.

    Inputs: X,G,B are N FP32 elements; REPS recurrence steps; BLOCK lanes;
        FUSED selects explicit fused FP32 FMA; UNROLL controls loop unrolling.
    Output: Y[i] = repeatedly applied (Y[i] * G[i] + B[i]).
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    g = tl.load(G + i, i < N, other=0)
    b = tl.load(B + i, i < N, other=0)
    for _ in tl.range(0, REPS, loop_unroll_factor=UNROLL):
        if FUSED:
            x = tl.fma(x, g, b)
        else:
            x = (x * g) + b
    tl.store(Y + i, x, i < N)


@triton.jit
def ilp_kernel(X, G, B, Y, N: tl.constexpr, REPS: tl.constexpr,
               BLOCK: tl.constexpr, INTERLEAVED: tl.constexpr):
    """Purpose: compare dependent vs independent FP32 FMA scheduling.

    Inputs: one X,G,B element/lane; four nonidentical accumulator initializers;
        REPS recurrence steps for each accumulator.
    Output: sum of four recurrence results in Y; same mathematical operation
        and external memory IO for both variants.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    g = tl.load(G + i, i < N, other=0)
    b = tl.load(B + i, i < N, other=0)
    a0 = x
    a1 = x + 0.125
    a2 = x * 0.75
    a3 = 0.03125 - x
    if INTERLEAVED:
        for _ in range(REPS):
            a0 = tl.fma(a0, g, b)
            a1 = tl.fma(a1, g, b)
            a2 = tl.fma(a2, g, b)
            a3 = tl.fma(a3, g, b)
    else:
        for _ in range(REPS):
            a0 = tl.fma(a0, g, b)
        for _ in range(REPS):
            a1 = tl.fma(a1, g, b)
        for _ in range(REPS):
            a2 = tl.fma(a2, g, b)
        for _ in range(REPS):
            a3 = tl.fma(a3, g, b)
    tl.store(Y + i, (a0 + a1) + (a2 + a3), i < N)


@triton.jit
def polynomial_kernel(X, Y, N: tl.constexpr, REPS: tl.constexpr,
                      BLOCK: tl.constexpr, ESTRIN: tl.constexpr):
    """Purpose: evaluate the same degree-7 polynomial by Horner or Estrin.

    Inputs: X, N elements, REPS polynomial composition steps, tile BLOCK.
    Output: Y[i] = polynomial composed REPS times on X[i].
    The two evaluation orders differ slightly in FP32 rounding.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    for _ in range(REPS):
        if ESTRIN:
            x2 = x * x
            x4 = x2 * x2
            p01 = tl.fma(x, 0.65, 0.02)
            p23 = tl.fma(x, -0.08, 0.11)
            p45 = tl.fma(x, -0.025, 0.05)
            p67 = tl.fma(x, 0.002, 0.01)
            q0 = tl.fma(p23, x2, p01)
            q1 = tl.fma(p67, x2, p45)
            x = tl.fma(q1, x4, q0)
        else:
            p = tl.full((BLOCK,), 0.002, tl.float32)
            p = tl.fma(p, x, 0.01)
            p = tl.fma(p, x, -0.025)
            p = tl.fma(p, x, 0.05)
            p = tl.fma(p, x, -0.08)
            p = tl.fma(p, x, 0.11)
            p = tl.fma(p, x, 0.65)
            x = tl.fma(p, x, 0.02)
    tl.store(Y + i, x, i < N)


@triton.jit
def gemm_acc_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr,
                    K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr, DOT_ACC: tl.constexpr):
    """Purpose: compare tl.dot accumulation vs separate dot and add.

    Inputs: contiguous BF16 A[M,K], B[K,N]; same tile size and launch grid.
    Output: FP32 C[M,N] = A @ B (within BF16 input precision and FP32
        accumulation-roundoff tolerance). Global loads/stores are identical.
    """
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for offset in range(tl.cdiv(K, BK)):
        kk = offset * BK + k
        a = tl.load(A + m[:, None] * K + kk[None, :],
                    (m[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(B + kk[:, None] * N + n[None, :],
                    (kk[:, None] < K) & (n[None, :] < N), other=0)
        if DOT_ACC:
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        else:
            part = tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
            acc = acc + part
    tl.store(C + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


LABS = {
    "C01": ("Separate MUL+ADD vs explicit FMA", "FP32 arithmetic execution"),
    "C02": ("Serial vs interleaved independent FMA chains", "instruction-level parallelism"),
    "C03": ("Horner vs Estrin polynomial", "dependency-chain length"),
    "C04": ("4 vs 8 warps per program", "compute occupancy"),
    "C05": ("Rolled loop vs loop unroll=4", "loop instruction overhead"),
    "C06": ("tl.dot partial + add vs tl.dot with accumulator", "matrix compute"),
}


def args_parser() -> argparse.Namespace:
    """Purpose: parse reproducible lab options. Inputs: CLI. Output: Namespace."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lab", choices=[*LABS, "all"], default="all")
    p.add_argument("--variant", choices=["vanilla", "candidate"], help="Run just one variant")
    p.add_argument("--n", type=int, default=1 << 18, help="FP32 elements for C01-C05")
    p.add_argument("--iters", type=int, default=512, help="Base recurrence repetitions (scaled by lab)")
    p.add_argument("--gemm-m", type=int, default=1024)
    p.add_argument("--gemm-n", type=int, default=1024)
    p.add_argument("--gemm-k", type=int, default=2048)
    p.add_argument("--samples", type=int, default=7, help="Independent event samples per variant")
    p.add_argument("--batch-launches", type=int, default=12, help="Kernel launches per timed sample")
    p.add_argument("--csv", help="Write A/B result table to CSV")
    p.add_argument("--native", action="store_true", help="One warmed launch inside CUDA profiler range")
    p.add_argument("--dump-ir-dir", help="Save available compiler IR/assembly per variant")
    return p.parse_args()


def ensure_environment(a: argparse.Namespace) -> None:
    """Purpose: refuse invalid options and unsupported devices. Output: None."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA-compatible GPU is visible to PyTorch")
    for name in ("n", "iters", "gemm_m", "gemm_n", "gemm_k", "samples", "batch_launches"):
        if getattr(a, name) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    print(json.dumps({"torch": torch.__version__, "triton": triton.__version__,
                      "device": torch.cuda.get_device_name(0), "dtype": "fp32 + bf16 GEMM"}))


def prepare_inputs(a: argparse.Namespace, lab: str) -> tuple[dict, torch.Tensor]:
    """Purpose: allocate and initialize all tensors outside timed launches.

    Inputs: options and lab ID. Output: dict of operands plus output tensor.
    """
    torch.manual_seed(1337)
    if lab == "C06":
        aa = torch.randn((a.gemm_m, a.gemm_k), device="cuda", dtype=torch.float32).to(torch.bfloat16)
        bb = torch.randn((a.gemm_k, a.gemm_n), device="cuda", dtype=torch.float32).to(torch.bfloat16)
        out = torch.empty((a.gemm_m, a.gemm_n), device="cuda", dtype=torch.float32)
        return {"a": aa, "b": bb, "out": out}, out
    x = (torch.rand(a.n, device="cuda") - 0.5) * 0.5
    g = 0.97 + 0.02 * torch.rand(a.n, device="cuda")
    b = 0.004 * (torch.rand(a.n, device="cuda") - 0.5)
    out = torch.empty_like(x)
    return {"x": x, "g": g, "b": b, "out": out}, out


def build_launch(a: argparse.Namespace, lab: str, variant: str, t: dict) -> tuple[Callable, int, int, str]:
    """Purpose: configure one kernel with fixed IO and chosen compute strategy.

    Inputs: CLI, lab ID, variant, preallocated tensors. Output: launch callback,
        useful FLOP count per launch, logical bytes per launch and kernel name.
    """
    candidate = variant == "candidate"
    if lab == "C06":
        m, n, k = a.gemm_m, a.gemm_n, a.gemm_k
        bm, bn, bk = 32, 64, 32
        def launch():
            return gemm_acc_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                t["a"], t["b"], t["out"], m, n, k, bm, bn, bk, candidate,
                num_warps=4)
        flops = 2 * m * n * k
        bytes_io = 2 * (m * k + k * n) + 4 * m * n
        return launch, flops, bytes_io, "gemm_acc_kernel"

    n = a.n
    block = 256
    if lab == "C01":
        reps = a.iters
        def launch():
            return fma_chain_kernel[(triton.cdiv(n, block),)](
                t["x"], t["g"], t["b"], t["out"], n, reps,
                block, candidate, 1, num_warps=4, enable_fp_fusion=False)
        flops = 2 * n * reps
        kernel = "fma_chain_kernel"
    elif lab == "C02":
        reps = max(1, a.iters // 4)
        def launch():
            return ilp_kernel[(triton.cdiv(n, block),)](
                t["x"], t["g"], t["b"], t["out"], n, reps,
                block, candidate, num_warps=4)
        flops = 8 * n * reps
        kernel = "ilp_kernel"
    elif lab == "C03":
        reps = max(1, a.iters // 4)
        def launch():
            return polynomial_kernel[(triton.cdiv(n, block),)](
                t["x"], t["out"], n, reps,
                block, candidate, num_warps=4)
        flops = 14 * n * reps  # same useful polynomial work; Estrin has 2 extra multiplications
        bytes_io = 8 * n
        return launch, flops, bytes_io, "polynomial_kernel"
    elif lab == "C04":
        reps = a.iters
        def launch():
            return fma_chain_kernel[(triton.cdiv(n, block),)](
                t["x"], t["g"], t["b"], t["out"], n, reps,
                block, True, 1, num_warps=8 if candidate else 4)
        flops = 2 * n * reps
        kernel = "fma_chain_kernel"
    elif lab == "C05":
        reps = a.iters
        def launch():
            return fma_chain_kernel[(triton.cdiv(n, block),)](
                t["x"], t["g"], t["b"], t["out"], n, reps,
                block, True, 4 if candidate else 1, num_warps=4)
        flops = 2 * n * reps
        kernel = "fma_chain_kernel"
    else:
        raise ValueError(lab)
    return launch, flops, 16 * n, kernel


def verify_pair(vanilla: torch.Tensor, candidate: torch.Tensor, lab: str) -> tuple[float, float]:
    """Purpose: enforce numerical agreement. Inputs: two outputs and lab.

    Output: maximum absolute and relative difference. Raises on failure.
    """
    if not torch.isfinite(vanilla).all() or not torch.isfinite(candidate).all():
        raise AssertionError(f"{lab} produced non-finite values")
    abs_diff = (vanilla - candidate).abs()
    max_abs = abs_diff.max().item()
    max_rel = (abs_diff / vanilla.abs().clamp_min(1e-5)).max().item()
    tol = {"C01": (2e-4, 2e-4), "C03": (2e-4, 2e-4),
           "C06": (0.02, 0.002)}.get(lab, (1e-5, 1e-5))
    if not torch.allclose(vanilla, candidate, atol=tol[0], rtol=tol[1]):
        raise AssertionError(f"{lab} mismatches: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}; tolerances={tol}")
    return max_abs, max_rel


def dump_ir(handle: object, path: str | None, lab: str, variant: str) -> None:
    """Purpose: save compiler output when available. Output: files in path."""
    if not path:
        return
    target = Path(path) / lab / variant
    target.mkdir(parents=True, exist_ok=True)
    asm = getattr(handle, "asm", None)
    if not isinstance(asm, dict):
        print(f"NOTE: no compiler .asm dict available for {lab}/{variant}")
        return
    for kind, output in asm.items():
        if isinstance(output, str):
            (target / (str(kind) + ".txt")).write_text(output, encoding="utf-8")
    print(f"IR {lab}/{variant}: {target} ({','.join(asm)})")


def time_one_batch(launch: Callable, batches: int) -> float:
    """Purpose: measure GPU event time across identical warmed kernel launches.

    Inputs: launch callback, number of launches. Output: ms per launch.
    Host submission gaps may contribute if kernels are too short.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(batches):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / batches


def paired_timing(launches: dict, variants: list[str],
                  samples: int, batches: int) -> dict[str, tuple[float, float]]:
    """Purpose: alternate A/B benchmark order to mitigate thermal/time bias.

    Inputs: variant-to-launch mapping, variants, samples, batch launches.
    Output: per-variant median and median absolute deviation in ms/launch.
    """
    observations: dict[str, list[float]] = {v: [] for v in variants}
    for sample in range(samples):
        order = variants if sample % 2 == 0 else list(reversed(variants))
        for v in order:
            observations[v].append(time_one_batch(launches[v][0], batches))
    results = {}
    for v in variants:
        median = statistics.median(observations[v])
        mad = statistics.median(abs(x - median) for x in observations[v])
        results[v] = (median, mad)
    return results


def run_lab(a: argparse.Namespace, lab: str) -> list[dict]:
    """Purpose: correctness-test variants, then benchmark compute throughput.

    Inputs: options and lab name. Output: structured result rows for CSV.
    """
    t, out = prepare_inputs(a, lab)
    variants = [a.variant] if a.variant else ["vanilla", "candidate"]
    launches = {v: build_launch(a, lab, v, t) for v in variants}
    output_snapshots: dict[str, torch.Tensor] = {}
    handles = {}
    for v in variants:
        launch, _, _, _ = launches[v]
        handles[v] = launch()  # JIT compile outside any profiler/timing region
        torch.cuda.synchronize()
        output_snapshots[v] = out.clone()
        dump_ir(handles[v], a.dump_ir_dir, lab, v)
    if len(variants) == 2:
        diff, rel = verify_pair(output_snapshots["vanilla"], output_snapshots["candidate"], lab)
        print(f"CORRECT {lab}: max_abs={diff:.4g}, max_rel={rel:.4g}")
        if lab == "C06":
            # Independent correctness reference, outside measured/profiler region.
            reference = torch.matmul(t["a"].float(), t["b"].float())
            for v in variants:
                if not torch.allclose(output_snapshots[v], reference, atol=0.02, rtol=0.002):
                    deviation = (output_snapshots[v] - reference).abs().max().item()
                    raise AssertionError(f"C06/{v} disagrees with torch.matmul: max_abs={deviation}")
            print("CORRECT C06: both variants agree with independent FP32 torch.matmul")
    else:
        print(f"SINGLE-VARIANT {lab}/{variants[0]}: compare with vanilla separately before acceptance")

    for v in variants:
        for _ in range(3):
            launches[v][0]()
    torch.cuda.synchronize()
    if a.native:
        if not a.variant:
            raise ValueError("--native requires --variant to isolate one kernel")
        try:
            torch.cuda.profiler.start()
            launches[a.variant][0]()
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()
        print(f"PROFILE_OK {lab} {a.variant}")
        return []
    rows = []
    times = paired_timing(launches, variants, a.samples, a.batch_launches)
    for v in variants:
        launch, flops, bytes_io, kernel_name = launches[v]
        ms, mad = times[v]
        gflops = flops / (ms / 1000.0) / 1e9
        intensity = flops / bytes_io
        row = dict(lab=lab, variant=v, kernel=kernel_name, median_ms=round(ms, 6),
                   mad_ms=round(mad, 6), useful_gflops=round(gflops, 2),
                   logical_bytes=bytes_io, useful_flops=flops,
                   arithmetic_intensity=round(intensity, 3))
        rows.append(row)
        print(f"{lab} {v:9s} {ms*1000:9.2f} us  ±MAD {mad*1000:6.2f} us  "
              f"{gflops/1000:8.3f} useful TFLOP/s  AI={intensity:.1f} FLOP/B")
    if len(variants) == 2:
        by = {r["variant"]: r for r in rows}
        speedup = by["vanilla"]["median_ms"] / by["candidate"]["median_ms"]
        print(f"{lab} candidate/vanilla speedup = {speedup:.3f}x "
              f"({'faster' if speedup > 1 else 'slower or equal'})")
    return rows


def main() -> None:
    """Purpose: run requested experiments and optionally export CSV. Output: stdout/CSV."""
    a = args_parser()
    ensure_environment(a)
    labs = list(LABS) if a.lab == "all" else [a.lab]
    rows = []
    for lab in labs:
        print(f"\n--- {lab}: {LABS[lab][0]} ---")
        rows.extend(run_lab(a, lab))
    if a.csv and rows:
        with open(a.csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved: {a.csv}")


if __name__ == "__main__":
    main()
