#!/usr/bin/env python3
"""FMA dependency A/B: four serial FMAs vs four independent FMA chains.

Equivalent FLOP counts and identical input/output traffic; the arithmetic
functions intentionally differ. Each output is checked against a FP64 CPU
reference, not compared directly across A/B variants.

Run with the already-installed BI-V150 Triton/PyTorch stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lab_common import (
    add_common_arguments, append_csv, benchmark_ms, compiled_metadata,
    dump_kernel_asm, parse_powers_of_two, require_runtime, setup_inputs,
    validate_common_args, validate_sample, write_environment,
)

try:
    import triton
    import triton.language as tl
except ImportError:
    raise SystemExit("Run in an environment with Triton installed")


@triton.jit
def serial_fma(X, A, B, Y, N: tl.constexpr, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    x = tl.load(X + idx, mask=mask, other=0)
    a = tl.load(A + idx, mask=mask, other=1)
    b = tl.load(B + idx, mask=mask, other=0)
    acc = x
    for _ in range(ITERS):
        acc = tl.fma(acc, a, b)
        acc = tl.fma(acc, a, b)
        acc = tl.fma(acc, a, b)
        acc = tl.fma(acc, a, b)
    tl.store(Y + idx, acc, mask=mask)


@triton.jit
def parallel_fma(X, A, B, Y, N: tl.constexpr, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    x = tl.load(X + idx, mask=mask, other=0)
    a = tl.load(A + idx, mask=mask, other=1)
    b = tl.load(B + idx, mask=mask, other=0)
    c0 = x
    c1 = x + 0.001
    c2 = x + 0.002
    c3 = x + 0.003
    for _ in range(ITERS):
        c0 = tl.fma(c0, a, b)
        c1 = tl.fma(c1, a, b)
        c2 = tl.fma(c2, a, b)
        c3 = tl.fma(c3, a, b)
    tl.store(Y + idx, 0.25 * ((c0 + c1) + (c2 + c3)), mask=mask)


def reference(cpu, mode: str, steps: int, samples: int):
    x, a, b = [t[:samples].double().tolist() for t in cpu]
    expected = []
    for xx, aa, bb in zip(x, a, b):
        if mode == "serial":
            val = xx
            for _ in range(steps * 4):
                val = val * aa + bb
            expected.append(val)
        else:
            acc = [xx + delta for delta in (0.0, 0.001, 0.002, 0.003)]
            for _ in range(steps):
                acc = [v * aa + bb for v in acc]
            expected.append(sum(acc) * 0.25)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--blocks", default="128,256", help="Comma-separated powers of two")
    parser.add_argument("--warps", default="4", help="Comma-separated powers of two")
    parser.add_argument("--iters", type=int, default=256, help="Loop iterations; 4 FMAs per iteration")
    parser.add_argument("--out", default="results/ilp.csv")
    args = parser.parse_args()
    validate_common_args(args)
    if args.iters <= 0:
        parser.error("--iters must be positive")
    blocks = parse_powers_of_two(args.blocks, what="block")
    warps = parse_powers_of_two(args.warps, what="num_warps")

    torch, installed_triton = require_runtime()
    cpu, (x, a, b) = setup_inputs(torch, args.n, args.seed)
    y = torch.empty_like(x)
    write_environment(str(Path(args.out).with_suffix(".env.json")), torch, installed_triton, vars(args))
    ref = {mode: reference(cpu, mode, args.iters, min(args.n, 24))
           for mode in ("serial", "parallel")}
    records = []

    for block in blocks:
        for warps_count in warps:
            for mode, kernel in (("serial", serial_fma), ("parallel", parallel_fma)):
                rec = {"test": "ilp", "mode": mode, "n": args.n,
                       "block": block, "num_warps": warps_count,
                       "iters": args.iters, "fma_per_element": 4 * args.iters,
                       "logical_gflop": 2.0 * args.n * 4 * args.iters / 1e9}
                try:
                    launch = lambda: kernel[(installed_triton.cdiv(args.n, block),)](
                        x, a, b, y, args.n, block, args.iters,
                        num_warps=warps_count,
                    )
                    compiled = launch()
                    torch.cuda.synchronize()
                    validate_sample(torch, y, ref[mode])
                    rec.update(compiled_metadata(compiled))
                    dump_kernel_asm(compiled, args.dump_asm, f"ilp_{mode}_b{block}_w{warps_count}")
                    rec.update(benchmark_ms(launch, args.warmup_ms, args.rep_ms, args.trials))
                    rec["effective_tflops"] = rec["logical_gflop"] / (rec["latency_us"] / 1000.0)
                    rec["status"] = "ok"
                    print(f"{mode:8s} block={block:4d} warps={warps_count:2d} "
                          f"{rec['latency_us']:9.2f} us, {rec['effective_tflops']:.3f} TFLOP/s "
                          f"regs={rec['registers_per_thread']} spills={rec['spills']}")
                except Exception as exc:
                    rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    print(f"FAILED {mode} block={block} warps={warps_count}: {rec['error']}")
                    if args.strict:
                        records.append(rec)
                        append_csv(args.out, records)
                        raise
                records.append(rec)
    append_csv(args.out, records)
    # Only compare pairs with same block, warps and problem size.
    for block in blocks:
        for w in warps:
            group = {r["mode"]: r for r in records if r.get("status") == "ok"
                     and r["block"] == block and r["num_warps"] == w}
            if len(group) == 2:
                ratio = group["serial"]["latency_us"] / group["parallel"]["latency_us"]
                print(f"ILP speedup block={block} warps={w}: parallel/serial = {ratio:.3f}x")
    if not any(r.get("status") == "ok" for r in records):
        raise SystemExit("No valid configurations; inspect errors in CSV")


if __name__ == "__main__":
    main()
