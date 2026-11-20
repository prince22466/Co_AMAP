#!/usr/bin/env python3
"""Triton scheduling/resource tradeoff lab: blocks x warps x live accumulators.

Each configuration has the same logical FP32 FMA count per output element,
and the same global input/output bytes. Changing accumulator count also
changes FMA dependency depth and reduction work: it is NOT a pure occupancy
experiment. To isolate warp effects, compare rows with the same block and
accumulator count. Actual occupancy requires device hardware counters.
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
def accumulator_sweep(
    X, A, B, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_ACC: tl.constexpr,
    ITERS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0)
    a = tl.load(A + offsets, mask=mask, other=1)
    b = tl.load(B + offsets, mask=mask, other=0)
    # Shape [NUM_ACC, BLOCK]: NUM_ACC independent live accumulator vectors.
    # Must be power of two due to tl.arange. The outer dimension increases
    # register footprint and exposes more ILP (two simultaneous effects).
    row = tl.arange(0, NUM_ACC)
    acc = x[None, :] + row[:, None].to(tl.float32) * 0.001
    for _ in range(ITERS):
        acc = tl.fma(acc, a[None, :], b[None, :])
    # Reduction cost grows with NUM_ACC; it is included in measured latency.
    result = tl.sum(acc, axis=0) / NUM_ACC
    tl.store(Y + offsets, result, mask=mask)


def reference(cpu, num_acc: int, steps: int, samples: int):
    x, a, b = [t[:samples].double().tolist() for t in cpu]
    expected = []
    for xx, aa, bb in zip(x, a, b):
        acc = [xx + i * 0.001 for i in range(num_acc)]
        for _ in range(steps):
            acc = [v * aa + bb for v in acc]
        expected.append(sum(acc) / num_acc)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--blocks", default="128,256,512")
    parser.add_argument("--warps", default="4,8")
    parser.add_argument("--accumulators", default="1,2,4,8,16")
    parser.add_argument("--fma-per-element", type=int, default=512,
                        help="Constant across configs; divisible by accumulator counts")
    parser.add_argument("--out", default="results/occupancy.csv")
    args = parser.parse_args()
    validate_common_args(args)
    blocks = parse_powers_of_two(args.blocks, what="block")
    warps = parse_powers_of_two(args.warps, what="num_warps")
    accumulators = parse_powers_of_two(args.accumulators, what="accumulators")
    if args.fma_per_element <= 0 or any(args.fma_per_element % v for v in accumulators):
        parser.error("--fma-per-element must be > 0 and divisible by all accumulator counts")

    torch, installed_triton = require_runtime()
    cpu, (x, a, b) = setup_inputs(torch, args.n, args.seed)
    y = torch.empty_like(x)
    write_environment(str(Path(args.out).with_suffix(".env.json")), torch, installed_triton, vars(args))
    refs = {nacc: reference(cpu, nacc, args.fma_per_element // nacc, min(args.n, 24))
            for nacc in accumulators}
    records = []

    for block in blocks:
        for num_acc in accumulators:
            for warps_count in warps:
                iters = args.fma_per_element // num_acc
                rec = {"test": "occupancy_resource", "n": args.n,
                       "block": block, "num_warps": warps_count,
                       "num_acc": num_acc, "iters_per_chain": iters,
                       "fma_per_element": args.fma_per_element,
                       "logical_gflop": 2.0 * args.n * args.fma_per_element / 1e9}
                try:
                    launch = lambda: accumulator_sweep[(installed_triton.cdiv(args.n, block),)](
                        x, a, b, y, args.n, block, num_acc, iters,
                        num_warps=warps_count,
                    )
                    compiled = launch()
                    torch.cuda.synchronize()
                    validate_sample(torch, y, refs[num_acc])
                    rec.update(compiled_metadata(compiled))
                    dump_kernel_asm(compiled, args.dump_asm,
                                    f"occupancy_a{num_acc}_b{block}_w{warps_count}")
                    rec.update(benchmark_ms(launch, args.warmup_ms, args.rep_ms, args.trials))
                    rec["effective_tflops"] = rec["logical_gflop"] / (rec["latency_us"] / 1000.0)
                    rec["status"] = "ok"
                    print(f"acc={num_acc:2d} block={block:4d} warps={warps_count:2d} "
                          f"{rec['latency_us']:9.2f} us, {rec['effective_tflops']:.3f} TFLOP/s "
                          f"regs={rec['registers_per_thread']} spills={rec['spills']}")
                except Exception as exc:
                    rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    print(f"FAILED acc={num_acc} block={block} warps={warps_count}: {rec['error']}")
                    if args.strict:
                        records.append(rec)
                        append_csv(args.out, records)
                        raise
                records.append(rec)
    append_csv(args.out, records)
    if not any(r.get("status") == "ok" for r in records):
        raise SystemExit("No valid configurations; inspect errors in CSV")
    print("NOTE: registers/spills are compiler metadata; actual occupancy requires GPU counters.")
    print("Isolate warp effect by filtering equal (block, num_acc) across num_warps.")


if __name__ == "__main__":
    main()
