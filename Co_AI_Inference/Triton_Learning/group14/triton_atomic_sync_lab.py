#!/usr/bin/env python3
"""Group14: GPU integer atomics, contention and multi-CTA reduction.

B01: same number of atomics with hot-spot versus spread destinations.
B02: per-lane atomics versus per-CTA aggregation on IDENTICAL logical bins.
B03: CTA atomic reduction versus scratch-buffer + second kernel.
Integer inputs/reference avoid FP32 atomic-order nondeterminism; no race-prone
cross-CTA spin locks or assumptions about thread-block scheduling.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_lab_common import (base_parser, bench, check, fail_on_errors, ints,
                            metadata, print_rows, require_gpu, save, validate_args)
try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise SystemExit("Requires Triton") from exc


@triton.jit
def scatter_atomic(VALUES, INDEX, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    valid = pos < N
    idx = tl.load(INDEX+pos, mask=valid, other=0)
    val = tl.load(VALUES+pos, mask=valid, other=0)
    tl.atomic_add(OUT+idx, val, mask=valid, sem="relaxed")


@triton.jit
def grouped_lane_atomic(VALUES, OUT, N: tl.constexpr,
                        BINS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pos = pid*BLOCK + tl.arange(0, BLOCK)
    val = tl.load(VALUES+pos, mask=pos<N, other=0)
    # Every active lane in one CTA updates the same bin.
    destinations = tl.zeros((BLOCK,), tl.int32) + (pid % BINS)
    tl.atomic_add(OUT+destinations, val, mask=pos<N, sem="relaxed")


@triton.jit
def grouped_cta_atomic(VALUES, OUT, N: tl.constexpr,
                       BINS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pos = pid*BLOCK + tl.arange(0, BLOCK)
    val = tl.load(VALUES+pos, mask=pos<N, other=0)
    # One global atomic per CTA, same logical indices as grouped_lane_atomic.
    amount = tl.sum(val, 0)
    tl.atomic_add(OUT+(pid%BINS), amount, sem="relaxed")


@triton.jit
def atomic_cta_sum(VALUES, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    val = tl.load(VALUES+pos, mask=pos<N, other=0)
    tl.atomic_add(OUT, tl.sum(val, 0), sem="relaxed")


@triton.jit
def partial_cta_sum(VALUES, PARTIAL, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pos = pid*BLOCK + tl.arange(0, BLOCK)
    val = tl.load(VALUES+pos, mask=pos<N, other=0)
    tl.store(PARTIAL+pid, tl.sum(val, 0))


@triton.jit
def finish_cta_sum(PARTIAL, OUT, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    val = tl.load(PARTIAL+i, mask=i<PARTS, other=0)
    tl.store(OUT, tl.sum(val, 0))


def main():
    p = base_parser(__doc__, lab_choices=["B01", "B02", "B03"])
    p.add_argument("--n", type=int, default=65536)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--bins", type=int, default=64)
    p.add_argument("--hot-bins", default="1,8,64")
    p.add_argument("--variant", default="all",
                   help="Select a variant: hot_N, lane, cta, atomic, two_pass")
    args = p.parse_args()
    validate_args(args)
    if args.n < 1 or args.n >= 2**30 or args.block < 32 or args.block & (args.block-1):
        p.error("n must be in [1, 2^30) and block must be a power of two >= 32")
    if args.bins < 1:
        p.error("bins must be positive")
    hot_bins = ints(args.hot_bins)
    if any(b > args.bins for b in hot_bins):
        p.error("hot-bins values cannot exceed bins")
    groups = triton.cdiv(args.n, args.block)
    if triton.next_power_of_2(groups) > 8192:
        p.error("B03's single-CTA final reduction supports <=8192 partial groups")
    torch, triton_rt = require_gpu()
    cpu_values = torch.ones(args.n, dtype=torch.int32)
    values = cpu_values.cuda()
    labs = ["B01", "B02", "B03"] if args.lab == "all" else [args.lab]
    rows = []

    def run(lab, name, output_size, expected, operation, *, atomic_ops, launches):
        row = {"lab": lab, "variant": name, "n": args.n, "block": args.block,
               "bins": output_size, "atomic_ops": atomic_ops,
               "launch_count_without_reset": launches}
        out = torch.zeros(output_size, dtype=torch.int32, device="cuda")
        try:
            # operation returns (complete benchmark callable, compiled kernels).
            launch, compiled_kernels = operation(out)
            launch()
            row["max_abs_error"] = check(torch, out, expected, atol=0, rtol=0)
            for j, kernel in enumerate(compiled_kernels):
                row.update({f"kernel{j}_{key}": val for key, val in
                            metadata(kernel, args.dump_asm or None,
                                     f"{lab}_{name}_k{j}").items()})
            row.update(bench(triton_rt, launch, args))
            row["status"] = "ok"
        except Exception as exc:
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)

    if "B01" in labs:
        for active_bins in hot_bins:
            name = f"hot_{active_bins}"
            if args.variant not in ("all", name):
                continue
            # Same number of increments, different destination distributions.
            indices_cpu = (torch.arange(args.n) % active_bins).to(torch.int32)
            indices = indices_cpu.cuda()
            expected = torch.bincount(indices_cpu.long(), minlength=args.bins).to(torch.int32)
            def operation(out, indices=indices):
                def launch():
                    # A correctly reinitialized output is required on EVERY trial.
                    out.zero_()
                    return scatter_atomic[(groups,)](values, indices, out,
                                                     args.n, args.block, num_warps=4)
                k = launch()
                return launch, [k]
            run("B01", name, args.bins, expected, operation,
                atomic_ops=args.n, launches=1)

    if "B02" in labs:
        ids = (torch.arange(args.n) // args.block % args.bins).long()
        expected = torch.bincount(ids, minlength=args.bins).to(torch.int32)
        for name, kernel in (("lane", grouped_lane_atomic), ("cta", grouped_cta_atomic)):
            if args.variant not in ("all", name):
                continue
            def operation(out, kernel=kernel):
                def launch():
                    out.zero_()
                    return kernel[(groups,)](values, out, args.n, args.bins,
                                             args.block, num_warps=4)
                k = launch()
                return launch, [k]
            run("B02", name, args.bins, expected, operation,
                atomic_ops=(args.n if name == "lane" else groups), launches=1)

    if "B03" in labs:
        expected = torch.tensor([args.n], dtype=torch.int32)
        if args.variant in ("all", "atomic"):
            def atomic_op(out):
                def launch():
                    out.zero_()
                    return atomic_cta_sum[(groups,)](values, out, args.n,
                                                     args.block, num_warps=4)
                k = launch()
                return launch, [k]
            run("B03", "atomic", 1, expected, atomic_op,
                atomic_ops=groups, launches=1)
        if args.variant in ("all", "two_pass"):
            partial = torch.empty(groups, dtype=torch.int32, device="cuda")
            def twopass_op(out):
                def launch():
                    k0 = partial_cta_sum[(groups,)](values, partial, args.n,
                                                   args.block, num_warps=4)
                    k1 = finish_cta_sum[(1,)](partial, out, groups,
                                              triton.next_power_of_2(groups),
                                              num_warps=4)
                    return k0, k1
                ks = launch()
                return launch, ks
            run("B03", "two_pass", 1, expected, twopass_op,
                atomic_ops=0, launches=2)

    save(rows, args.out, args, torch, triton_rt)
    print_rows(rows)
    if not rows:
        raise SystemExit("No matching variants")
    fail_on_errors(rows)


if __name__ == "__main__":
    main()
