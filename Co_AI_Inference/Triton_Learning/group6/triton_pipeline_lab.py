#!/usr/bin/env python3
"""Group6 — advanced Triton pipeline performance lab (BI-V150 oriented).

P01: serial load->compute->load vs early issue of both independent loads.
P02: manual 0/1/2/4 iteration lookahead with the same logical load count.
P03: tl.range software-pipeline num_stages sweep in BF16 GEMM.
P04: stage-count x reduction length (K) to study fill/drain overhead.

All tests compare the same mathematical output within each fixed-shape group.
P01/P02 merely expose opportunities for overlap; a source change is NOT proof
that the compiler generated asynchronous loads or a different machine pipeline.
This is not a competition benchmark; no vLLM/FlagGems files are changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_lab_common import (
    check_sample, cpu_fma_series, dump_asm, kernel_info, paired_bench,
    positive_ints, runtime, sample_ids, write_csv, write_env,
)

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise SystemExit("Use the installed PyTorch+Triton runtime on BI-V150") from exc


@triton.jit
def load_compute_order_kernel(A, B, Y, N: tl.constexpr, STEPS: tl.constexpr,
                              BLOCK: tl.constexpr, WORK: tl.constexpr,
                              EARLY_B_LOAD: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    acc_a = tl.full((BLOCK,), 0., tl.float32)
    acc_b = tl.full((BLOCK,), 0., tl.float32)
    # Explicit stages=1 avoids requesting multi-stage loop pipelining here;
    # the backend may still reorder and overlap ordinary loads.
    for t in tl.range(0, STEPS, num_stages=1):
        va = tl.load(A + t * N + idx, mask=mask, other=0) * 0.2
        if EARLY_B_LOAD:
            vb = tl.load(B + t * N + idx, mask=mask, other=0) * 0.2
        for _ in range(WORK):
            acc_a = tl.fma(acc_a, 0.75, va)
        if not EARLY_B_LOAD:
            vb = tl.load(B + t * N + idx, mask=mask, other=0) * 0.2
        for _ in range(WORK):
            acc_b = tl.fma(acc_b, 0.75, vb)
    tl.store(Y + idx, acc_a + acc_b, mask=mask)


@triton.jit
def prefetch_distance_kernel(X, Y, N: tl.constexpr, STEPS: tl.constexpr,
                             BLOCK: tl.constexpr, WORK: tl.constexpr,
                             DISTANCE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    acc = tl.full((BLOCK,), 0., tl.float32)
    if DISTANCE == 0:
        for t in tl.range(0, STEPS, num_stages=1):
            v = tl.load(X + t * N + idx, mask=mask, other=0) * 0.2
            for _ in range(WORK):
                acc = tl.fma(acc, 0.75, v)
    else:
        v0 = tl.load(X + idx, mask=mask, other=0) * 0.2
        if DISTANCE >= 2:
            v1 = tl.load(X + N + idx, mask=mask, other=0) * 0.2
        if DISTANCE == 4:
            v2 = tl.load(X + 2 * N + idx, mask=mask, other=0) * 0.2
            v3 = tl.load(X + 3 * N + idx, mask=mask, other=0) * 0.2
        for t in tl.range(0, STEPS, num_stages=1):
            # Fetch the future iteration BEFORE doing arithmetic on v0;
            # do not confuse this with guaranteed async-copy instructions.
            if t + DISTANCE < STEPS:
                incoming = tl.load(X + (t + DISTANCE) * N + idx,
                                   mask=mask, other=0) * 0.2
            else:
                incoming = tl.full((BLOCK,), 0., tl.float32)
            for _ in range(WORK):
                acc = tl.fma(acc, 0.75, v0)
            if DISTANCE == 1:
                v0 = incoming
            elif DISTANCE == 2:
                v0 = v1
                v1 = incoming
            else:
                v0 = v1
                v1 = v2
                v2 = v3
                v3 = incoming
    tl.store(Y + idx, acc, mask=mask)


@triton.jit
def gemm_pipeline_kernel(A, B, C,
                         M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                         STAGES: tl.constexpr):
    im = tl.program_id(0) * BM + tl.arange(0, BM)
    jn = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    # tl.range(num_stages=...) is a compiler software-pipeline hint.
    # Some vendor backends might reject multi-stage lowering; record failure.
    for step in tl.range(0, tl.cdiv(K, BK), num_stages=STAGES):
        k = step * BK + kk
        at = tl.load(A + im[:, None] * K + k[None, :],
                     mask=(im[:, None] < M) & (k[None, :] < K), other=0)
        bt = tl.load(B + k[:, None] * N + jn[None, :],
                     mask=(k[:, None] < K) & (jn[None, :] < N), other=0)
        acc = tl.dot(at, bt, acc)
    tl.store(C + im[:, None] * N + jn[None, :], acc,
             mask=(im[:, None] < M) & (jn[None, :] < N))


def check_matrix(torch, got, a, b):
    reference = a.float() @ b.float()
    # BF16 inputs, FP32 output, potentially different matrix accumulation order.
    torch.testing.assert_close(got, reference, atol=0.03, rtol=0.02)


def run_variants(args, torch, lab: str, configs: list[dict],
                 launch_maker, verify, base: dict | None = None):
    """Compile+validate every configuration, then rotate timing order."""
    rows = []
    launches = {}
    for cfg in configs:
        key = cfg["key"]
        rec = {"lab": lab, **(base or {}), **cfg}
        try:
            run = launch_maker(cfg)
            handle = run()  # compilation outside timing
            torch.cuda.synchronize()
            verify(cfg)
            rec.update(kernel_info(handle))
            dump_asm(handle, args.dump_asm, f"{lab}_{key}")
            rec["status"] = "ok"
            launches[key] = run
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{lab}/{key} FAILED: {rec['error']}", flush=True)
            rows.append(rec)
            if args.strict:
                raise
            continue
        rows.append(rec)
    if not args.single_run and launches:
        try:
            measured = paired_bench(launches, args.trials, args.warmup_ms, args.rep_ms)
            for row in rows:
                if row["status"] == "ok":
                    row.update(measured[row["key"]])
                    if "useful_flops" in row:
                        row["effective_tflops"] = row["useful_flops"] / (row["latency_us"] * 1e6)
                    if "logical_read_bytes" in row:
                        row["logical_requested_GBps"] = row["logical_read_bytes"] / (row["latency_us"] * 1e3)
                    print(f"{lab}/{row['key']}: {row['latency_us']:.3f} us, "
                          f"regs={row['registers_per_thread']} "
                          f"spills={row['spills']} shared={row['shared_bytes']}", flush=True)
        except Exception as exc:
            for row in rows:
                if row["status"] == "ok":
                    row.update(status="error", error=f"Timing: {type(exc).__name__}: {exc}")
            if args.strict:
                raise
    elif args.single_run:
        print(f"{lab}: compiled, launched and validated; no timing (--single-run)", flush=True)
    return rows


def p01(args, torch, installed_triton):
    n = args.elements
    steps = args.steps
    torch.manual_seed(args.seed)
    a = torch.randn((steps, n), device="cuda", dtype=torch.float32) * 0.1
    b = torch.randn((steps, n), device="cuda", dtype=torch.float32) * 0.1
    out = torch.empty(n, device="cuda", dtype=torch.float32)
    ids = sample_ids(torch, n)
    ref_a = cpu_fma_series(a[:, ids].cpu().numpy(), args.work)
    ref_b = cpu_fma_series(b[:, ids].cpu().numpy(), args.work)
    reference = ref_a + ref_b
    variants = [{"key": "serial", "early_b_load": 0},
                {"key": "early", "early_b_load": 1}]
    if args.variant:
        variants = [x for x in variants if x["key"] == args.variant]
    if not variants:
        raise ValueError("P01 --variant must be serial or early")

    def launch(cfg):
        return lambda: load_compute_order_kernel[(installed_triton.cdiv(n, args.block),)](
            a, b, out, n, steps, args.block, args.work, cfg["early_b_load"],
            num_warps=args.warps)

    def verify(_cfg):
        check_sample(torch, out, reference, ids)

    return run_variants(args, torch, "P01", variants, launch, verify,
                        {"elements": n, "steps": steps, "block": args.block,
                         "num_warps": args.warps, "work": args.work,
                         "logical_read_bytes": 2 * n * steps * 4})


def p02(args, torch, installed_triton):
    n = args.elements
    steps = args.steps
    distances = []
    for token in args.prefetch_distances.split(","):
        val = int(token.strip())
        if val not in (0, 1, 2, 4):
            raise ValueError("--prefetch-distances supports 0,1,2,4")
        if val not in distances:
            distances.append(val)
    if steps < max(distances) or not distances:
        raise ValueError("--steps must be >= largest prefetch distance")
    torch.manual_seed(args.seed)
    x = torch.randn((steps, n), device="cuda", dtype=torch.float32) * 0.1
    out = torch.empty(n, device="cuda", dtype=torch.float32)
    ids = sample_ids(torch, n)
    reference = cpu_fma_series(x[:, ids].cpu().numpy(), args.work)
    variants = [{"key": f"d{d}", "prefetch_distance": d} for d in distances]
    if args.variant:
        variants = [x for x in variants if x["key"] == args.variant]
    if not variants:
        raise ValueError("P02 --variant must be one of " + ",".join(f"d{d}" for d in distances))

    def launch(cfg):
        return lambda: prefetch_distance_kernel[(installed_triton.cdiv(n, args.block),)](
            x, out, n, steps, args.block, args.work, cfg["prefetch_distance"],
            num_warps=args.warps)

    def verify(_cfg):
        check_sample(torch, out, reference, ids)

    return run_variants(args, torch, "P02", variants, launch, verify,
                        {"elements": n, "steps": steps, "block": args.block,
                         "num_warps": args.warps, "work": args.work,
                         "logical_read_bytes": n * steps * 4})


def gemm_case(args, torch, installed_triton, lab: str, k: int, stages: list[int]):
    m, n = args.m, args.n
    torch.manual_seed(args.seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32).to(torch.bfloat16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)
    variants = [{"key": f"k{k}_s{s}", "stages": s, "k": k} for s in stages]
    if args.variant:
        variants = [x for x in variants if x["key"] == args.variant]
    if not variants:
        raise ValueError("GEMM --variant should be k<K>_s<S>, matching selected sizes")

    def launch(cfg):
        return lambda: gemm_pipeline_kernel[(installed_triton.cdiv(m, args.bm),
                                              installed_triton.cdiv(n, args.bn))](
            a, b, c, m, n, k, args.bm, args.bn, args.bk, cfg["stages"],
            num_warps=args.warps)

    def verify(_cfg):
        check_matrix(torch, c, a, b)

    return run_variants(args, torch, lab, variants, launch, verify,
                        {"m": m, "n": n, "k": k, "bm": args.bm, "bn": args.bn,
                         "bk": args.bk, "num_warps": args.warps,
                         "useful_flops": 2 * m * n * k,
                         "tensor_bytes_lower_bound": 2 * (m*k + k*n) + 4 * m*n})


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lab", choices=("all", "P01", "P02", "P03", "P04"), default="all")
    p.add_argument("--variant", help="Run only one variant (for device profiler); e.g. serial, d2, k1024_s2")
    p.add_argument("--single-run", action="store_true", help="Compile + launch + validate once; do not time")
    p.add_argument("--strict", action="store_true", help="Raise immediately on first kernel failure")
    p.add_argument("--out", default="results/group6.csv")
    p.add_argument("--dump-asm", metavar="DIRECTORY")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup-ms", type=int, default=30)
    p.add_argument("--rep-ms", type=int, default=100)
    p.add_argument("--elements", type=int, default=262144)
    p.add_argument("--steps", type=int, default=16, help="Streaming steps for P01/P02")
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--work", type=int, default=8, help="Dependent FMAs per loaded value")
    p.add_argument("--prefetch-distances", default="0,1,2,4")
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=1024, help="GEMM K dimension for P03")
    p.add_argument("--k-sizes", default="128,512,2048", help="GEMM K sweep for P04")
    p.add_argument("--bm", type=int, default=32)
    p.add_argument("--bn", type=int, default=64)
    p.add_argument("--bk", type=int, default=32)
    p.add_argument("--warps", type=int, default=4)
    p.add_argument("--stages", default="1,2,3,4", help="Compiler loop software-pipeline stage counts")
    args = p.parse_args()
    if min(args.elements, args.steps, args.block, args.work, args.m, args.n, args.k,
           args.bm, args.bn, args.bk, args.warps, args.trials,
           args.warmup_ms, args.rep_ms) <= 0:
        p.error("All sizes/counts/timing parameters must be positive")
    for name in ("block", "bm", "bn", "bk", "warps"):
        value = getattr(args, name)
        if value & (value - 1):
            p.error(f"--{name} must be a power of two")
    if args.bm < 16 or args.bn < 16 or args.bk < 16:
        p.error("GEMM tile sizes must be >= 16")
    if args.variant and args.lab == "all":
        p.error("--variant requires selecting one lab")
    return args


def main():
    args = arguments()
    stages = positive_ints(args.stages, name="pipeline stages", allowed={1, 2, 3, 4})
    k_sizes = positive_ints(args.k_sizes, name="K sizes")
    torch, installed_triton = runtime()
    write_env(args.out, torch, installed_triton, args)
    print(f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} "
          f"triton={installed_triton.__version__}", flush=True)
    rows = []
    labs = ["P01", "P02", "P03", "P04"] if args.lab == "all" else [args.lab]
    for lab in labs:
        try:
            if lab == "P01":
                rows.extend(p01(args, torch, installed_triton))
            elif lab == "P02":
                rows.extend(p02(args, torch, installed_triton))
            elif lab == "P03":
                rows.extend(gemm_case(args, torch, installed_triton, lab, args.k, stages))
            else:
                for k in k_sizes:
                    rows.extend(gemm_case(args, torch, installed_triton, lab, k, stages))
        except Exception as exc:
            print(f"{lab} FAILED: {type(exc).__name__}: {exc}", flush=True)
            if args.strict:
                if rows:
                    write_csv(args.out, rows)
                raise
            rows.append({"lab": lab, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    write_csv(args.out, rows)
    failures = [r for r in rows if r.get("status") != "ok"]
    if failures:
        raise SystemExit(f"{len(failures)} failed configurations: see CSV; results incomplete")
    print("PASS: all requested kernels compile, launch and agree with references.", flush=True)
    if not args.single_run:
        print("NOTE: source lookahead / stages do not prove hardware overlap; verify IR/ISA + counters.", flush=True)


if __name__ == "__main__":
    main()
