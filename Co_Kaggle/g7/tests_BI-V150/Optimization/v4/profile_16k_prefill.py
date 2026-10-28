#!/usr/bin/env python3
"""BI-V150 long-prefill attention tile experiment (NOT a competition benchmark).

Measure the *existing* FlagGems varlen kernel, with identical Q/K/V and output
buffer for each tile. Reports GPU-event latency and numerical deviation from
baseline. Theoretical K/V load instructions are NOT physical DRAM counters.
Use the vendor hardware profiler for L2 misses, DRAM reads and achieved occupancy.
"""

import argparse
import csv
import logging
import math
import os
import re
import statistics
from pathlib import Path

import torch

import flag_gems
from flag_gems.config import use_c_extension
from flag_gems.ops.flash_api import mha_varlan_fwd, mha_varlan_fwd_opt
from flag_gems.ops.iluvatar_prefill_tiles import (
    TILE_CANDIDATES,
    ENV_NAME,
    select_prefill_tiles,
)

MODES = ("baseline", *TILE_CANDIDATES, "auto")
META_RE = re.compile(
    r"BLOCK_M=(\d+) BLOCK_N=(\d+) regs/thread=(\S+) "
    r"spills/thread=(\S+) shared_bytes=(\S+) warps=(\S+)"
)


class KernelMetadata(logging.Handler):
    def __init__(self):
        super().__init__()
        self.last = {}

    def emit(self, record):
        match = META_RE.search(record.getMessage())
        if match:
            m, n, regs, spills, shared, warps = match.groups()
            self.last = dict(
                block_m=int(m),
                block_n=int(n),
                registers_per_thread=regs,
                spills_per_thread=spills,
                shared_bytes=shared,
                num_warps=warps,
            )


def ceiling(n, d):
    return (n + d - 1) // d


def modeled_kv_bytes(query_len, kv_len, head_dim, q_heads, kv_heads,
                     batch, block_m, block_n, bytes_per_element, causal):
    """Logical per-CTA K/V load demand, ignoring L1/L2 and masked edge bytes.

    For a full causal prefill with Q length = KV length, each Q CTA streams a
    growing K/V prefix. This provides a *tile comparison proxy*, not a
    measurement of physical K/V bytes transferred from DRAM.
    """
    loads_per_head = 0
    for m in range(ceiling(query_len, block_m)):
        max_keys = min(kv_len, (m + 1) * block_m) if causal else kv_len
        loads_per_head += ceiling(max_keys, block_n) * block_n
    logical = loads_per_head * head_dim * bytes_per_element * 2 * q_heads * batch
    minimum = kv_len * head_dim * bytes_per_element * 2 * kv_heads * batch
    return logical, minimum


def measured_occupancy_bound(meta, args):
    """Optional resource-based CTA/SM limit, NOT hardware achieved occupancy."""
    try:
        regs = int(meta["registers_per_thread"])
        shared = int(meta["shared_bytes"])
        warps = int(meta["num_warps"])
    except (KeyError, ValueError):
        return None
    if not (args.registers_per_sm and args.shared_bytes_per_sm
            and args.threads_per_sm and args.max_ctas_per_sm):
        return None
    threads = warps * args.warp_size
    if threads <= 0 or regs <= 0:
        return None
    resident = min(
        args.max_ctas_per_sm,
        args.registers_per_sm // (regs * threads),
        args.shared_bytes_per_sm // shared if shared else args.max_ctas_per_sm,
        args.threads_per_sm // threads,
    )
    return round(resident * threads / args.threads_per_sm, 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--contiguous-kv", action="store_true")
    parser.add_argument("--entrypoint", choices=("regular", "optimized"),
                        default="regular", help="Choose which varlen forward path to test")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", default="v4_prefill_tiles.csv")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--registers-per-sm", type=int)
    parser.add_argument("--shared-bytes-per-sm", type=int)
    parser.add_argument("--threads-per-sm", type=int)
    parser.add_argument("--max-ctas-per-sm", type=int)
    parser.add_argument("--warp-size", type=int, default=32,
                        help="Override for the actual BI-V150 lane-group size")
    args = parser.parse_args()

    if flag_gems.vendor_name != "iluvatar":
        raise SystemExit("This experiment requires the Iluvatar FlagGems backend")
    if use_c_extension:
        raise SystemExit(
            "FlagGems C-extension attention bypasses the patched Triton kernel; "
            "disable the C extension before profiling v4"
        )
    if not torch.cuda.is_available():
        raise SystemExit("Requires a CUDA-compatible BI-V150 device and installed FlagGems")
    if args.seq_len < 8192:
        raise SystemExit("This harness is for long prefill (>=8192 tokens)")
    if args.heads <= 0 or args.kv_heads <= 0 or args.heads % args.kv_heads:
        raise SystemExit("--heads must be positive and divisible by --kv-heads")
    if args.head_dim <= 0 or args.head_dim > 256 or args.head_dim % 8:
        raise SystemExit("--head-dim must be a multiple of 8 in [8, 256]")
    if args.page_size <= 0 or args.batch <= 0 or args.iterations <= 0 or args.warmup < 1:
        raise SystemExit("Positive page size, batch and iterations required")

    torch.manual_seed(2026)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n = args.seq_len
    h = args.heads
    hk = args.kv_heads
    d = args.head_dim
    q = torch.randn((args.batch * n, h, d), dtype=dtype, device=device)
    if args.contiguous_kv:
        k = torch.randn((args.batch * n, hk, d), dtype=dtype, device=device)
        v = torch.randn_like(k)
        page_table = None
    else:
        pages_per_seq = ceiling(n, args.page_size)
        pages = pages_per_seq * args.batch
        k = torch.randn((pages, args.page_size, hk, d), dtype=dtype, device=device)
        v = torch.randn_like(k)
        page_table = torch.arange(pages, device=device, dtype=torch.int32).reshape(
            args.batch, pages_per_seq
        )
    cu = torch.arange(args.batch + 1, device=device, dtype=torch.int32) * n
    # Match vLLM's paged-KV call contract (seqused_k is mandatory).
    seqused_k = (
        torch.full((args.batch,), n, device=device, dtype=torch.int32)
        if page_table is not None
        else None
    )
    out = torch.empty_like(q)
    lse = torch.empty((h, args.batch * n), dtype=torch.float32, device=device)
    scale = 1 / math.sqrt(d)

    def call():
        # Exercise the two existing FlagGems varlen Triton launch paths.
        common = (
            cu, cu, seqused_k, None, page_table, None,
            n, n, 0.0, scale, False, True, -1, -1, 0.0, False, None,
        )
        if args.entrypoint == "optimized":
            return mha_varlan_fwd_opt(q, k, v, out, lse, *common)[0]
        return mha_varlan_fwd(q, k, v, out, *common)[0]

    capture = KernelMetadata()
    logger = logging.getLogger("flag_gems.ops.flash_api")
    previous_logger_level = logger.level
    previous_report = os.environ.get("FLAGGEMS_ILUVATAR_PREFILL_REPORT")
    logger.addHandler(capture)
    logger.setLevel(logging.WARNING)
    os.environ["FLAGGEMS_ILUVATAR_PREFILL_REPORT"] = "1"
    original_mode = os.environ.get(ENV_NAME)
    rows = []
    reference = None
    try:
        for mode in dict.fromkeys(("baseline", *args.modes)):
            os.environ[ENV_NAME] = mode
            capture.last = {}
            os.environ["FLAGGEMS_ILUVATAR_PREFILL_REPORT"] = "1"
            try:
                for _ in range(args.warmup):
                    call()
                torch.cuda.synchronize()
                # Reject silent fallback to guessed baseline tile parameters.
                if not capture.last:
                    raise RuntimeError(
                        "No v4 Triton varlen kernel report captured. Verify that "
                        "the local FlagGems installation is loaded and C-extension "
                        "attention is disabled."
                    )
                expected = select_prefill_tiles(
                    max_seqlen_q=n, max_seqlen_k=n, head_size=d,
                    is_dropout=False, is_local=False, mode=mode,
                )
                if expected is not None:
                    actual = (capture.last["block_m"], capture.last["block_n"])
                    if actual != expected:
                        raise RuntimeError(
                            f"Tile selection failed: requested {expected}, got {actual}"
                        )
                # Metadata logging is a CPU-side delay; disable it for timed runs.
                os.environ["FLAGGEMS_ILUVATAR_PREFILL_REPORT"] = "0"
                # GPU event envelope; use hardware profiler for exact kernel cycles.
                timings = []
                for _ in range(args.iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    result = call()
                    end.record()
                    end.synchronize()
                    timings.append(start.elapsed_time(end))
                os.environ["FLAGGEMS_ILUVATAR_PREFILL_REPORT"] = "1"
                result = result.clone()
                if reference is None:
                    reference = result
                    deviation = 0.0
                    correct = bool(torch.isfinite(result).all().item())
                    if not correct:
                        raise RuntimeError("Baseline kernel produced non-finite output")
                else:
                    deviation = (result.float() - reference.float()).abs().max().item()
                    correct = bool(torch.allclose(
                        result.float(), reference.float(), rtol=0.03, atol=0.03
                    ))
                meta = capture.last
                bm = meta["block_m"]
                bn = meta["block_n"]
                logical, minimum = modeled_kv_bytes(
                    n, n, d, h, hk, args.batch, bm, bn, q.element_size(), True
                )
                row = dict(
                    mode=mode,
                    status="ok",
                    median_gpu_ms=round(statistics.median(timings), 4),
                    min_gpu_ms=round(min(timings), 4),
                    max_abs_error=round(deviation, 6),
                    correct_vs_baseline=correct,
                    logical_kv_load_bytes=logical,
                    unique_kv_bytes_lower_bound=minimum,
                    physical_dram_read_bytes="requires_vendor_hardware_profiler",
                    achieved_occupancy="requires_vendor_hardware_profiler",
                    resource_occupancy_upper_bound=measured_occupancy_bound(meta, args),
                    **meta,
                )
                rows.append(row)
                print(row, flush=True)
            except Exception as exc:
                # An experimental tile can fail compilation or resource limits.
                # Record it without losing measurements of remaining candidates.
                if mode == "baseline":
                    raise
                os.environ["FLAGGEMS_ILUVATAR_PREFILL_REPORT"] = "1"
                row = dict(
                    mode=mode,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    correct_vs_baseline=False,
                )
                rows.append(row)
                print(row, flush=True)
    finally:
        logger.removeHandler(capture)
        logger.setLevel(previous_logger_level)
        if original_mode is None:
            os.environ.pop(ENV_NAME, None)
        else:
            os.environ[ENV_NAME] = original_mode
        if previous_report is None:
            os.environ.pop("FLAGGEMS_ILUVATAR_PREFILL_REPORT", None)
        else:
            os.environ["FLAGGEMS_ILUVATAR_PREFILL_REPORT"] = previous_report

    if rows:
        baseline = rows[0]["median_gpu_ms"]
        for row in rows:
            measured = row.get("median_gpu_ms")
            row["speedup_vs_baseline"] = round(baseline / measured, 4) if measured else ""
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {output}")
        if not all(row["correct_vs_baseline"] for row in rows):
            raise SystemExit("Output mismatch: do not enable a candidate")


if __name__ == "__main__":
    main()
