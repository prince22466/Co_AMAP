#!/usr/bin/env python3
"""Standalone BI-V150 A/B experiment: *same* varlen operator, paged KV lookup.

Not the competition benchmark. Times GPU-event launch envelopes; use the vendor
profiler for true HBM/L2 transactions and occupancy. Never modifies tile
selection, model hyperparameters or benchmark scripts.
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
from flag_gems.ops.iluvatar_grouped_paged_kv import ENV_NAME

META_RE = re.compile(
    r"V4_GROUPED_PAGED_KV_KERNEL q=(\d+) kv=(\d+) paged=(\S+) "
    r"BLOCK_M=(\d+) BLOCK_N=(\d+) grouped_page_lookup=(\S+) "
    r"regs/thread=(\S+) spills/thread=(\S+) shared_bytes=(\S+) warps=(\S+)"
)


class KernelReport(logging.Handler):
    def __init__(self):
        super().__init__()
        self.last = None

    def emit(self, record):
        match = META_RE.search(record.getMessage())
        if match is None:
            return
        q, kv, paged, m, n, grouped, regs, spills, shared, warps = match.groups()
        self.last = dict(
            query_len=int(q),
            kv_len=int(kv),
            paged=(paged == "True"),
            block_m=int(m),
            block_n=int(n),
            grouped_page_lookup=(grouped == "True"),
            registers_per_thread=regs,
            spills_per_thread=spills,
            shared_bytes=shared,
            num_warps=warps,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, default=16384,
                   help="total KV length including previously cached context")
    p.add_argument("--query-len", type=int, default=None,
                   help="query chunk length; defaults to --seq-len")
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--entrypoint", choices=("regular", "optimized"),
                   default="regular")
    p.add_argument("--shuffle-pages", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--output", default="profiling/v4_grouped_paged_kv.csv")
    args = p.parse_args()

    nq = args.query_len if args.query_len is not None else args.seq_len
    nk = args.seq_len
    if (
        not 128 <= nq <= nk
        or nk < 8192
        or args.heads <= 0
        or args.kv_heads <= 0
        or args.heads % args.kv_heads
        or args.head_dim <= 0
        or args.head_dim > 256
        or args.head_dim % 8
        or args.batch <= 0
        or args.page_size <= 0
        or args.warmup < 1
        or args.iterations < 1
    ):
        raise SystemExit("Invalid long-prefill geometry or iteration counts")
    if flag_gems.vendor_name != "iluvatar" or not torch.cuda.is_available():
        raise SystemExit("Requires the Iluvatar BI-V150 GPU and FlagGems")
    if use_c_extension:
        raise SystemExit("C-extension attention bypasses the patched Triton kernel")

    torch.manual_seed(2026)
    dtype, device = torch.bfloat16, "cuda"
    h, hk, d = args.heads, args.kv_heads, args.head_dim
    q = torch.randn((args.batch * nq, h, d), dtype=dtype, device=device)
    pages_per_seq = (nk + args.page_size - 1) // args.page_size
    pages = args.batch * pages_per_seq
    k = torch.randn((pages, args.page_size, hk, d), dtype=dtype, device=device)
    v = torch.randn_like(k)
    table = (
        torch.randperm(pages, device=device, dtype=torch.int64).to(torch.int32)
        if args.shuffle_pages
        else torch.arange(pages, device=device, dtype=torch.int32)
    )
    page_table = table.reshape(args.batch, pages_per_seq)
    cu_q = torch.arange(args.batch + 1, device=device, dtype=torch.int32) * nq
    cu_k = torch.arange(args.batch + 1, device=device, dtype=torch.int32) * nk
    seqused_k = torch.full((args.batch,), nk, device=device, dtype=torch.int32)
    out = torch.empty_like(q)
    lse = torch.empty((h, args.batch * nq), device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(d)

    def launch():
        common = (
            cu_q, cu_k, seqused_k, None, page_table, None,
            nq, nk, 0.0, scale, False, True, -1, -1, 0.0, False, None,
        )
        if args.entrypoint == "optimized":
            return mha_varlan_fwd_opt(q, k, v, out, lse, *common)[0]
        return mha_varlan_fwd(q, k, v, out, *common)[0]

    logger = logging.getLogger("flag_gems.ops.flash_api")
    original_level = logger.level
    before = {
        ENV_NAME: os.environ.get(ENV_NAME),
        "FLAGGEMS_V4_GROUPED_REPORT": os.environ.get("FLAGGEMS_V4_GROUPED_REPORT"),
    }
    capture = KernelReport()
    logger.addHandler(capture)
    logger.setLevel(logging.WARNING)
    rows = []
    reference = None
    baseline_meta = None
    try:
        for mode in ("baseline", "grouped"):
            os.environ[ENV_NAME] = mode
            os.environ["FLAGGEMS_V4_GROUPED_REPORT"] = "1"
            capture.last = None
            for _ in range(args.warmup):
                launch()
            torch.cuda.synchronize()
            meta = capture.last
            if meta is None:
                raise RuntimeError("No varlen kernel report: wrong FlagGems import or backend")
            if not meta["paged"] or meta["query_len"] != nq or meta["kv_len"] != nk:
                raise RuntimeError(f"Unexpected kernel shape / paged mode: {meta}")
            if meta["grouped_page_lookup"] != (mode == "grouped"):
                raise RuntimeError(f"Grouped specialization did not activate: {meta}")
            if baseline_meta is None:
                baseline_meta = meta
            elif (meta["block_m"], meta["block_n"], meta["num_warps"]) != (
                baseline_meta["block_m"], baseline_meta["block_n"],
                baseline_meta["num_warps"],
            ):
                raise RuntimeError("A/B test changed tile/warp selection")

            # The trace would distort timing; use GPU events with logging disabled.
            os.environ["FLAGGEMS_V4_GROUPED_REPORT"] = "0"
            timings = []
            result = None
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = launch()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
            result = result.clone()
            if not torch.isfinite(result).all().item():
                raise RuntimeError(f"{mode} returned non-finite output")
            if reference is None:
                reference = result
                correct, error = True, 0.0
            else:
                error = (result.float() - reference.float()).abs().max().item()
                correct = bool(torch.allclose(
                    result.float(), reference.float(), rtol=0.03, atol=0.03
                ))
            row = dict(
                mode=mode, entrypoint=args.entrypoint, query_len=nq, kv_len=nk,
                status="ok", median_gpu_ms=round(statistics.median(timings), 4),
                min_gpu_ms=round(min(timings), 4),
                max_abs_error=round(error, 6),
                correct_vs_baseline=correct, **meta,
            )
            rows.append(row)
            print(row, flush=True)
    finally:
        logger.removeHandler(capture)
        logger.setLevel(original_level)
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    baseline_ms = rows[0]["median_gpu_ms"]
    for row in rows:
        row["speedup_vs_baseline"] = round(
            baseline_ms / row["median_gpu_ms"], 4
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}", flush=True)
    if not all(row["correct_vs_baseline"] for row in rows):
        raise SystemExit("BF16 output mismatch: reject grouped candidate")


if __name__ == "__main__":
    main()
