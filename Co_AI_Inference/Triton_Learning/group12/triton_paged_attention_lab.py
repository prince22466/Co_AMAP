#!/usr/bin/env python3
"""Group12: exact, single-token paged GQA attention (4K/16K).

A01: token-wise table indexing versus one table load per page.
A02: 16-token versus 32-token online-softmax tiles (same result).
All kernels consume the SAME page table and K/V tensors, with no quantization,
sparse attention, top-k or benchmark/serving changes. These are decode-like
microbenchmarks, not a full prefill FlashAttention implementation.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_lab_common import (base_parser, bench, check, fail_on_errors, ints,
                            metadata, print_rows, require_gpu, save, validate_args)
try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise SystemExit("Run on target with Triton installed") from exc


@triton.jit
def paged_online_decode(Q, K, V, TABLE, O,
                        CONTEXT: tl.constexpr, D: tl.constexpr,
                        Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr,
                        PAGE: tl.constexpr, TILE: tl.constexpr,
                        PAGE_SCALAR: tl.constexpr, SCALE: tl.constexpr):
    head = tl.program_id(0)
    kv_head = head // (Q_HEADS // KV_HEADS)
    d = tl.arange(0, D)
    lane = tl.arange(0, TILE)
    q = tl.load(Q + head * D + d)
    running_m = tl.full((), -float("inf"), tl.float32)
    running_l = tl.full((), 0, tl.float32)
    running_v = tl.full((D,), 0, tl.float32)

    for chunk in range(tl.cdiv(CONTEXT, TILE)):
        t = chunk * TILE + lane
        valid = t < CONTEXT
        if PAGE_SCALAR:
            # Legal only for TILE == PAGE: every lane belongs to one KV page.
            page = tl.load(TABLE + chunk)
            physical = ((page * KV_HEADS + kv_head) * PAGE + lane)
            addresses = physical[:, None] * D + d[None, :]
        else:
            # Full logical table lookup for each token (TILE may span pages).
            pages = tl.load(TABLE + (t // PAGE), mask=valid, other=0)
            physical = ((pages * KV_HEADS + kv_head) * PAGE + t % PAGE)
            addresses = physical[:, None] * D + d[None, :]
        k = tl.load(K + addresses, mask=valid[:, None] & (d[None, :] < D), other=0)
        v = tl.load(V + addresses, mask=valid[:, None] & (d[None, :] < D), other=0)
        logits = tl.sum(k * q[None, :], 1) * SCALE
        logits = tl.where(valid, logits, -float("inf"))
        tile_m = tl.max(logits, 0)
        next_m = tl.maximum(running_m, tile_m)
        old_scale = tl.exp(running_m - next_m)
        p = tl.exp(logits - next_m)
        running_v = running_v * old_scale + tl.sum(p[:, None] * v, 0)
        running_l = running_l * old_scale + tl.sum(p, 0)
        running_m = next_m
    tl.store(O + head * D + d, running_v / running_l)


def reference(torch, q, k, v, table, context, page, q_heads, kv_heads):
    """Independent CPU float64 stable softmax over the actual block-table order."""
    token_idx = torch.arange(context, dtype=torch.long)
    physical_page = table[token_idx // page].long()
    offset = token_idx % page
    outputs = []
    for head in range(q_heads):
        kv_head = head // (q_heads // kv_heads)
        keys = k[physical_page, kv_head, offset].double()
        vals = v[physical_page, kv_head, offset].double()
        logits = (keys @ q[head].double()) / math.sqrt(q.shape[-1])
        probs = torch.softmax(logits, dim=0)
        outputs.append((probs @ vals).float())
    return torch.stack(outputs, 0)


def setup(torch, context, page, q_heads, kv_heads, dim, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    npages = triton.cdiv(context, page)
    q = torch.randn((q_heads, dim), generator=generator)
    k = torch.randn((npages, kv_heads, page, dim), generator=generator)
    v = torch.randn((npages, kv_heads, page, dim), generator=generator)
    table = torch.randperm(npages, generator=generator).to(torch.int32)
    target = reference(torch, q, k, v, table, context, page, q_heads, kv_heads)
    return (q.cuda(), k.cuda(), v.cuda(), table.cuda()), target


def main():
    parser = base_parser(__doc__, lab_choices=["A01", "A02"])
    parser.add_argument("--contexts", default="4096,16384")
    parser.add_argument("--page", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--variant", choices=["all", "token16", "page16", "token32"],
                        default="all")
    args = parser.parse_args()
    validate_args(args)
    contexts = ints(args.contexts)
    if args.page <= 0 or args.page & (args.page - 1):
        parser.error("page must be a power of two")
    if args.dim < 16 or args.dim & (args.dim - 1):
        parser.error("dim must be a power of two and >= 16")
    if args.q_heads < 1 or args.kv_heads < 1 or args.q_heads % args.kv_heads:
        parser.error("q-heads must be a positive multiple of kv-heads")
    if args.page != 16:
        parser.error("this controlled suite uses PAGE=16; change TILE design for other pages")
    torch, triton_rt = require_gpu()
    rows = []
    experiments = (["A01", "A02"] if args.lab == "all" else [args.lab])
    variants = {"A01": ["token16", "page16"], "A02": ["page16", "token32"]}
    for context in contexts:
        tensors, expected = setup(torch, context, args.page, args.q_heads,
                                  args.kv_heads, args.dim, args.seed)
        q, k, v, table = tensors
        for lab in experiments:
            for name in variants[lab]:
                if args.variant != "all" and name != args.variant:
                    continue
                row = {"lab": lab, "variant": name, "context": context,
                       "dim": args.dim, "page": args.page,
                       "q_heads": args.q_heads, "kv_heads": args.kv_heads,
                       "online_softmax": True, "kv_layout": "page_table"}
                try:
                    tile = 32 if name == "token32" else args.page
                    scalar = name == "page16"
                    out = torch.full_like(q, float("nan"))
                    def launch():
                        return paged_online_decode[(args.q_heads,)](
                            q, k, v, table, out, context, args.dim,
                            args.q_heads, args.kv_heads, args.page, tile,
                            scalar, 1 / math.sqrt(args.dim), num_warps=4)
                    compiled = launch()
                    row["max_abs_error"] = check(torch, out, expected,
                                                 atol=4e-3, rtol=4e-3)
                    row.update(metadata(compiled, args.dump_asm or None,
                                        f"{lab}_{name}_ctx{context}"))
                    row.update(bench(triton_rt, launch, args))
                    row["status"] = "ok"
                except Exception as exc:
                    row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
                rows.append(row)
    save(rows, args.out, args, torch, triton_rt)
    print_rows(rows)
    if not rows:
        raise SystemExit("No matching variants")
    fail_on_errors(rows)


if __name__ == "__main__":
    main()
