#!/usr/bin/env python3
"""Group13: shared memory, layouts, synchronization and real async copy.

H00: core Triton transpose tile lowering (backend-managed shared memory).
H01: opt-in Gluon explicit shared allocation, store/barrier/load, swizzle A/B.
H02: synchronous core-Triton copy vs NVIDIA Ampere+ Gluon cp.async (opt-in).
H03: NVIDIA Ampere+ explicit 1-buffer vs 2-buffer cp.async pipeline (opt-in).

No H00 code path is described as explicit async; unsupported H01/H02 remain
UNSUPPORTED, never silently replaced with a different operator.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_lab_common import (base_parser, bench, check, fail_on_errors, metadata,
                            print_rows, require_gpu, save, validate_args)
try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise SystemExit("Requires Triton") from exc

try:
    from triton.experimental import gluon as g
    from triton.experimental.gluon import language as gl
    GLUON_IMPORT_ERROR = ""
except (ImportError, AttributeError) as exc:
    g = gl = None
    GLUON_IMPORT_ERROR = str(exc)

try:
    from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
except (ImportError, AttributeError):
    cp = None


@triton.jit
def transpose_tiled(X, Y, M: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    r = tl.program_id(0) * TILE + tl.arange(0, TILE)
    c = tl.program_id(1) * TILE + tl.arange(0, TILE)
    x = tl.load(X + r[:, None] * N + c[None, :],
                (r[:, None] < M) & (c[None, :] < N), 0)
    xt = tl.trans(x)
    tl.store(Y + c[:, None] * M + r[None, :], xt,
             (c[:, None] < N) & (r[None, :] < M))


@triton.jit
def copy_sync(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, mask=i < N, other=0)
    tl.store(Y + i, x, mask=i < N)


if g is not None:
    @g.jit
    def shared_layout_copy(X, Y, M: gl.constexpr, N: gl.constexpr,
                           TILE: gl.constexpr, LANES: gl.constexpr,
                           SWIZZLED: gl.constexpr):
        in_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, LANES],
                                                  [4, 1], [1, 0])
        out_layout: gl.constexpr = gl.BlockedLayout([1, 1], [LANES, 1],
                                                   [1, 4], [0, 1])
        r = gl.program_id(0)*TILE + gl.arange(0, TILE, gl.SliceLayout(1, in_layout))
        c = gl.program_id(1)*TILE + gl.arange(0, TILE, gl.SliceLayout(0, in_layout))
        value = gl.load(X + r[:, None]*N + c[None, :],
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
        converted = smem.load(out_layout)
        rr = gl.program_id(0)*TILE + gl.arange(0, TILE, gl.SliceLayout(1, out_layout))
        cc = gl.program_id(1)*TILE + gl.arange(0, TILE, gl.SliceLayout(0, out_layout))
        gl.store(Y + rr[:, None]*N + cc[None, :], converted,
                 mask=(rr[:, None] < M) & (cc[None, :] < N))
else:
    shared_layout_copy = None


if g is not None and cp is not None:
    @g.jit
    def copy_cp_async(X, Y, N: gl.constexpr, BLOCK: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        i = gl.program_id(0)*BLOCK + gl.arange(0, BLOCK, layout=layout)
        smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[0])
        smem = gl.allocate_shared_memory(gl.float32, [BLOCK], layout=smem_layout)
        cp.async_load(smem, X+i, mask=i < N)
        cp.commit_group()
        cp.wait_group(0)
        val = smem.load(layout)
        gl.store(Y+i, val, mask=i < N)
else:
    copy_cp_async = None



if g is not None and cp is not None:
    @g.jit
    def copy_add_pipeline(A, B, C, COLS: gl.constexpr,
                          BLOCK: gl.constexpr, BUFFERS: gl.constexpr):
        """True explicitly issued async-copy pipeline with 1 or 2 SMEM buffers.

        Preload -> steady-state issue/wait/read -> drain; no barrier spin loop.
        Only compiled after an NVIDIA Ampere capability check in main().
        """
        row = gl.program_id(0)
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        idx = gl.arange(0, BLOCK, layout=layout)
        smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[1, 0])
        sa = gl.allocate_shared_memory(gl.float32, [BUFFERS, BLOCK], layout=smem_layout)
        sb = gl.allocate_shared_memory(gl.float32, [BUFFERS, BLOCK], layout=smem_layout)
        base = row * COLS
        chunks: gl.constexpr = gl.cdiv(COLS, BLOCK)
        # Pipeline fill.
        for pre in gl.static_range(BUFFERS - 1):
            pos = pre*BLOCK + idx
            cp.async_load(sa.index(pre), A+base+pos, mask=pos<COLS)
            cp.async_load(sb.index(pre), B+base+pos, mask=pos<COLS)
            cp.commit_group()
        # Issue next copy while computing from the oldest ready buffer.
        for i in range(chunks - (BUFFERS - 1)):
            load_idx = i + BUFFERS - 1
            pos = load_idx*BLOCK + idx
            cp.async_load(sa.index(load_idx % BUFFERS), A+base+pos,
                          mask=pos<COLS)
            cp.async_load(sb.index(load_idx % BUFFERS), B+base+pos,
                          mask=pos<COLS)
            cp.commit_group()
            cp.wait_group(BUFFERS - 1)
            old_pos = i*BLOCK + idx
            av = sa.index(i % BUFFERS).load(layout)
            bv = sb.index(i % BUFFERS).load(layout)
            gl.store(C+base+old_pos, av+bv, mask=old_pos<COLS)
        # Drain earlier issued copies, if any.
        for j in gl.static_range(BUFFERS - 1):
            old = chunks - (BUFFERS - 1) + j
            cp.wait_group(BUFFERS - 2 - j)
            old_pos = old*BLOCK + idx
            av = sa.index(old % BUFFERS).load(layout)
            bv = sb.index(old % BUFFERS).load(layout)
            gl.store(C+base+old_pos, av+bv, mask=old_pos<COLS)
else:
    copy_add_pipeline = None


def cuda_async_supported(torch):
    """Never enable an NVIDIA-specific instruction on a generic CUDA-like facade."""
    try:
        target = triton.runtime.driver.active.get_current_target()
        name = torch.cuda.get_device_name(0).lower()
        cc = torch.cuda.get_device_capability(0)
        # Explicit user opt-in is also required. Exclude known non-NVIDIA devices.
        return (target.backend == "cuda" and cc[0] >= 8
                and not any(s in name for s in ("iluvatar", "bi-v", "ix", "amd", "radeon")))
    except Exception:
        return False


def main():
    p = base_parser(__doc__, lab_choices=["H00", "H01", "H02", "H03"])
    p.add_argument("--n", type=int, default=262144)
    p.add_argument("--matrix-size", type=int, default=512)
    p.add_argument("--pipe-cols", type=int, default=1025)
    p.add_argument("--tile", type=int, default=64)
    p.add_argument("--enable-nvidia-async", action="store_true",
                   help="Opt in to NVIDIA Ampere cp.async (never runs on BI-V150)")
    p.add_argument("--require-gluon", action="store_true",
                   help="Treat unavailable H01 Gluon implementation as a failure")
    p.add_argument("--require-nvidia-async", action="store_true",
                   help="Treat unavailable H02 cp.async implementation as a failure")
    p.add_argument("--variant", default="all",
                   choices=["all", "tile16", "tile32", "plain", "swizzled", "sync", "async",
                            "buffer1", "buffer2"])
    args = p.parse_args()
    validate_args(args)
    if args.pipe_cols < 128:
        p.error("pipe-cols must be >= 128 for the 2-buffer pipeline")
    if args.n <= 0 or args.matrix_size <= 0:
        p.error("n and matrix-size must be positive")
    if args.tile < 16 or args.tile & (args.tile-1):
        p.error("tile must be a power of two >= 16")
    torch, triton_rt = require_gpu()
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    n = args.n
    m = args.matrix_size
    x = torch.randn((m, m), generator=rng)
    dx = x.cuda()
    v = torch.randn(n, generator=rng)
    dv = v.cuda()
    labs = ["H00", "H01", "H02", "H03"] if args.lab == "all" else [args.lab]
    rows = []
    warp_size = int(getattr(torch.cuda.get_device_properties(0), "warp_size", 0) or 0)
    if warp_size not in (32, 64):
        warp_size = 64 if "bi-v" in torch.cuda.get_device_name(0).lower() else 32

    def run(lab, variant, launch, out, expected, *, require_shared=False):
        row = {"lab": lab, "variant": variant, "status": "ok"}
        try:
            out.fill_(float("nan"))
            compiled = launch()
            row["max_abs_error"] = check(torch, out, expected, atol=0, rtol=0)
            row.update(metadata(compiled, args.dump_asm or None, f"{lab}_{variant}"))
            if require_shared and not int(row.get("shared_bytes") or 0):
                raise RuntimeError("No compiler-confirmed shared allocation")
            row.update(bench(triton_rt, launch, args))
        except Exception as exc:
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)

    def unavailable(lab, variant, reason, strict=False):
        rows.append({"lab": lab, "variant": variant,
                     "status": "error" if strict else "unsupported", "reason": reason})

    if "H00" in labs:
        for tile in (16, 32):
            name = f"tile{tile}"
            if args.variant not in ("all", name):
                continue
            y = torch.empty((m, m), device="cuda")
            def launch(tile=tile, y=y):
                return transpose_tiled[(triton.cdiv(m, tile), triton.cdiv(m, tile))](
                    dx, y, m, m, tile, num_warps=4)
            run("H00", name, launch, y, x.T.contiguous())

    if "H01" in labs:
        for swizzled, name in ((False, "plain"), (True, "swizzled")):
            if args.variant not in ("all", name):
                continue
            if shared_layout_copy is None:
                unavailable("H01", name, "Gluon unavailable: " + GLUON_IMPORT_ERROR,
                            args.require_gluon)
                continue
            y = torch.empty((m, m), device="cuda")
            def launch(swizzled=swizzled, y=y):
                return shared_layout_copy[(triton.cdiv(m, args.tile),
                                           triton.cdiv(m, args.tile))](
                    dx, y, m, m, args.tile, warp_size, swizzled, num_warps=4)
            # A compiler that cannot lower Gluon is an unsupported backend,
            # not proof that the benchmark has a fallback implementation.
            run("H01", name, launch, y, x, require_shared=True)
            if rows[-1]["status"] == "error" and not args.require_gluon:
                msg = rows[-1].get("reason", "").lower()
                if any(key in msg for key in ("not supported", "not compiled",
                                               "unsupported", "not implemented")):
                    rows[-1]["status"] = "unsupported"

    if "H02" in labs:
        for name in ("sync", "async"):
            if args.variant not in ("all", name):
                continue
            if name == "async" and (not args.enable_nvidia_async or
                                   copy_cp_async is None or not cuda_async_supported(torch)):
                unavailable("H02", name, "NVIDIA Ampere+ Gluon cp.async not enabled/supported",
                            args.require_nvidia_async)
                continue
            y = torch.empty(n, device="cuda")
            if name == "async":
                def launch(y=y):
                    return copy_cp_async[(triton.cdiv(n, 256),)](
                        dv, y, n, 256, num_warps=4)
            else:
                def launch(y=y):
                    return copy_sync[(triton.cdiv(n, 256),)](
                        dv, y, n, 256, num_warps=4)
            run("H02", name, launch, y, v, require_shared=(name == "async"))
            if name == "async" and rows[-1]["status"] == "error" and not args.require_nvidia_async:
                msg = rows[-1].get("reason", "").lower()
                if any(key in msg for key in ("not supported", "not compiled",
                                               "unsupported", "not implemented")):
                    rows[-1]["status"] = "unsupported"

    if "H03" in labs:
        rows_count, cols = 32, args.pipe_cols
        acpu = torch.randn((rows_count, cols), generator=rng)
        bcpu = torch.randn((rows_count, cols), generator=rng)
        a, b = acpu.cuda(), bcpu.cuda()
        for buffers, name in ((1, "buffer1"), (2, "buffer2")):
            if args.variant not in ("all", name):
                continue
            if (not args.enable_nvidia_async or copy_add_pipeline is None
                    or not cuda_async_supported(torch)):
                unavailable("H03", name,
                            "Explicit Gluon NVIDIA Ampere+ pipeline unavailable",
                            args.require_nvidia_async)
                continue
            y = torch.empty_like(a)
            def launch(y=y, buffers=buffers):
                return copy_add_pipeline[(rows_count,)](
                    a, b, y, cols, 64, buffers, num_warps=4)
            run("H03", name, launch, y, acpu + bcpu, require_shared=True)
            if rows[-1]["status"] == "error" and not args.require_nvidia_async:
                msg = rows[-1].get("reason", "").lower()
                if any(key in msg for key in ("not supported", "not compiled",
                                               "unsupported", "not implemented")):
                    rows[-1]["status"] = "unsupported"

    save(rows, args.out, args, torch, triton_rt)
    print_rows(rows)
    if not rows:
        raise SystemExit("No matching variants")
    fail_on_errors(rows)


if __name__ == "__main__":
    main()
