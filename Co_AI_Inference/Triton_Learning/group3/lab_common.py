"""Shared utilities for standalone Triton ILP/occupancy microbenchmarks.

Only torch + triton are required on the target GPU. This helper intentionally
uses Triton's event-based do_bench API (not CUDA graphs or official benchmarks).
"""

from __future__ import annotations

import csv
import json
import math
import platform
import socket
import statistics
import sys
from pathlib import Path
from typing import Callable


def require_runtime():
    try:
        import torch
        import triton
        import triton.testing
    except ImportError as exc:
        raise SystemExit(
            "Missing torch/triton. Run in the BI-V150 environment where the "
            "vLLM Triton backend is already installed."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("No compatible GPU visible through torch.cuda.")
    return torch, triton


def parse_powers_of_two(text: str, *, what: str) -> list[int]:
    result = []
    for piece in text.split(","):
        try:
            value = int(piece.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid {what}: {piece!r}") from exc
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{what} must be positive powers of two, got {value}")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError(f"Empty {what}")
    return result


def setup_inputs(torch, n: int, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = 0.20 + 0.60 * torch.rand(n, dtype=torch.float32, generator=generator)
    # Values are deliberately close to 1 to avoid overflow/underflow under long
    # FMA loops, but not exactly 1 (which would make part of the math trivial).
    a = 0.9999 + 0.0002 * torch.rand(n, dtype=torch.float32, generator=generator)
    b = 0.0001 + 0.0002 * torch.rand(n, dtype=torch.float32, generator=generator)
    return (x, a, b), (x.to("cuda"), a.to("cuda"), b.to("cuda"))


def benchmark_ms(fn: Callable[[], object], warmup_ms: int, rep_ms: int, trials: int) -> dict:
    import triton.testing

    medians = []
    for _ in range(trials):
        # do_bench returns milliseconds; quantiles are sorted low -> high.
        q20, median, q80 = triton.testing.do_bench(
            fn, warmup=warmup_ms, rep=rep_ms, quantiles=[0.2, 0.5, 0.8]
        )
        if not all(math.isfinite(float(x)) and float(x) > 0 for x in (q20, median, q80)):
            raise RuntimeError("Invalid GPU event benchmark result")
        medians.append(float(median))
    return {
        "latency_us": statistics.median(medians) * 1000.0,
        "trial_medians_us": ";".join(f"{x * 1000:.3f}" for x in medians),
    }


def compiled_metadata(kernel) -> dict:
    md = getattr(kernel, "metadata", None)
    return {
        "registers_per_thread": getattr(kernel, "n_regs", ""),
        "spills": getattr(kernel, "n_spills", ""),
        "shared_bytes": getattr(md, "shared", ""),
        "backend_warp_size": getattr(md, "warp_size", ""),
    }


def dump_kernel_asm(kernel, folder: str | None, stem: str):
    if not folder:
        return
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    # Different Triton backends expose different IR and machine code keys.
    for key, val in getattr(kernel, "asm", {}).items():
        if isinstance(val, str):
            (root / f"{stem}.{key}.txt").write_text(val, encoding="utf-8")
    # Never assume PTX or a particular vendor ISA is available.


def append_csv(path: str, records: list[dict]):
    if not records:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Each standalone invocation writes a self-consistent table. If rerunning,
    # pass a different path to avoid accidentally mixing incompatible schemas.
    keys = list(dict.fromkeys(key for rec in records for key in rec))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {output} ({len(records)} rows)")


def write_environment(path: str, torch, triton, arguments: dict):
    info = {
        "hostname": socket.gethostname(),
        "system": platform.platform(),
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "unknown"),
        "triton": getattr(triton, "__version__", "unknown"),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "arguments": arguments,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_sample(torch, result, reference, *, atol=6e-4, rtol=4e-3):
    import numpy as np

    expected = np.asarray(reference, dtype=np.float64)
    observed = result[:len(expected)].detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(observed)):
        raise AssertionError("Kernel returned NaN/Inf")
    if not np.allclose(observed, expected, atol=atol, rtol=rtol):
        errors = np.abs(observed - expected)
        k = int(np.argmax(errors))
        raise AssertionError(
            f"Numerical mismatch at sample {k}: got {observed[k]:.8f}, "
            f"expected {expected[k]:.8f}, abs error {errors[k]:.6g}"
        )


def add_common_arguments(parser):
    parser.add_argument("--n", type=int, default=262144, help="Number of FP32 elements")
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--rep-ms", type=int, default=150)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dump-asm", default=None, metavar="DIR",
                        help="Save backend textual assembly/IR if exposed")
    parser.add_argument("--strict", action="store_true",
                        help="Stop at first unsupported configuration (default: record errors)")


def validate_common_args(args):
    if args.n <= 0 or args.trials <= 0 or args.warmup_ms <= 0 or args.rep_ms <= 0:
        raise SystemExit("--n, --trials, --warmup-ms and --rep-ms must all be > 0")
