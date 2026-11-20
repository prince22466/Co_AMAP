"""Self-contained measurement helpers for Group5 (no model/benchmark changes)."""
from __future__ import annotations

import csv
import json
import math
import platform
import socket
import statistics
import sys
from pathlib import Path


def runtime():
    try:
        import torch
        import triton
        import triton.testing
    except ImportError as exc:
        raise SystemExit("Requires PyTorch + Triton on the target GPU") from exc
    if not torch.cuda.is_available():
        raise SystemExit("No compatible GPU detected through torch.cuda")
    torch.cuda.set_device(0)
    return torch, triton


def parse_int_list(value: str, *, power2: bool = False) -> list[int]:
    out = []
    for raw in value.split(","):
        try:
            n = int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid integer list item: {raw!r}") from exc
        if n <= 0 or (power2 and n & (n - 1)):
            raise ValueError(f"Values must be positive{' powers of two' if power2 else ''}: {n}")
        if n not in out:
            out.append(n)
    if not out:
        raise ValueError("Empty integer list")
    return out


def median_trial_us(fn, trials: int, warmup_ms: int, rep_ms: int) -> list[float]:
    import triton.testing
    times = []
    for _ in range(trials):
        measured = triton.testing.do_bench(
            fn, warmup=warmup_ms, rep=rep_ms, quantiles=[0.5]
        )
        median_ms = float(measured[0] if isinstance(measured, (tuple, list)) else measured)
        if not math.isfinite(median_ms) or median_ms <= 0:
            raise RuntimeError(f"Non-finite/zero timing: {median_ms}")
        times.append(median_ms * 1000.0)
    return times


def measure_pair(a, b, *, trials: int, warmup_ms: int, rep_ms: int):
    """Alternate AB / BA trial order; report median over event timings."""
    import triton.testing
    samples = {"A": [], "B": []}
    functions = {"A": a, "B": b}
    for t in range(trials):
        for name in (("A", "B") if t % 2 == 0 else ("B", "A")):
            result = triton.testing.do_bench(
                functions[name], warmup=warmup_ms, rep=rep_ms,
                quantiles=[0.5]
            )
            latency = float(result[0] if isinstance(result, (list, tuple)) else result) * 1000.0
            if not (math.isfinite(latency) and latency > 0):
                raise RuntimeError(f"Invalid timing for {name}: {latency}")
            samples[name].append(latency)
    return {
        name: {
            "latency_us": statistics.median(samples[name]),
            "trial_medians_us": ";".join(f"{v:.3f}" for v in samples[name]),
        }
        for name in ("A", "B")
    }


def compiled_info(handle) -> dict:
    md = getattr(handle, "metadata", None)
    return {
        "registers_per_thread": getattr(handle, "n_regs", ""),
        "spills": getattr(handle, "n_spills", ""),
        "shared_bytes_compiler": getattr(md, "shared", ""),
        "warp_size_backend": getattr(md, "warp_size", ""),
    }


def dump_asm(handle, directory: str | None, label: str) -> None:
    if not directory:
        return
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    for k, v in getattr(handle, "asm", {}).items():
        if isinstance(v, str):
            (folder / f"{label}.{k}.txt").write_text(v, encoding="utf-8")


def save_csv(path: str, rows: list[dict]):
    if not rows:
        raise RuntimeError("No experiment rows")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows: {target}")


def environment(out: str, torch, triton, args, warp_size: int):
    record = {
        "host": socket.gethostname(),
        "system": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "warp_size_detected": warp_size,
        "arguments": vars(args),
        "notes": "Measured memory traffic and bank conflicts require device counters; compiler metadata is not a hardware-counter measurement.",
    }
    target = Path(out).with_suffix(".env.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def theoretical_gemm_operand_bytes(m: int, n: int, k: int, bm: int, bn: int, input_bytes: int = 2) -> int:
    """Instruction-free lower-level model of logical tile operand requests, excluding cache effects.

    Uses padded tile sizes, as defined in the kernel. Does not model HBM traffic,
    hardware cache hits, or compiler redundant load removal.
    """
    return math.ceil(m / bm) * math.ceil(n / bn) * (bm * k + bn * k) * input_bytes
