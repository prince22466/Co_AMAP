"""Host-side helpers for Group6 Triton pipelining A/B experiments.

The GPU timer uses event timing through triton.testing.do_bench; compilation,
validation, output copies and allocation are outside the timed callback.
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


def runtime():
    try:
        import torch
        import triton
        import triton.testing  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Requires torch + triton on a compatible GPU (e.g. BI-V150)") from exc
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch.cuda; run on the BI-V150 machine")
    return torch, triton


def positive_ints(value: str, *, name: str, allowed: set[int] | None = None):
    result = []
    for token in value.split(","):
        try:
            item = int(token.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid {name} item: {token!r}") from exc
        if item <= 0 or (allowed is not None and item not in allowed):
            raise ValueError(f"Unsupported {name}: {item}; allowed={sorted(allowed) if allowed else 'positive integers'}")
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError(f"No {name} supplied")
    return result


def kernel_info(handle):
    metadata = getattr(handle, "metadata", None)
    return {
        "registers_per_thread": getattr(handle, "n_regs", ""),
        "spills": getattr(handle, "n_spills", ""),
        "shared_bytes": getattr(metadata, "shared", ""),
        "backend_warp_size": getattr(metadata, "warp_size", ""),
    }


def dump_asm(handle, directory: str | None, name: str):
    if not directory:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for key, value in (getattr(handle, "asm", None) or {}).items():
        if isinstance(value, str):
            (root / f"{name}.{key}.txt").write_text(value, encoding="utf-8")


def paired_bench(launches: dict, trials: int, warmup_ms: int, rep_ms: int):
    """Rotate order each trial to reduce drift. Returns median per variant (us)."""
    import triton.testing
    names = list(launches)
    history = {name: [] for name in names}
    for trial in range(trials):
        order = names[trial % len(names):] + names[:trial % len(names)]
        for name in order:
            latency_ms = float(triton.testing.do_bench(
                launches[name], warmup=warmup_ms, rep=rep_ms))
            if not (math.isfinite(latency_ms) and latency_ms > 0):
                raise RuntimeError(f"Invalid timing for {name}: {latency_ms}")
            history[name].append(latency_ms * 1000)
    return {
        name: {
            "latency_us": statistics.median(values),
            "trial_medians_us": ";".join(f"{v:.3f}" for v in values),
            "trial_mad_us": statistics.median(abs(v - statistics.median(values)) for v in values),
        }
        for name, values in history.items()
    }


def write_csv(destination: str, rows: list[dict]):
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("Nothing to write")
    columns = list(dict.fromkeys(col for row in rows for col in row))
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {target} ({len(rows)} rows)", flush=True)


def write_env(destination: str, torch, triton, args):
    target = Path(destination).with_suffix(".env.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "host": socket.gethostname(),
        "system": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(0),
        "parameters": vars(args),
        "note": "Compiler stage hints are not proof of runtime overlap. Verify generated IR/ISA and device counters.",
    }
    target.write_text(json.dumps(fields, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check_sample(torch, output, reference, sample_indices, *, atol=5e-5, rtol=3e-4):
    observed = output.index_select(0, sample_indices).float()
    expected = torch.as_tensor(reference, device=observed.device, dtype=torch.float32)
    torch.testing.assert_close(observed, expected, rtol=rtol, atol=atol)


def sample_ids(torch, n: int, sample: int = 64):
    # Includes first, middle, and final logical elements (important for masks).
    return torch.linspace(0, n - 1, steps=min(sample, n), device="cuda").long().unique()


def cpu_fma_series(data, work: int):
    """FP64 reference for stream shape (steps, selected_columns)."""
    import numpy as np
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or work < 1:
        raise ValueError("data must be 2-D and work positive")
    acc = np.zeros(values.shape[1], dtype=np.float64)
    for t in range(values.shape[0]):
        v = values[t] * 0.2
        for _ in range(work):
            acc = acc * 0.75 + v
    return acc


def prefetch_schedule(steps: int, distance: int):
    """CPU-only index trace for the manually pipelined ring (educational test).

    fetches is the full set of source indices issued; consumes lists the values
    whose arithmetic is performed. All distances must fetch each step once.
    """
    if steps <= 0 or distance not in (0, 1, 2, 4) or (distance and steps < distance):
        raise ValueError("invalid (steps, distance)")
    if distance == 0:
        return list(range(steps)), list(range(steps))
    window = list(range(distance))
    fetches = window.copy()
    consumes = []
    for t in range(steps):
        consumes.append(window.pop(0))
        if t + distance < steps:
            window.append(t + distance)
            fetches.append(t + distance)
    return fetches, consumes
