"""Shared, benchmark-independent helpers for Triton Learning groups 12–14.

Never used by competition benchmarks or by the inference serving process.
GPU event timings are microbenchmark measurements, not model tokens/second.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import platform
import socket
import statistics
import sys
from pathlib import Path


def require_gpu():
    try:
        import torch
        import triton
        import triton.testing
    except ImportError as exc:
        raise SystemExit("Requires PyTorch and Triton in the target GPU environment") from exc
    if not torch.cuda.is_available():
        raise SystemExit("No GPU available via torch.cuda")
    torch.cuda.set_device(0)
    return torch, triton


def ints(text, *, min_value=1):
    try:
        values = list(dict.fromkeys(int(x.strip()) for x in text.split(",")))
    except ValueError as exc:
        raise ValueError("Expected comma-separated integers") from exc
    if not values or any(x < min_value for x in values):
        raise ValueError("Integers must be >= " + str(min_value))
    return values


def check(torch, observed_gpu, expected_cpu, *, atol=2e-3, rtol=2e-3):
    assert expected_cpu.device.type == "cpu", "Independent reference must be on CPU"
    torch.cuda.synchronize()
    actual = observed_gpu.detach().cpu()
    if actual.shape != expected_cpu.shape:
        raise AssertionError(f"shape: {tuple(actual.shape)} != {tuple(expected_cpu.shape)}")
    if not bool(torch.isfinite(actual).all()):
        raise AssertionError("NaN/Inf output (possibly unwritten data)")
    diff = (actual.double() - expected_cpu.double()).abs()
    maximum = float(diff.max().item()) if diff.numel() else 0.0
    if not torch.allclose(actual, expected_cpu, atol=atol, rtol=rtol):
        raise AssertionError(f"CPU-reference mismatch: max_abs_error={maximum:.8g}")
    return maximum


def bench(triton, fn, args):
    """Time the complete callable, including required resets and extra launches."""
    if args.single_run:
        return {"latency_us": "", "trial_medians_us": "", "trial_mad_us": ""}
    samples = []
    for _ in range(args.trials):
        timing = triton.testing.do_bench(fn, warmup=args.warmup_ms, rep=args.rep_ms,
                                         quantiles=[0.2, 0.5, 0.8])
        v = float(timing[1]) * 1000
        if not math.isfinite(v) or v <= 0:
            raise RuntimeError(f"Invalid GPU-event duration: {v}")
        samples.append(v)
    return {
        "latency_us": round(statistics.median(samples), 6),
        "trial_medians_us": ";".join(f"{v:.3f}" for v in samples),
        "trial_mad_us": round(statistics.median(abs(v - statistics.median(samples)) for v in samples), 6),
    }


def metadata(kernel, outdir=None, label=None):
    """Save exposed compiler artifacts; do not infer dynamic hardware counters."""
    md = getattr(kernel, "metadata", None)
    out = {
        "registers_per_thread": getattr(kernel, "n_regs", None),
        "spills": getattr(kernel, "n_spills", None),
        "shared_bytes": getattr(md, "shared", None),
        "backend_warp_size": getattr(md, "warp_size", None),
    }
    stages = {}
    for stage, obj in (getattr(kernel, "asm", None) or {}).items():
        if isinstance(obj, str):
            data, suffix = obj.encode("utf8"), "txt"
        elif isinstance(obj, (bytes, bytearray)):
            data, suffix = bytes(obj), "bin"
        else:
            continue
        stages[str(stage)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if outdir and label:
            p = Path(outdir)
            p.mkdir(parents=True, exist_ok=True)
            safe_stage = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(stage))
            (p / f"{label}.{safe_stage}.{suffix}").write_bytes(data)
    if outdir and label:
        (Path(outdir) / f"{label}.stages.json").write_text(json.dumps(stages, indent=2) + "\n")
    out["compiler_stages"] = json.dumps(stages, sort_keys=True)
    return out


def save(rows, out, args, torch=None, triton=None):
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    info = {"args": vars(args), "host": socket.gethostname(),
            "system": platform.platform(), "python": sys.version.split()[0]}
    if torch is not None:
        info.update({"torch": torch.__version__, "triton": triton.__version__,
                     "gpu": torch.cuda.get_device_name(0)})
    path.with_suffix(".env.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    print(f"Saved {len(rows)} records: {path}")


def base_parser(description, *, lab_choices):
    import argparse
    p = argparse.ArgumentParser(description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lab", choices=["all", *lab_choices], default="all")
    p.add_argument("--out", default="results/lab.csv")
    p.add_argument("--dump-asm", default="")
    p.add_argument("--single-run", action="store_true",
                   help="One correctness-checked launch, no timed benchmarking")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup-ms", type=int, default=25)
    p.add_argument("--rep-ms", type=int, default=60)
    p.add_argument("--seed", type=int, default=1234)
    return p


def validate_args(args):
    if args.trials < 1 or args.warmup_ms < 1 or args.rep_ms < 1:
        raise SystemExit("Trials, warmup-ms, and rep-ms must be >= 1")


def print_rows(rows):
    for row in rows:
        print(" ".join(f"{k}={v}" for k, v in row.items()
                       if k in ("lab", "variant", "context", "status",
                                "latency_us", "max_abs_error", "reason")))


def fail_on_errors(rows):
    errors = [r for r in rows if r.get("status") == "error"]
    if errors:
        raise SystemExit(f"{len(errors)} failed experiment(s); see CSV for details")
