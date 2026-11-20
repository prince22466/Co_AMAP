# Group11 — Triton Compiler Optimization Lab (BI-V150)

Learn how **source transformations become compiled instructions** on the installed Triton backend, rather than assuming a source-code optimization necessarily changes hardware execution. This directory is standalone; it does not modify vLLM/FlagGems or the competition benchmarks.

**Requirements:** Python 3.10+, NumPy, PyTorch and Triton 3.x on a supported GPU. Developed for the BI-V150 environment previously used with Groups 1–10; GPU execution has **not** been validated in the creation environment.

## Start

```bash
cd /root/triton_learning/group11
python3 test_compiler_math.py
MODE=smoke bash profile_triton_compiler_lab.sh
bash profile_triton_compiler_lab.sh
```

Run a specific experiment:

```bash
python3 triton_compiler_lab.py --lab K01 --unrolls 1,2,4,8 --out results/K01.csv
python3 triton_compiler_lab.py --lab K02 --tiles 16x64,32x32,64x16 --out results/K02.csv
python3 triton_compiler_lab.py --lab K03 --out results/K03.csv
python3 triton_compiler_lab.py --lab K04 --out results/K04.csv
```

Each experiment checks the GPU output against a **CPU-computed** reference before measuring latency. CPU references avoid the device-side FP64 issue identified in Group8. Compilation, initialization, reference computation and artifact export are excluded from GPU-event timing. Output buffers are seeded with NaNs prior to correctness checks to reveal missing writes. Run timing on an otherwise idle GPU.

## Experiments

| ID | A/B comparison | What is actually controlled | Interpretation caveat |
|---|---|---|---|
| **K01** | `loop_unroll_factor=1,2,4,8` | Compiler loop unrolling hint for identical FMA computations | Backend may ignore the hint, or choose the same machine code |
| **K02** | Transpose tile configurations: 16×64, 32×32, 64×16 | Core Triton tile shape; compiler selects layout automatically | Tile dimensions also change locality and the CTA grid; **not** direct explicit layout control |
| **K03** | Load eight input streams before vs after an FMA loop | Source-level live-range placement with equal logical loads, FLOPs, and outputs | Compiler may hoist/sink loads and erase differences; unchanged code is a valid result |
| **K04** | Loop invariant code motion enabled vs disabled | `tl.range(..., disable_licm=...)` applied to an invariant `tl.exp()` | Backend may ignore/change scheduling; compare code stages |
| **K02G (optional)** | Gluon `BlockedLayout` `size_per_thread=1,2,4` | **Explicit** per-thread layout with detected device warp size | Requires `triton.experimental.gluon` and supported backend; not substituted silently |

K01/K04 use independent reference calculations based on FP64 **on CPU**, converted to FP32 for comparison (numerical differences are expected for FMA and exp). K02 is an exact FP32 transpose. K03 performs equal logical loads and arithmetic across variants and sums in the same source order. None of the lab sweeps alters model accuracy settings or the official benchmark.

### Optional explicit layouts with Gluon

```bash
RUN_GLUON=1 MODE=smoke bash profile_triton_compiler_lab.sh
RUN_GLUON=1 bash profile_triton_compiler_lab.sh
# or explicitly
python3 triton_compiler_lab.py --lab K02G --vector-widths 1,2,4 \
  --dump-asm results/K02G_asm --out results/K02G.csv
```

The hardware warp width is read from device properties or previously compiled-kernel metadata, rather than assuming NVIDIA's 32 lanes. On BI-V150, previous Group3 compiled metadata reported warp width 64, but this lab re-checks it rather than hardcoding it. An unsupported compiler extension produces an explicit error/unsupported result.

## Generated IR and assembly

Default compiler dump directory: `results/asm/`. For each A/B variant the lab writes any backend-provided stages such as `ttir`, `ttgir`, `llir`, vendor-specific textual ISA, and binary artifacts (if exposed), plus `<variant>.stages.json` with SHA-256, byte counts and limited textual heuristics. **A hash of TTIR or TTGIR is not a proof of a different machine instruction stream.** When comparing variants, prefer final compiled binaries/ISA if available; if neither is exposed, report that the final code difference is unknown. Do not interpret raw text occurrence counts as dynamic instructions, stalls, or achieved occupancy.

Compiler metadata in CSV includes `registers_per_thread`, `spills`, `shared_bytes`, and `backend_warp_size` when the backend provides them. `latency_us` is the median across independently repeated GPU-event timings, not compilation time. A/B comparisons require the same matrix shape, buffer placement, warm-cache policy and GPU clock conditions. Timing order is fixed within each lab; for small claimed gains, rerun with reversed order or a new run to check drift. Results are not full model tokens/s improvements.

## Hardware profiler

```bash
PROFILE=1 bash profile_triton_compiler_lab.sh
# If needed supply a known supported ixkn-cli metric list:
METRICS=time_duration PROFILE=1 bash profile_triton_compiler_lab.sh
```

The optional `ixkn-cli` captures are single-run correctness launches and separate from unprofiled timings. Check `available_metrics.txt` before adding metrics and verify actual rows were captured. A profiler query may itself need adjustment for a particular version of the vendor tool.

## Typical questions to answer

1. Did K01 produce fewer loop branches or different instruction scheduling without increasing spills?
2. Did the K02 tile shape change `convert_layout`, shared-memory use, or code generation?
3. Did the compiler shrink the K03 early/late live-range difference into identical code?
4. Does disabling LICM in K04 force repeated `exp` instructions, or does the backend still hoist it?
5. Do K02G explicit layouts change contiguous memory transactions, register pressure, and effective latency?

## Known boundaries

- The installed BI-V150 Triton backend may not support `loop_unroll_factor`, `disable_licm` or Gluon `BlockedLayout`; preserve failures instead of pretending a fallback measured the same thing.
- Group11 isolates compiler-behavior examples; it does not identify the dominant production MiniCPM kernel bottleneck on its own.
- K03 tests *source-level live ranges*, not pure register allocation. Measure registers/spills from compilation and inspect generated code.
- Core Triton does not offer guaranteed explicit per-lane layouts, hence optional Gluon K02G.
- Device-local floating point and backend lowering can change results; correctness must pass before any timing data is trusted.

### Compare two compiler-generated variants

```bash
python3 compare_compiler_ir.py --dir results/asm \
  --a K01_unroll1 --b K01_unroll4 --out results/K01_unroll_diff.txt
# Or choose a particular stage exposed on your device:
python3 compare_compiler_ir.py --dir results/asm \
  --a K04_licm_enabled --b K04_licm_disabled --stage ttgir
```

If you used the combined runner, point `--dir` to the timestamped `results/run_*/asm/K01` (or `K04`) directory instead. The diff utility operates on saved artifacts and requires no GPU.
