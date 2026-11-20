# Group6 — Triton Advanced Pipelining A/B Lab

Target: **Iluvatar BI-V150**, using the existing PyTorch + Triton installation. Place files in:

`Co_AI_Inference/Triton_Learning/group6/`

No changes to vLLM, FlagGems, model configuration, inference hyperparameters, competition benchmark scripts, or official scoring.

## What is covered?

| Experiment | Variants | What it isolates (and what it does not) |
|---|---|---|
| **P01: load/compute ordering** | `serial` loads A, computes, then loads B and computes; `early` issues A and B loads before A computation | Equal mathematical work, elements, and global-memory access set. Tests whether exposing another load before dependent arithmetic helps. **Does not guarantee async execution**, since the compiler may reorder either version. |
| **P02: manual prefetch distance** | `d0`, `d1`, `d2`, `d4`: 0, 1, 2, or 4 iterations of lookahead | Identical stream and logical load count, with a register-resident queue of prefetched values. Distance changes live ranges, registers, instruction dependencies, code size, and potentially compiler placement. |
| **P03: compiler-managed pipeline stages** | `num_stages=1,2,3,4` inside a `tl.range` loop in BF16×BF16→FP32 GEMM | Same GEMM tile and arithmetic, changes **compiler software-pipeline hint**. Some backends may not support every stage count or may lower several settings identically. |
| **P04: pipeline startup/fill/drain** | Sweep GEMM K (`128,512,2048`) × stage count | Contrast short vs long reduction loops. **Only compare stages at the same K**; different K values have different work. |

P01/P02 are source-level software scheduling experiments, not warp specialization or vendor async-copy instruction experiments. P03/P04 request compiler software pipelining through standard Triton syntax. None of these prove that memory and arithmetic physically overlapped without inspection of generated IR/ISA and hardware counters.

## Files

- `triton_pipeline_lab.py` — P01–P04 GPU kernels, reference verification, CSV + metadata.
- `pipeline_lab_common.py` — event timing, order rotation, assembly/IR dumps, CPU numerical reference.
- `profile_triton_pipeline_lab.sh` — one-command smoke/full suite, optional hardware-counter probes.
- `test_pipeline_math.py` — CPU-only unit tests for reference and prefetch queue schedule.
- `README.md` — this documentation.

## First run on BI-V150

```bash
cd Co_AI_Inference/Triton_Learning/group6
python3 test_pipeline_math.py
MODE=smoke bash profile_triton_pipeline_lab.sh
```

The smoke run **compiles + launches + validates** two P01 variants, four P02 distances, two P03 stages and several P04 configurations. No timed benchmark is performed in smoke mode.

## Full A/B measurements

```bash
bash profile_triton_pipeline_lab.sh
```

By default, results land in `results/run_<UTC_TIMESTAMP>/`: `P01.csv` through `P04.csv`, `.log` files, compiler IR/assembly texts if exposed, and environment JSON files.

One experiment at a time:

```bash
python3 triton_pipeline_lab.py --lab P01 --steps 16 --work 8 --out results/P01.csv
python3 triton_pipeline_lab.py --lab P02 --prefetch-distances 0,1,2,4 --out results/P02.csv
python3 triton_pipeline_lab.py --lab P03 --stages 1,2,3,4 --dump-asm results/asm --out results/P03.csv
python3 triton_pipeline_lab.py --lab P04 --k-sizes 128,512,2048 --stages 1,2,3,4 --out results/P04.csv
```

For an experiment more compute-heavy than memory-heavy, increase `--work` (e.g. 2, 8, 32) in P01/P02. The same work count is used by every variant within the run. Do **not** compare raw times across different `--work` settings without treating them as different workloads.

### A single kernel for hardware profiling

```bash
python3 triton_pipeline_lab.py --lab P01 --variant serial --single-run --strict \
    --out results/P01_serial_probe.csv
python3 triton_pipeline_lab.py --lab P01 --variant early --single-run --strict \
    --out results/P01_early_probe.csv
python3 triton_pipeline_lab.py --lab P02 --variant d4 --single-run --strict \
    --out results/P02_d4_probe.csv
python3 triton_pipeline_lab.py --lab P03 --variant k1024_s3 --stages 1,3 \
    --single-run --strict --out results/P03_s3_probe.csv
```

A `--single-run` invocation compiles, executes and validates once, without timing. It makes it easier for a profiler to match a kernel. The default normal timing uses `triton.testing.do_bench` and **rotates variant order across trials** to reduce drift.

Optional `ixkn-cli` probes (only if available in your BI-V150 environment):

```bash
PROFILE=1 bash profile_triton_pipeline_lab.sh
```

The runner saves `ixkn-cli --query-metrics` output and captures selected variants using a default metric list. If the installed profiler uses different metric names, inspect `ixkn_available_metrics.txt` and pass a verified `METRICS=...` list. A successful process is **not proof** that counters were populated: inspect numeric rows and kernel names.

## Interpreting the numbers correctly

1. **P01:** A shorter `early` time can indicate useful load/compute overlap, but may also reflect changed instruction scheduling, spills or register allocation. Generated code may be identical.
2. **P02:** Source prefetch distance increases the lifetime of values. A large distance can be slower if spills or lower occupancy outweigh latency hiding. Confirm that compiler metadata changes as expected.
3. **P03:** `tl.range(..., num_stages=STAGES)` instructs the compiler to consider software pipelining. It is *not* an explicit asynchronous copy mechanism. Unsupported stages should be reported as errors, not silently replaced with a different kernel.
4. **P04:** Long-K matrix multiplication has more opportunities to amortize startup/fill/drain, but the response is hardware/compiler-specific. Compare within each K only.
5. **Logical memory bytes** are not measured HBM traffic. The CSV `logical_requested_GBps` estimates logical source-level read bytes divided by elapsed time. Real transactions depend on cache, overlap, masks and compiler lowering.
6. **Register counts / spills / shared_bytes** are backend compiler metadata, not occupancy or actual cache counters. Hardware occupancy and stall reasons require native profiling.

### Fairness & correctness

- Identical input tensors, shapes, dtypes, arithmetic and *logical* input/output traffic within each fixed-shape comparison.
- P01 and P02 FP32 output checked at sampled positions against a high-precision CPU reference (including final index); P03/P04 entire FP32 GEMM output compared with `torch.matmul` (BF16 inputs).
- Compilation and correctness checks excluded from event timing. Timings are median across configured trials, with per-trial times captured in the CSV.
- `--strict` fails fast; even without it the process exits nonzero for any error. A compilation failure is never counted as a speedup.
- Default: `--elements 262144 --steps 16 --block 256 --work 8`; GEMM `M=N=512 K=1024`, `BM=32 BN=64 BK=32`, `num_warps=4`.
- Experiment kernels intentionally use a generic backend-supported Triton API, not NVIDIA-specific PTX, TMA, `cp.async`, warp-specialization or GPU graph APIs.

## Useful inspections

Inspect `P01_serial.*` vs `P01_early.*`, `P02_d0.*` vs `P02_d4.*`, and `P03_k1024_s1.*` vs `P03_k1024_s3.*` from the generated `asm/` directory. Look for memory-load placement, shared-memory allocation, loop transformation, number of outstanding memory operations, barriers, dot/matrix instructions and spill code. **Do not assume any particular IR key exists** on the Iluvatar backend.

Suggested BI-V150 profiler counters (only if available): kernel duration, memory-dependency stalls, memory throughput, active warps, shared-memory transactions, register/spill-related metrics. Counter names are profiler-version-specific.

## References

- Triton `tl.range` and `num_stages`: https://triton-lang.org/main/python-api/generated/triton.language.range.html
- Triton matrix multiplication: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- Triton `triton.Config` stages: https://triton-lang.org/main/python-api/generated/triton.Config.html
- Triton async-copy tutorial (different, backend-specific mechanism): https://triton-lang.org/main/getting-started/tutorials/gluon/async-copy.html

## Validation performed here

Python syntax (`py_compile`), Bash syntax (`bash -n`), and the pure-CPU unit tests. There is no GPU/Triton runtime in the script-creation environment, so compilation, correctness and timing on BI-V150 **must still be tested on that device**. In particular, multi-stage dot kernels may be unsupported by its backend.
