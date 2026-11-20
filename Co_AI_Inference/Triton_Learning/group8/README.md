# Group8 — Triton reductions and softmax A/B lab

Standalone microbenchmarks for **BI-V150 / Triton 3.6**, using only the installed PyTorch and Triton GPU runtime. This package does **not** modify MiniCPM, vLLM, FlagGems, hyperparameters, or competition benchmark scripts. Results are exploratory microbenchmark measurements, **not** inference throughput improvements.

## Experiments

| ID | A (baseline) | B (candidate) | Measures | Interpretation limitation |
|---|---|---|---|---|
| **R01** | A: dependent serial scan per row, vectorized across independent rows | B: vectorized `tl.sum` tree | Serial dependency + parallel reduction benefit | Changes coalescing, occupancy, and effective thread utilization too; **not** arithmetic-reduction-only |
| **R02** | A: flat `tl.sum` reduction | B: two-level chunk reduction (`tl.sum` of per-chunk `tl.sum`) | Logical local and cross-chunk reduction structure | A `CHUNK=64` is **not guaranteed** to map to one BI-V150 warp; backend may compile both similarly |
| **R03** | A: three launches (max → sum exp → normalize) | B: one fused stable softmax | Global intermediate traffic and kernel launches vs fusion | Includes launch overhead and additional reads, not only reduction time |
| **R04** | A: single full-row fused softmax | B: streaming online `(max, sum)` stats, then reread input and normalize | Online reduction on long rows, live state vs extra loads | **Not** a one-pass output softmax or FlashAttention; online needs two passes for all probabilities |

All variants use contiguous FP32 matrices, row-major `[rows, cols]`. All outputs are checked against an independent **CPU FP64 PyTorch reference**, copied to the GPU as FP32 after calculation. R01/R02 compare against the CPU FP64 row sum converted to FP32; R03/R04 compare against CPU FP64 softmax converted to FP32, with input values containing occasional large ±12 values. Tails are masked for non-power-of-two column counts. Input/output addresses are the same within an A/B pair; the implementations are intentionally different.

## Run

Put these six files together under `Co_AI_Inference/Triton_Learning/group8/`.

```bash
# CPU-only mathematical checks
python3 test_reduction_math.py

# GPU compilation + correctness for all four labs, including masked tails
MODE=smoke bash profile_triton_reduction_lab.sh

# Full, GPU-event-timed experiments
bash profile_triton_reduction_lab.sh
```

Run individual labs and save their compiler output:

```bash
python3 triton_reduction_lab.py --lab R01 --out results/R01.csv
python3 triton_reduction_lab.py --lab R02 --chunks 32,64,128 --warps 4,8 --out results/R02.csv
python3 triton_reduction_lab.py --lab R03 --out results/R03.csv
python3 triton_reduction_lab.py --lab R04 --stream-chunks 128,256,512 --out results/R04.csv
python3 triton_reduction_lab.py --lab R04 --rows 128 --cols 8192 \
  --dump-asm results/R04_asm --out results/R04_long.csv
```

CLI shape options `--rows` and `--cols` override the per-lab defaults (0 means use defaults):

- R01: **128 × 256** (serial row scan is intentionally slow; 64 independent rows per CTA)
- R02: **512 × 1024**
- R03: **512 × 1024**
- R04: **128 × 4096**

`--single-run` compiles and checks correctness without timing. `--variant` selects a single variant for independent GPU profiling (requires `--lab`). `--strict` exits on any unsupported or invalid configuration. Unsupported GPU kernels, compilation problems, and numerical mismatches are **failures**, not silently skipped optimizations.

## Optional BI-V150 profiling

```bash
PROFILE=1 bash profile_triton_reduction_lab.sh
# Check the supported metric list before using advanced counters:
METRICS='time_duration,active_warps_per_active_cycle' PROFILE=1 \
  bash profile_triton_reduction_lab.sh
```

Hardware profiling is optional and only uses `ixkn-cli` if installed. The runner saves the tool's supported metric list, captures kernel-specific output for R02/R04, and verifies a functional-completion marker. Inspect the **numeric hardware counter rows** and names yourself: no counter validity or warp-level attribution is inferred from the marker alone. `PROFILE=1` records separate instrumented runs; do not compare their execution times with uninstrumented CSV results.

## Measurement and analysis

- Results are written into timestamped `results/run_*/`, with per-lab CSV/logs, environment JSON, and backend-generated textual IR/assembly if exposed.
- `latency_us` is the median of repeated `triton.testing.do_bench` GPU-event timings. A/B order alternates between trials. Compilation and warmup are outside timed measurements.
- `candidate_speedup = baseline_latency / candidate_latency`; greater than 1 indicates a faster candidate.
- `logical_traffic_gbps` is a model based on *requested global bytes divided by elapsed time*. It is **not measured HBM bandwidth**, and R03 staged includes modeled scratch reads/writes.
- `registers_per_thread`, `spills`, `shared_bytes`, and `backend_warp_size` come from compiled kernel metadata when the backend supplies them. For R03 staged, semicolon-separated values correspond to the three kernels.
- Neither an explicit logical chunk in R02 nor `num_warps` alone proves that the reduction was intra-warp. Inspect compiler layouts, lowered instructions, barriers/shared-memory usage, and BI-V150 counters where available.
- Online softmax R04 first scans chunks to compute stable global normalizers, then reloads input for probabilities. It trades input traffic for a smaller live working set and may be slower on short rows.
- Differences in row length, warp count, input cache residency, and compiler lowering can dominate outcomes. For reliable conclusions, rerun successful configurations and verify the actual assembly.

## Suggested analysis

1. **R01:** See how serial dependency and strided lane-wise loads compare with a vectorized tree reduction. Serial uses one independent sequential chain per row, processing up to 64 rows per CTA; tree uses one CTA per row. Differences in CTA count, active lanes, load coalescing, and occupancy are part of this comparison, so do not attribute the entire speedup to `tl.sum` alone.
2. **R02:** Compare chunk 32/64/128 at identical rows/cols/warp count. Look for compiled instructions and changes in shared memory or register pressure.
3. **R03:** Separate fusion benefits from the fact the staged method runs three kernels and rereads X three times.
4. **R04:** Sweep 1024/4096/8192/16384 columns and online chunk lengths to find whether lower live register pressure outweighs extra reads.

GPU compilation and performance must be validated on the BI-V150 machine. CPU tests do not establish backend compatibility.

## BI-V150 R01 correctness fix (2026-11-20)

The original `reduce_serial_kernel` used a **scalar-only** `tl.load` +
scalar accumulation inside one CTA per row. In the initial reported BI-V150 / Triton 3.6 run, every output failed validation. A later run exposed the actual GPU outputs and zero-valued references; recreating the reported seeded inputs on CPU matched the revised serial kernel outputs. Thus the initial failure **does not establish a scalar-kernel bug**, and the original scalar kernel needs separate validation before assigning blame to its lowering.

The revised serial kernel runs `SERIAL_ROWS_PER_CTA=64` independent row-wise
serial chains using vector-valued loads and stores. Within each row, addition
remains sequential: it does **not** introduce `tl.sum`. Output buffers are
initialized to NaN before initial validation to expose missing writes.

Check the updated kernel on the actual GPU **before** timing:

```bash
python3 triton_reduction_lab.py --lab R01 --rows 17 --cols 137 \
  --variant serial --single-run --strict --dump-asm results/r01_asm \
  --out results/r01_serial_fix_smoke.csv
MODE=smoke bash profile_triton_reduction_lab.sh
bash profile_triton_reduction_lab.sh
```

Both results from any previous `reduce_serial_kernel` version and any old
R01 latency comparisons should be discarded. CPU-only tests cannot guarantee
correct GPU compiler lowering; if the revised version fails, share the new
CSV, error output, and dumped IR rather than disabling validation or loosening
tolerances.

## R01/R02 reference correctness fix (2026-11-20, second correction)

The subsequent BI-V150 diagnostic showed the revised serial kernel producing
correct-looking nonzero sums, while the GPU-based `x.double().sum(dim=1)`
reference was **all zeros**. Re-creating the exact 17x137 seeded data on CPU
reproduced the serial kernel's first eight outputs (within FP32 tolerance).
This implicates device-side FP64 reference calculation, not the reduction
kernel. We have **not** isolated whether the failing device operation is FP64
conversion or the FP64 reduction.

`cpu_reference(torch, x, lab)` now transfers input FP32 to the CPU **before**
converting to FP64 and computing the reference. R01/R02 sum on CPU FP64;
R03/R04 compute CPU FP64 softmax. The reference is transferred to the input
device in FP32 only for comparison; all of this happens outside timing.
There is no correctness-tolerance relaxation or change to benchmark workloads.

Run the reported regression test then the BI-V150 smoke suite:

```bash
python3 test_reduction_math.py
python3 triton_reduction_lab.py --lab R01 --variant serial \
  --rows 17 --cols 137 --single-run --strict \
  --out results/r01_ref_fixed.csv
MODE=smoke bash profile_triton_reduction_lab.sh
```

The revised serial kernel remains in this package; it should not be described
as a proven fix for the original failure. Its performance cannot be compared
directly with the original scalar-CTA version because work distribution changed.
