# Triton compute-efficiency lab (BI-V150 / FlagTree)

This lab complements `triton_data_lab.py` and `profile_triton_data_lab.sh`.
It studies **compute execution**, keeping the same input/output tensors,
shapes, dtypes, source-level global loads/stores, and arithmetic task across
vanilla/candidate variants. No physical GPU kernel is entirely free of data
movement. Actual spills/caches can still vary and must be measured.

## Prepare

Copy `triton_compute_lab.py` and `profile_triton_compute_lab.sh` into the same
folder on BI-V150. Shut down serving processes; use the installed working
PyTorch + Triton/FlagTree environment. No `flag_gems.enable()` is required.

## Run

```bash
cd /root/triton_learning
python3 triton_compute_lab.py --lab C01
python3 triton_compute_lab.py --lab all --csv compute_results.csv | tee compute_results.log
```

The first command checks and benchmarks one A/B pair. The second runs all six.
By default, all input data and output buffers are allocated outside timed
regions; the Triton kernels are warmed before timing. Both variants run in
alternated order over seven samples, using CUDA-event timings. Median and MAD
are reported. Reported *useful* TFLOP/s uses the task's mathematical FLOP count
and is not a hardware counter or a tensor-core utilization claim.

| ID | Vanilla | Candidate | Main lesson |
| --- | --- | --- | --- |
| C01 | multiply then add, `enable_fp_fusion=False` | explicit `tl.fma` | FMA throughput and rounding |
| C02 | four serial recurrence loops | four interleaved recurrence chains | instruction-level parallelism / loop overhead |
| C03 | Horner degree-7 polynomial | Estrin degree-7 polynomial | dependency depth; Estrin uses two additional multiplications |
| C04 | 4 warps | 8 warps | register allocation, occupancy and scheduling |
| C05 | rolled recurrence loop | factor-4 unroll | loop overhead versus instruction/code footprint |
| C06 | `acc += tl.dot(a,b)` | `acc = tl.dot(a,b,acc)` | BF16 matrix pipeline / accumulation |

C01–C05 are FP32. C06 uses BF16 operands with FP32 accumulation and output.
For every A/B pair, inputs and logical global-memory IO are the same. C06
retains the same GEMM tile sizes, program grid, and tile load addresses; the
compilation backend may generate different temporary memory operations.

C02 isolates the presence of independent accumulation chains **imperfectly**:
changing one loop into four also changes loop-control overhead. Check IR to
separate causes. C03 is a mathematical equivalence comparison rather than an
identical-instruction-count comparison. C04 and C05 compare compilation/layout
choices, not universally faster kernels. Candidate performance can regress.

## Test different compute intensities

```bash
python3 triton_compute_lab.py --lab C01 --iters 128
python3 triton_compute_lab.py --lab C01 --iters 1024
python3 triton_compute_lab.py --lab C02 --iters 512
python3 triton_compute_lab.py --lab C06 --gemm-m 1024 --gemm-n 1024 --gemm-k 2048
```

Increasing `--iters` increases arithmetic while preserving external memory
traffic. Most scalar kernels have FP32 arithmetic intensity approximately
`2 * iters / 16` FLOP/byte (C01/C04/C05); C02 has four chains with one-fourth
as many iterations per chain, giving the same useful FLOP/byte. C03 has 7 FMAs
per repetition, equivalent to 14 *useful* FLOPs, and 8 external bytes per item.
The C06 intensity estimate uses compulsory matrix operand/output bytes, not
actual HBM transfers; matrix tiles can be reloaded many times.

## Inspect compiler output

```bash
python3 triton_compute_lab.py --lab C01 --dump-ir-dir compute_ir
python3 triton_compute_lab.py --lab C06 --dump-ir-dir compute_ir
```

Inspect TTIR/TTGIR and backend assembly when exported. Verify that C01
candidate really contains FMAs and that C06's generated matrix instructions
or accumulator handling changed. The compiler may normalize the two versions
into the same machine code: a meaningful finding, not a failed experiment.

## Native BI-V150 hardware profiling

```bash
bash profile_triton_compute_lab.sh C01 C02 C06
```

The script first times each A/B pair **without profiling**; then it records
one warmed kernel invocation per variant with `ixkn-cli` and the CUDA profiler
start/stop API. It captures profiler version, supported metrics and raw CSV.
Change the `METRICS` environment variable to names confirmed by your local
`ixkn-cli --query-metrics` if the default list is unsupported.

Check `matrix_inst`, `issue_stall_reason_pipeline_busy`,
`issue_stall_reason_execution_dependency`, `active_warps_per_active_cycle`,
`stalled_warps_percentage`, and kernel timing. Compute kernel timing still
contains physical loads/stores; interpret only with available counters and
inspection of generated instructions. Do not treat a profiling replay's
reported duration as an application throughput measurement. A `PROFILE_OK`
marker confirms only that the capture region completed.

## Boundary / limitations

- The scripts are standalone **educational diagnostics**, not official
  competition benchmark scripts. They do not change model hyperparameters,
  dispatch policy, FlagGems/vLLM source, or model weights.
- FMA changes rounding, Horner/Estrin use different evaluation orders, and
  dot accumulation can differ slightly; equality uses numeric tolerances.
- Cross-hardware speedups are **not presumed**; numbers should be obtained
  on the actual GPU, and first verify correctness and compiler support.
- Run one case on an idle GPU and compare repeated measurements before
  attributing differences to a specific compute bottleneck.
