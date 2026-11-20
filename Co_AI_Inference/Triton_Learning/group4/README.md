# Group4 — Triton Memory Latency & Cache Lab (BI-V150)

These standalone A/B microbenchmarks continue `Co_AI_Inference/Triton_Learning/group1`–`group3`. They study **GPU memory hierarchy, memory dependencies and latency hiding** without changing vLLM, FlagGems, MiniCPM, or any competition benchmark/parameters. No quantization, speculative decoding, or benchmark-script tuning.

The folder is self-contained. Run on BI-V150 with your existing Python/PyTorch/Triton installation (`torch.cuda.is_available()` must report `True`). There are no machine-specific kernel launch choices beyond ordinary CLI sweep parameters.

## Files

- `triton_memory_latency_lab.py`: four experiments, correctness checks, event timings, compiler metadata, CSV output
- `memory_lab_common.py`: parsing, CPU references, interleaved benchmarking, CSV, environment metadata
- `profile_triton_memory_lab.sh`: smoke/full run; optional `ixkn-cli` counters
- `test_memory_lab_math.py`: CPU-only unit test of index arithmetic/references

## Run order

```bash
cd /root/Triton_Learning/group4   # change to the actual location

# Can run on a CPU-only development machine:
python3 test_memory_lab_math.py

# First GPU compilation + correctness + very short timing:
MODE=smoke bash profile_triton_memory_lab.sh

# Full GPU experiments:
bash profile_triton_memory_lab.sh
```

The runner creates `results/run_YYYYMMDD_HHMMSS/` containing:

- `memory_latency.csv`: each configuration, latency in µs, logical read rate, register/spill metadata, status
- `memory_latency.env.json`: device and test arguments
- `asm/`: textual compiler IR and ISA *where this Triton backend exposes them*
- `bench.log`: stdout/stderr and any compilation/runtime errors

To isolate one experiment:

```bash
python3 triton_memory_latency_lab.py --lab M01 --out results/m01.csv
python3 triton_memory_latency_lab.py --lab M02 --out results/m02.csv
python3 triton_memory_latency_lab.py --lab M03 --out results/m03.csv
python3 triton_memory_latency_lab.py --lab M04 --out results/m04.csv
```

### M01 — Working-set size / cache residency

**A/B:** same compiled load pattern, data source and total logical read operations; vary addressable working set (0.25, 2, 16, 64 MiB by default). The kernel repeatedly reads FP32 entries into an accumulator and writes one FP32 output per lane. Indexes change each iteration, and `volatile=True` prevents load elimination by the compiler.

**Metrics:** latency, logical read GB/s, configured working set and upper bound on distinct bytes touched. The stream accesses wrap modulo the working-set size. Each invocation is repeatedly timed after warm-up, so this is a *steady-state working-set comparison*, **not** a direct L1/L2/HBM classification, not a controlled cold-cache benchmark and not measured device-DRAM bandwidth. Cache sizes and hit rates require independent hardware-counter validation.

```bash
python3 triton_memory_latency_lab.py --lab M01 \
  --cache-n 262144 --cache-reps 64 \
  --cache-sizes-mib 0.25,2,16,64 --block 256 \
  --out results/m01.csv
```

### M02 — Serial memory dependency vs four independent chains

**A/B:** random-permutation pointer table, one chain of 128 dependent loads versus four independent chains of 32 dependent loads each. Both issue **128 logical int32 loads per output element**. All chains have the same random table and data source; each variant has its own CPU reference.

**Important:** the variants do *not* calculate identical output values. They have equal logical load counts but different starting indices, dependency depth, reduction work, per-program resource demand and memory-request parallelism. Therefore the ratio measures a practical latency-hiding trade-off, **not** isolated DRAM latency or pure MLP.

```bash
python3 triton_memory_latency_lab.py --lab M02 \
  --chase-n 65536 --pool-entries 1048576 --total-loads 128 \
  --block 256 --out results/m02.csv
```

### M03 — Memory-level parallelism (independent-chain count)

Sweeps 1, 2, 4, 8, and 16 independent pointer chains per output with constant total loads. Each individual chain remains serial, but more chains can overlap. This extends M02 from an A/B pair into a curve and intentionally repeats its 1- and 4-chain endpoints to allow separately reproducible comparisons.

```bash
python3 triton_memory_latency_lab.py --lab M03 \
  --chains 1,2,4,8,16 --total-loads 128 \
  --out results/m03.csv
```

Interpret with occupancy, spills, emitted load instructions and cache counters. Increasing chains also increases register and reduction cost.

### M04 — Lane locality × Triton block size

**A/B:** strides 1, 17 and 257 between adjacent lane addresses, crossed with blocks 128, 256 and 512. Total logical reads and working-set allocation remain fixed. This characterizes **memory coalescing / cache-line utilization and block-resource interactions**, not isolated temporal-cache locality. Higher strides also change spatial footprint and transaction count.

```bash
python3 triton_memory_latency_lab.py --lab M04 \
  --cache-size-mib 64 --cache-n 262144 --cache-reps 64 \
  --locality-strides 1,17,257 --blocks 128,256,512 \
  --out results/m04.csv
```

## Optional native GPU profiling

If `ixkn-cli` and its BI-V150 counter plugin are installed:

```bash
PROFILE=1 bash profile_triton_memory_lab.sh
```

This **first completes the normal timed run**, then profiles six single-launch probe configurations: M01 (small/large), M02 (1/4 chains), and M04 (stride 1/257 at block 128). The script saves counter logs and available metric names. Check that reported hardware counters contain **actual numeric rows** and that the intended kernel was captured; `PROFILE_OK` only confirms completion and correctness, not a successful counter capture.

You can override the metrics accepted by your exact profiler build:

```bash
PROFILE=1 METRICS=time_duration,llc_hit_rate,mem_load_throughput \
  bash profile_triton_memory_lab.sh
```

For a single native variant, use `--native --lab M02 --native-index 1`; this runs the four-chain kernel once, validates outputs, and prints `PROFILE_OK`. It intentionally does not run a timing loop.

## How to interpret CSV

- `latency_us`: median of per-trial medians reported by event-based Triton `do_bench`.
- `logical_read_bytes`: explicit int32/FP32 load requests (4 bytes/request), not bytes transacted at HBM.
- `logical_read_gbps`: `logical_read_bytes / latency`; cache hits and scattered memory transactions can make this dramatically different from measured HBM GB/s.
- `ws_mib`: configured addressable working-set size for stream labs. `touched_mib_upper_bound`: maximum distinct bytes for M01 (may be smaller than allocation).
- `registers_per_thread`, `spills`, `shared_bytes`, `backend_warp_size`: available compiler metadata, **not measured occupancy**.
- `status` and `error`: correctness, compilation, and timing success/failure.

## Fairness / pitfalls

1. No compilation, setup, numpy reference, or correctness checks are included in timed kernel runs.
2. All variants within a given experiment keep output count and total logical load count constant; M02/M03 vary chain count and thereby change live state, reduction, and dependency structure.
3. All test variants are compiled and validated before timing. The timing sequence is reversed on alternating trials to reduce fixed-order bias, but GPU clocks, concurrent processes and residual cache state still affect results.
4. No explicit cache flush occurs between timed runs. Do not describe M01 variants as definitively "L1", "L2", "HBM", "hot", or "cold" without counters and a cold-cache-specific experiment.
5. `volatile=True` constrains compiler load removal, not a guaranteed cache policy. Verify the emitted ISA if surprising results occur.
6. The random-permutation pointer table is created on the host and copied to GPU before timing. The table remains read-only during tests.
7. For tiny kernels, event timing/launch overhead can dominate; increase `--cache-n` or `--chase-n` and check latency stability across trials.
8. These tests do not make any claims about MiniCPM total tokens/s or `math_500` accuracy. A production kernel change must be separately tested in the actual inference workload.

## Suggested experiment sequence

First `M01`, then `M02`, then `M03`, then `M04`. After the initial measurements, collect `ixkn-cli` memory dependency stalls, cache hits, load transactions and active warp counters (where supported) for configurations with large differences. Cross-reference those counters with compiler output before deciding whether the bottleneck is latency, cache miss behavior, coalescing, register pressure or dispatch.
