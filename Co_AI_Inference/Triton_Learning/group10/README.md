# Group10 — Triton Irregular Workloads Lab (BI-V150)

Standalone Triton A/B microbenchmarks for **gather/scatter, masks, non-power-of-two lengths and ragged input layouts**. These scripts do **not** modify FlagGems, vLLM, MiniCPM, or any competition scoring benchmark.

## Requirements

- Existing BI-V150 environment with Python 3, `torch`, `triton` (FlagTree-compatible), and `numpy`.
- GPU exposed through `torch.cuda` in that environment, as used in Groups 1–9.
- Optional: `ixkn-cli` for hardware profiling. The GPU backend may not expose all counter names.

**No GPU has been available in the script-authoring environment.** Python syntax, Bash syntax and CPU-reference unit tests have been checked; GPU compilation/performance must be checked on BI-V150.

## Files

| File | Purpose |
|---|---|
| `triton_irregular_lab.py` | I01–I04 Triton A/B kernels, correctness and timing |
| `irregular_reference.py` | CPU-only deterministic generators and independent FP64 references |
| `irregular_lab_common.py` | GPU-event timing, compiler metadata, CPU correctness, CSV/JSON |
| `test_irregular_math.py` | CPU-only reference and indexing regression tests |
| `profile_triton_irregular_lab.sh` | Smoke/full experiments and optional isolated `ixkn-cli` captures |
| `README.md` | This methodology and CLI guide |

## Run

Extract these files together under `Co_AI_Inference/Triton_Learning/group10/` (or `~/triton_learning/group10/`).

```bash
cd ~/triton_learning/group10

# First: CPU indexing/reference tests (works without Triton)
python3 test_irregular_math.py

# Then: short correctness-only compilation test on GPU
MODE=smoke bash profile_triton_irregular_lab.sh

# After smoke passes: A/B GPU timing, IR/assembly and CSV output
bash profile_triton_irregular_lab.sh

# Optional separate hardware counter captures; not used for primary timing
PROFILE=1 bash profile_triton_irregular_lab.sh
```

Run just one experiment:

```bash
python3 triton_irregular_lab.py --lab I01 --out results/I01.csv
python3 triton_irregular_lab.py --lab I02 --out results/I02.csv
python3 triton_irregular_lab.py --lab I03 --out results/I03.csv
python3 triton_irregular_lab.py --lab I04 --out results/I04.csv
```

Single-variant capture example:

```bash
python3 triton_irregular_lab.py --lab I02 --variant sorted \
  --n 65536 --single-run --strict --dump-asm results/I02_asm \
  --out results/I02_sorted.csv
```

With no GPU, running the Triton entry point will clearly report that its runtime prerequisites are missing.

## Experimental design

### I01 — Gather ordering

- **A (`direct`)**: each output position reads `source[index[position]]` and writes output positions sequentially.
- **B (`sorted`)**: indices are sorted on the CPU **before** timing. A kernel reads the same source indices in sorted order and scatters each result back to its original output position.
- Patterns: `uniform` indices across the source, `hotset` indices concentrated within at most 4096 source elements.
- Both produce the **same final output**, and issue the same count of useful source data loads; the sorted path has **extra index reads** and changes **output store locality**.
- Sorting, index preprocessing, allocation, and transfer to the GPU are **excluded** from kernel timings. If each request needs a fresh sort, this microbenchmark may exaggerate the practical value of B. Do not interpret an advantage as solely improved load coalescing.

### I02 — Unique-destination scatter

- **A (`direct`)**: write `values[i]` to `output[destination[i]]` in source order.
- **B (`sorted`)**: sort unique destinations on the CPU, gather `values[source_position]` and write targets in sorted order.
- `random` is a full permutation; `block_local` permutes targets only within 256-element regions.
- No atomics are used because all destinations are **unique**; this lab does not measure collision resolution. B adds a source index lookup and changes input read locality, so it is not a pure write-coalescing measurement. The host sort is excluded.

### I03 — Non-power-of-two tail handling

- **A (`masked`)**: one grid, masked load/store in each CTA.
- **B (`tail_split`)**: all complete blocks use unmasked loads/stores; the final partial block uses one extra, masked kernel when needed.
- Both use the **same allocation and math**, and read/write each logical element exactly once. B pays for up to **two launches**, which are included in the GPU-event timing; this is intentional. Compiler inspection may show that the compiler already eliminates much of the masking overhead in A.
- Default sizes: `4093,65537,262147`; default blocks `128,256,512`.

### I04 — Packed versus padded ragged reductions

- **A (`padded`)**: `[rows, max_len]` input representation, one CTA sums the valid prefix of each row.
- **B (`packed`)**: values concatenated in a single array with row offsets, one CTA sums each ragged segment.
- Each input value participates in the same FP32 row sum. Both kernels mask invalid lanes. The packed layout adds offset reads but has a smaller input allocation and potentially improved locality. Conversion to padded/packed layout and GPU input transfer are excluded from kernel timings.
- The generated data contains empty rows, a full-length row, a length-one row and sampled variable-length rows. CPU references use FP64 summation, then cast to FP32.

## Result interpretation

Every successful variant has a CSV row with `latency_us` (GPU-event median), `trial_times_us`, `max_abs_error`, compiler-provided register/spill/shared-memory/warp metadata where available, and `status`. A/B speedup is **baseline latency / candidate latency**; values > 1 favor the candidate. Each family has a different named baseline:

| Lab | Baseline | Candidate |
|---|---|---|
| I01 | `direct` | `sorted` |
| I02 | `direct` | `sorted` |
| I03 | `masked` | `tail_split` |
| I04 | `padded` | `packed` |

GPU compilation, CPU/GPU transfer, host-side preprocessing, and validation are outside the timed region. `triton.testing.do_bench` uses GPU event-based timing; a configuration that performs two launches includes both launches in the timed callback. It is not an end-to-end Python dispatch overhead study.

**Important cautions**:

1. Comparing sorted/unsorted data changes several resource effects at once (index reads, store locality, cache reuse). Inspect IR and hardware counters before attributing an observed speedup to one specific mechanism.
2. Cache state is warmed by the benchmark harness and is **not** explicitly flushed. These are primarily warm-cache measurements, not isolated HBM bandwidth tests.
3. Do not infer bank conflicts, occupancy, cache misses, or memory transactions solely from kernel time. Check `ixkn-cli` numeric counters and kernel coverage where supported.
4. Actual arithmetic for I04 is FP32, with tree-reduction order chosen by the compiler. Numerical tolerances permit normal FP32 reduction differences, not incorrect outputs.
5. A GPU result may depend on batch size, working-set size and device architecture. These experiments do not imply a full-model tokens/s improvement.

## Suggested further experiments

```bash
# Gather locality and index volume
python3 triton_irregular_lab.py --lab I01 --n 1048576 \
  --source-size 4194304 --gather-patterns uniform,hotset \
  --out results/I01_large.csv

# Scatter locality, including whole-array and within-block permutations
python3 triton_irregular_lab.py --lab I02 --n 1048576 \
  --scatter-patterns random,block_local --out results/I02_large.csv

# Tail behavior on very small and moderate data sizes
python3 triton_irregular_lab.py --lab I03 \
  --elements 127,255,257,4093,65537 --blocks 128,256,512 \
  --out results/I03_tails.csv

# Raggedness with non-power-of-two maximum row length
python3 triton_irregular_lab.py --lab I04 --rows 1024 --max-len 1023 \
  --fill-rates 0.1,0.5,0.9 --out results/I04_ragged.csv
```

The optional `PROFILE=1` runner uses `ixkn-cli` with the `METRICS` environment variable (default `time_duration,llc_hit_rate,mem_load_transactions_per_request`). Confirm these counters are supported by the local profiler and that actual numeric rows were captured. The completion `PASS` message confirms the kernel ran, not that profiling succeeded.
