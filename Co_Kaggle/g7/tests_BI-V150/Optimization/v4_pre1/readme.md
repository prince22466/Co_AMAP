# v4_pre1 — 16K prefill attention tile refinement (BI-V150)

> **Directory renamed:** this experiment now lives in
> `Optimization/v4_pre1/` on PR #159. The rename preserves the original
> FlagGems and vLLM source tree; the example checkout paths below use
> `/root/Optimization/v4_pre1`. Existing Python test filenames and opt-in
> environment variable names retain their `v4` identifiers.

## Scope and competition compliance

This experiment refines **the existing FlagGems Triton varlen attention kernel**
used by the vLLM FlagGems backend. It changes only files under
`Co_Kaggle/g7/tests_BI-V150/Optimization/v4_pre1/`.

- No quantization, speculative decoding, or model hyperparameter changes.
- No edits to competition benchmark scripts or vLLM operator selection.
- No merging of newer FlagOS branches.
- Does not combine the separate v1, v2, or v3 experiments.

The original Iluvatar large-prefill tile is **BLOCK_M=128, BLOCK_N=32,
num_warps=4**. Default behavior is retained. v4 uses an opt-in experiment
variable rather than claiming a measured 16K speedup.

## Implemented changes

- `FlagGems/src/flag_gems/ops/iluvatar_prefill_tiles.py`: pure tile-selection
  policy for long Q/K (both >=8192), using explicit candidates and an
  experimental shape-dependent `auto` mode.
- `FlagGems/src/flag_gems/ops/flash_api.py`: applies selected BLOCK_M/BLOCK_N
  to the **existing** `flash_varlen_fwd_kernel` in both varlen forward paths
  only for `flag_gems.vendor_name == "iluvatar"`. Keeps original stages,
  warps, kernel code, and dispatcher.
- `FlagGems/tests/test_iluvatar_prefill_tiles_v4.py`: CPU-only policy tests.
- `profile_16k_prefill.py`: standalone GPU-kernel profiling and correctness
  experiment, not a modification to the competition benchmark.

Candidates:

| Mode | Q tile | KV tile | Purpose |
| --- | ---: | ---: | --- |
| `baseline` (default) | 128 | 32 | Control |
| `64x32` | 64 | 32 | More CTAs; reduced Q accumulator |
| `64x64` | 64 | 64 | Larger K/V tiles with modest Q storage |
| `128x32` | 128 | 32 | Explicit control |
| `128x64` | 128 | 64 | More KV reuse per Q tile, fewer loop iterations |
| `128x128` | 128 | 128 | Test register/shared-memory pressure |
| `256x32` | 256 | 32 | More Q reuse; occupancy risk |
| `auto` | shape-dependent | shape-dependent | Unvalidated candidate heuristic |

## BI-V150 results — 16K prefill tile sweep (2026-10-28)

**Observed outcome: no v4 tile candidate beats the existing `128x32` baseline
for this standalone attention test.** The best measured median is **58.6681 ms**.
Keep `FLAGGEMS_ILUVATAR_PREFILL_TILE` **unset** (or use `baseline`);
in particular, do **not** enable the current `auto` heuristic.

### Measured setup and limits

- Source: user-run `profile_16k_prefill.py` CSV output
  (`profiling/v4_prefill_tiles.csv`); the table below records the supplied
  results, not measurements reproduced in CI.
- BI-V150, standalone FlagGems **regular** `mha_varlan_fwd` path;
  16,384 Q/K tokens, batch=1, BF16, 8 query heads, **8 KV heads**,
  head dimension 128, causal attention, default paged KV (page size 16).
  These are the **harness defaults**, not a verified MiniCPM model trace.
- Default warmup=3 and measured iterations=10 per mode; GPU-event timing
  captures the launch envelope and is not an isolated hardware-kernel-cycle
  measurement. Input tensors were reused across modes.
- `correct_vs_baseline` means a BF16 output comparison at
  `rtol=0.03, atol=0.03`; it does **not** mean `math_500` passed.
  Physical DRAM bytes, L2 hits and achieved occupancy were **not measured**.

### Results

Lower median GPU-event latency is better. `speedup` is baseline latency
divided by candidate latency; below 1 means slower than baseline.

| Mode | Q×KV tile | Median (ms) | Speedup | Registers/thread | Spills/thread | Correct vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| **baseline** | **128×32** | **58.6681** | **1.0000×** | **183** | **0** | Yes |
| 64x32 | 64×32 | 64.6954 | 0.9068× | 124 | 0 | Yes |
| 64x64 | 64×64 | 100.9356 | 0.5812× | 151 | 0 | Yes |
| 128x32 | 128×32 | 58.7204 | 0.9991× | 183 | 0 | Yes |
| 128x64 | 128×64 | 115.4063 | 0.5084× | 213 | 0 | Yes |
| 128x128 | 128×128 | 123.4205 | 0.4754× | 256 | 33 | Yes |
| 256x32 | 256×32 | 96.5751 | 0.6075× | 256 | 59 | Yes |
| auto | 64×64 | 101.9687 | 0.5754× | 151 | 0 | Yes |

The explicit `128x32` run is only 0.09% slower than baseline, effectively
a repeat measurement of the same configuration, not a distinct optimization.
The current `auto` choice for this shape resolves to `64x64`; it is
roughly **74% slower** than the baseline, so `auto` should not be promoted.

### Findings and interpretation

1. **Wider K/V tiles regress.** At Q tile 128, increasing KV tile
   `32 → 64` raises the median from 58.6681 to 115.4063 ms; registers rise
   from 183 to 213 and reported shared memory from 49,408 to 66,048 bytes.
   Resource pressure may hurt residency/occupancy, but achieved occupancy
   was not measured, so this is a hypothesis rather than an established cause.
2. **More Q reuse does not guarantee better latency.** Moving from
   `128x32` to `256x32` lowers the *modeled* logical K/V load demand
   from 4,328,521,728 to 2,181,038,080 bytes, yet raises latency to
   96.5751 ms and produces **59 spills/thread**. These logical load estimates
   are **not actual DRAM traffic**.
3. **Smaller Q tiles save registers but increase logical work.**
   `64x32` uses 124 rather than 183 registers/thread and 33,024 instead of
   49,408 shared bytes, but is 10.3% slower; modeled K/V load demand
   increases to 8,623,489,024 bytes.
4. **Numerical results are encouraging but insufficient.** All eight
   rows passed the standalone BF16 comparison, with maximum absolute
   differences at most 0.003906; no `math_500` accuracy conclusion follows.
5. **No throughput improvement is demonstrated.** These are standalone
   attention timings, not the unchanged competition 4K/16K served-throughput
   scores. We have not established that the production server uses this
   FlagGems attention operator for the competition prefill path.

### Relationship to v1 and decision

v1 already investigates **larger query tiles / Q-to-KV reuse**, but in a
*different operator*: the vLLM Triton unified-attention implementation.
Its BI-V150 specialization targets BF16 GQA with **16 query heads / 2 KV
heads / head dimension 128** and has tests for *chunked* prefill
(e.g. 2,048 query tokens with a long existing KV prefix). In contrast,
v4 sweeps `BLOCK_M` and `BLOCK_N` in **FlagGems varlen FlashAttention**;
the current selector requires `max_seqlen_q >= 8192` and
`max_seqlen_k >= 8192`. The numeric tile parameters are not directly
interchangeable between these operator implementations.

**Decision:** Do not continue v4 as a Q/KV tile-only tuning effort or enable
`auto` based on this sweep. Keep v1 and v4 separate. If pursuing v4, first
confirm that the official 16K server actually dispatches to FlagGems
attention (the v4_pre1 plugin otherwise defaults to `TRITON_ATTN`), then work
on **genuine in-kernel** improvements such as paged-KV address-generation
cost, K/V load coalescing, cache behavior or pipeline scheduling. Capture
vendor-profiler HBM/L2 counters, register-spill traffic, achieved occupancy
and kernel latency, and test the *actual MiniCPM GQA dimensions and prefill
chunking*. Final selection still requires the unmodified official 4K/16K
benchmark and `math_500` acceptance.

## v4 follow-up — identify live attention backend and optimize paged-KV lookup

**Implementation status: experimental, opt-in, not yet compiled or timed on
BI-V150.** The preceding 16K tile sweep is completed; this is a separate,
genuine *in-kernel* memory-access change, not another tile selection.

### A. Confirm the attention backend used by the official workload

The v4_pre1 vLLM FlagGems plugin returns
`AttentionBackendEnum.TRITON_ATTN.get_path()` when
`VLLM_FL_USE_FLAGGEMS_ATTN` is unset. Setting
`FLAGGEMS_ILUVATAR_PAGED_KV=grouped` **alone does not cause vLLM to use
FlagGems**. We must first establish the server's actual attention path.

New opt-in diagnostic hooks (disabled by default):

- `FLAGGEMS_V4_TRACE_ATTENTION=1`: the plugin reports
  `V4_ATTN_BACKEND_SELECTED <path>` during attention backend selection.
  The FlagGems varlen kernel launch reports
  `V4_ATTN_KERNEL_EXECUTED ... paged=... tile=... grouped_page_lookup=...`
  after a successful launch; duplicate shape signatures are rate-limited.
- No operator dispatcher logic, vLLM scheduling, or competition benchmark
  script is changed; the diagnostic flags must be **unset for performance
  measurements**.

Start vLLM with the **local v4_pre1 files** and the ordinary competition settings,
without selecting a different attention backend:

```bash
cd /root/Optimization/v4_pre1
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
export FLAGGEMS_V4_TRACE_ATTENTION=1
unset VLLM_FL_USE_FLAGGEMS_ATTN
unset FLAGGEMS_ILUVATAR_PAGED_KV
mkdir -p profiling

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 --served-model-name minicpm \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  2>&1 | tee profiling/v4_backend_default.log
```

From another terminal, run the **unmodified** official 4K/16K benchmark
against port 9031, then inspect the server log:

```bash
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'

grep -E 'V4_ATTN_BACKEND_SELECTED|V4_ATTN_KERNEL_EXECUTED' \
  profiling/v4_backend_default.log
```

Check that the running server really imports the v4_pre1 plugin; absence of both
markers is **inconclusive**, not proof of any particular backend. A
`TRITON_ATTN` selection with no FlagGems kernel event indicates these
FlagGems-specific kernel modifications will not affect that run.

For a **controlled FlagGems comparison only**, restart the server with
`VLLM_FL_USE_FLAGGEMS_ATTN=1` and repeat the official benchmark without
editing it. Verify both the selected-backend marker **and** an executed-kernel
marker before interpreting this as a v4 optimization result. The FlagGems
backend choice is an experimental control, not itself a claim of performance
improvement or a substitution for kernel optimization.

### B. New paged-KV memory-access candidate

The existing paged loader computes the page-table mapping for every token
lane: `page_table[token // page_size]`. In the opt-in
`FLAGGEMS_ILUVATAR_PAGED_KV=grouped` mode, for power-of-two page sizes
dividing `BLOCK_N`, the same kernel instead:

1. Loads **one physical page index per page** in each aligned K/V tile
   (e.g. 2 entries rather than 32 per 32-token tile with 16-token pages).
2. Uses `tl.gather` to expand those page indices to per-token offsets.
3. Keeps K/V pointers, masks, dot-products, numerical operations and the
   existing attention backend dispatch unchanged.

The CPU selector enables the kernel specialization only for **paged** KV,
`max_seqlen_q >= 128`, `max_seqlen_k >= 8192`, aligned power-of-two
page/KV tile geometry, with neither dropout nor local attention. This
includes a 2,048-token query chunk at a 16K context length. It excludes
single-token decode. Default `baseline` keeps the original address path.
The `tl.gather` expansion itself can add communication or register cost;
**fewer page-table instructions do not guarantee reduced HBM traffic or
better latency**.

### C. Validation on BI-V150 (required before enabling)

```bash
cd /root/Optimization/v4_pre1
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"

# CPU-only eligibility, offset-equivalence and selector regressions.
python3 -m pytest --noconftest -q \
  FlagGems/tests/test_iluvatar_prefill_tiles_v4.py

# Real GPU correctness: shuffled physical pages, non-full tail page,
# 16 query heads / 2 KV heads, regular and optimized varlen paths.
python3 -m pytest --noconftest -vv -s \
  FlagGems/tests/test_biv150_grouped_paged_kv_v4.py

# Compare only actual memory-access changes (fixed baseline 128x32 tiles).
python3 profile_16k_prefill.py \
  --modes baseline --kv-modes baseline grouped \
  --seq-len 16384 --query-len 16384 \
  --heads 16 --kv-heads 2 --head-dim 128 \
  --shuffle-pages \
  --output profiling/v4_grouped_full16k.csv

# Repeat for 2K-query / 16K-context chunked prefill.
python3 profile_16k_prefill.py \
  --modes baseline --kv-modes baseline grouped \
  --seq-len 16384 --query-len 2048 \
  --heads 16 --kv-heads 2 --head-dim 128 \
  --shuffle-pages \
  --output profiling/v4_grouped_chunked16k.csv
```

The profiler now reports `kv_mode` and
`grouped_page_lookup`, asserts that the requested compiled specialization
was actually selected, and checks the grouped kernel output against baseline
using identical BF16 inputs. `--shuffle-pages` stresses page-address
correctness rather than assuming identity page mappings. These commands
**have not yet been executed on BI-V150 for this new code**. Treat Triton
compilation errors, output mismatches, spills or latency regressions as
rejection criteria; keep `FLAGGEMS_ILUVATAR_PAGED_KV` unset until device
validation, official throughput comparisons and `math_500` pass.

Next profiler targets: actual page-table/global-memory loads, K/V read
transactions/coalescing, L2 hit rate, achieved occupancy, and stall cycles.
Only after confirming a profitable memory-lookup specialization should we
consider a separate cache/pipeline experiment.

### Run on BI-V150

Use the installed local v4_pre1 FlagGems and v4_pre1 vLLM plugin (do not accidentally
import a separate system FlagGems installation):

```bash
cd /root/Optimization/v4_pre1
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
python3 -m pytest --noconftest -q FlagGems/tests/test_iluvatar_prefill_tiles_v4.py

# Standalone hardware experiment; use the same dtype, head count, and page
# layout as the MiniCPM 16K trace (defaults: BF16, 8 heads, head_dim=128).
python3 profile_16k_prefill.py \
  --seq-len 16384 --heads 8 --kv-heads 8 --head-dim 128 \
  --warmup 3 --iterations 10 --output profiling/v4_prefill_tiles.csv

# Also cover the alternative optimized varlen entry point and unpaged layout.
python3 profile_16k_prefill.py \
  --entrypoint optimized --contiguous-kv \
  --seq-len 16384 --heads 8 --kv-heads 8 --head-dim 128 \
  --output profiling/v4_prefill_opt_contiguous.csv

# No candidate won the initial sweep; keep baseline enabled for now.
unset FLAGGEMS_ILUVATAR_PREFILL_TILE
unset FLAGGEMS_ILUVATAR_PREFILL_REPORT
```

The experiment reports median GPU-event *launch envelopes*, baseline deviation, compiler
register count / spills / shared memory *when exposed by the Triton backend*,
and two analytically modeled K/V quantities: logical per-CTA load demand
and the unique K/V tensor footprint (not a physical DRAM-read lower bound). **Neither estimate
equals physical GPU DRAM traffic.** Cache reuse, memory coalescing, page-table
accesses, and L2 hits affect physical loads. Collect device-memory read bytes,
L2 hit rate, register spill traffic, and **achieved occupancy** separately
using the BI-V150 vendor GPU profiler. The harness will refuse to run if the
FlagGems C-extension attention path is enabled, because that path bypasses the
patched Triton operator. It also verifies actual tile launch reports and
records failed candidate compilation without abandoning the remaining sweep.
A resource-based occupancy *upper
bound* can be calculated only when accurate hardware limits are provided
to the harness via `--registers-per-sm`,
`--shared-bytes-per-sm`, `--threads-per-sm`, and
`--max-ctas-per-sm`; it is not achieved occupancy.

## Promotion gate

1. Confirm all candidate output tensors match baseline within BF16 tolerance;
   reject compilation failures or register spills.
2. Compare **achieved attention kernel latency**, actual HBM/L2 traffic, and
   device occupancy on 16K *uncached* prefill with identical input tensors.
3. Validate any proposed winner against the unmodified competition 4K/16K
   throughput workload. Isolate true prefill from prefix-cache hits.
4. Re-run `math_500` accuracy. Select the fastest candidate that passes the
   accuracy requirement, not simply the smallest tile.
5. Leave `FLAGGEMS_ILUVATAR_PREFILL_TILE` unset in production until the GPU
   results justify adopting the winning configuration.

### Status

Implementation and CPU selector tests are provided. A pre-existing
`torch.emtpty` typo in the optimized non-paged varlen entry point is fixed.
**User-reported BI-V150 standalone GPU-event timings are now documented
above**, and they show no improvement over the baseline for the tested shape.
Physical K/V DRAM traffic, L2/cache metrics, achieved occupancy, the
official 4K/16K served-throughput benchmark and `math_500` remain
unverified. The new opt-in page-grouped memory kernel and backend trace
have source-level review only and still require BI-V150 GPU compilation,
correctness, and performance tests. Keep this PR in draft; it is not a
validated performance gain.
