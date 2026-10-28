# v4 — 16K prefill attention tile refinement (BI-V150)

## Scope and competition compliance

This experiment refines **the existing FlagGems Triton varlen attention kernel**
used by the vLLM FlagGems backend. It changes only files under
`Co_Kaggle/g7/tests_BI-V150/Optimization/v4/`.

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

### Run on BI-V150

Use the installed local v4 FlagGems and v4 vLLM plugin (do not accidentally
import a separate system FlagGems installation):

```bash
cd /root/Optimization/v4
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
python3 -m pytest -q FlagGems/tests/test_iluvatar_prefill_tiles_v4.py

# Standalone hardware experiment; use the same dtype, head count, and page
# layout as the MiniCPM 16K trace (defaults: BF16, 8 heads, head_dim=128).
python3 profile_16k_prefill.py \
  --seq-len 16384 --heads 8 --kv-heads 8 --head-dim 128 \
  --warmup 3 --iterations 10 --output profiling/v4_prefill_tiles.csv

# Re-run only a confirmed winner. Baseline is always used when unset.
export FLAGGEMS_ILUVATAR_PREFILL_TILE=128x64
export FLAGGEMS_ILUVATAR_PREFILL_REPORT=1
```

The experiment reports median GPU-event time, baseline deviation, compiler
register count / spills / shared memory *when exposed by the Triton backend*,
and two analytically modeled K/V quantities: logical per-CTA load demand
and the lower bound of one unique K/V read per KV head. **Neither estimate
equals physical GPU DRAM traffic.** Cache reuse, memory coalescing, page-table
accesses, and L2 hits affect physical loads. Collect device-memory read bytes,
L2 hit rate, register spill traffic, and **achieved occupancy** separately
using the BI-V150 vendor GPU profiler. A resource-based occupancy *upper
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

Implementation and CPU selector tests are provided. **GPU timing, physical K/V
load counters, achieved occupancy, and math_500 results have not been measured
in this PR environment.** This is a candidate experiment for BI-V150 testing,
not a validated performance claim.
