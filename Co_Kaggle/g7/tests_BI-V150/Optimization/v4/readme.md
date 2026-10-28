# v4 — Grouped paged-KV lookup for BI-V150 attention

## Objective

Improve MiniCPM5-2B **long-context inference throughput**, especially the
competition's 16K workload, by reducing redundant page-table reads **inside
the existing FlagGems Triton variable-length attention kernel**.

This is a focused port of the grouped paged-KV lookup implementation already
tested in `Optimization/v4_preplay`. It does **not** include the earlier
query/KV tile-size sweep.

## What changed

The original paged-KV attention loader maps each KV token to its physical
cache page:

```text
virtual token -> page_table[token_index // page_size] -> physical page
              -> physical page stride + in-page token offset
```

Several adjacent tokens belong to the same page, so the original loader
performs a page-table lookup for each token lane. With the new
`GROUPED_PAGE_LOOKUP` kernel specialization, an aligned KV tile:

1. Loads **one physical page index per page** in that tile.
2. Expands those page indices to token lanes using Triton `tl.gather`.
3. Uses the resulting offsets for the **existing** K and V tensor loads.

For example, a 32-token KV tile with 16-token pages requires two logical
page-table loads instead of 32 token-lane lookups. This reduces page-index
lookup work; it does **not** imply 16x less actual DRAM traffic.

The attention computation, causal masking, softmax, K/V tensor values,
`BLOCK_M` / `BLOCK_N`, num_warps, num_stages, and attention backend
selection are unchanged. Both the masked tail loop and the main KV loop
use the same opt-in implementation.

## Modified files

| File | Purpose |
| --- | --- |
| `FlagGems/src/flag_gems/ops/flash_kernel.py` | Implements `virtual_to_cache_offset_grouped`; adds `GROUPED_PAGE_LOOKUP` to the existing `flash_varlen_fwd_kernel` paged-KV load paths |
| `FlagGems/src/flag_gems/ops/flash_api.py` | Passes the opt-in kernel specialization at both existing varlen attention launch sites; no tile or backend selection changes |
| `FlagGems/src/flag_gems/ops/iluvatar_grouped_paged_kv.py` | Defines the safe eligibility conditions and reads the environment flag |
| `vllm-plugin-FL/vllm_fl/dispatch/backends/flaggems/flaggems.py` | Uses optimized FlagGems attention by default on Iluvatar while preserving existing dispatch/fallback and override mechanisms |
| `readme.md` | Documents automatic activation and official validation |

No standalone profiling script or benchmark-script change is included.

## Built-in activation and eligibility

When the **v4 version of vllm-plugin-FL and FlagGems is installed and loaded**
on BI-V150, standard `vllm serve` automatically selects the optimized
FlagGems attention backend on Iluvatar via the existing FlagOS dispatcher.
No `VLLM_FL_USE_FLAGGEMS_ATTN=1` or
`FLAGGEMS_ILUVATAR_PAGED_KV=grouped` setting is required.

Within that backend, grouped page-table lookup is **automatically compiled**
for eligible paged-KV prefill; the baseline lookup remains for other shapes:

- Device vendor is `iluvatar`; running the FlagGems Triton varlen operator.
- KV cache is **paged**, not contiguous.
- `max_seqlen_q >= 128` and `max_seqlen_k >= 8192`.
- No dropout and no local/sliding-window attention.
- KV page size and `BLOCK_N` are powers of two; page size <= `BLOCK_N`
  and `BLOCK_N` is divisible by the page size.

For ineligible shapes—including single-token decode—the original page
lookup remains in use. The **4K workload is normally unaffected by the
grouped kernel**, although selecting FlagGems rather than the previously
default `TRITON_ATTN` still changes its serving backend. This makes the
official **4K and 16K** measurements and accuracy check mandatory.

The automatic backend choice is implemented only in the existing
`FlagGemsBackend.attention_backend()` hook of vllm-plugin-FL, which keeps
the existing FlagOS dispatch policy, vendor fallback, and attention-backend
overrides. If the operator-selection policy is explicitly configured to
prefer the Iluvatar vendor backend, it can still select `TRITON_ATTN`;
this patch does not override that policy.

For controlled baseline comparisons, the existing overrides remain
available:

```bash
# Optional diagnostic A/B control only (not required for normal serving)
export FLAGGEMS_ILUVATAR_PAGED_KV=baseline

# Optional rollback to previous default attention backend
export VLLM_FL_USE_FLAGGEMS_ATTN=0
```

`FLAGGEMS_ILUVATAR_PAGED_KV` defaults to `grouped` on all eligible shapes.
A fresh server process is required when changing either setting.

## Previous evidence (from v4_preplay, not yet new v4 results)

The same grouped kernel code was previously tested on BI-V150 using BF16,
16 query heads, 2 KV heads, head dimension 128, and paged KV:

| Attention workload | Baseline GPU-event median | Grouped GPU-event median | Speedup |
| --- | ---: | ---: | ---: |
| 16K query / 16K KV | 114.5281 ms | 97.5166 ms | **1.1744x** |
| 2K query / 16K KV | 27.8980 ms | 23.4170 ms | **1.1914x** |

Four GPU correctness checks passed in the prior `v4_preplay` experiment
(regular and optimized varlen paths, shuffled physical pages, and a partial
KV page). **These are standalone kernel results, not measured official
4K/16K serving-throughput gains.** The newly ported v4 integration has
not yet been validated in the official workload.

## Official 4K/16K evaluation procedure

**Use the normal vLLM serve command** with this version of the FlagOS plugin
installed/loaded. Do not add an optimization-enabling environment variable
to your production/competition launch:

```bash
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

Run the **unaltered competition throughput benchmark** from another terminal:

```bash
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

For a causal A/B of the **grouped lookup alone**, restart the server with
`FLAGGEMS_ILUVATAR_PAGED_KV=baseline` and re-run this identical benchmark.
Keep the same FlagGems backend in both runs; do not switch back to
`TRITON_ATTN` for the kernel-level control. Measure the default baseline
vLLM/TRITON server **separately** to evaluate whether changing the default
attention backend helps total competition throughput.

Before adopting this automatic default, also verify the server is actually
using the FlagGems **Triton** varlen implementation (not the optional
FlagGems C-extension path), then run the official `math_500` accuracy test.
Pay attention to 4K prefix-cache hits and to overall 4K throughput; the
grouped kernel targets long-context prefill, but the default backend change
may affect other cases.

## Competition compliance and status

- No quantization, speculative sampling, or model-hyperparameter changes.
- No benchmark-script modification or deletion of the principal
  operator-selection logic. The existing FlagGems backend is now the
  automatic Iluvatar preference because it includes a real optimized kernel;
  this backend choice still requires end-to-end validation.
- No direct upstream FlagOS merge.
- No v1/v2/v3/v4_preplay changes.

**Status:** built-in activation implemented on eligible BI-V150 workloads,
without a special enable flag. The automatic FlagGems backend choice and
new v4 integration have not yet passed the official 4K/16K throughput or
`math_500` accuracy checks. Keep this PR in Draft and do not claim an
end-to-end throughput improvement yet.
