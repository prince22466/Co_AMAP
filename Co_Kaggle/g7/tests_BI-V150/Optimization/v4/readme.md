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
| `readme.md` | Documents the implementation and official validation procedure |

No standalone profiling script or benchmark-script change is included.

## Activation and eligibility

By default, **the original kernel is used**:

```bash
unset FLAGGEMS_ILUVATAR_PAGED_KV
# Equivalent explicit baseline:
export FLAGGEMS_ILUVATAR_PAGED_KV=baseline
```

To enable the experimental specialization:

```bash
export FLAGGEMS_ILUVATAR_PAGED_KV=grouped
```

The grouped path is selected only when all of these conditions hold:

- Device vendor is `iluvatar`, and the active operator is FlagGems varlen attention.
- KV cache is **paged** (not contiguous).
- `max_seqlen_q >= 128` and `max_seqlen_k >= 8192`.
- No dropout and no local/sliding-window attention.
- KV page size and `BLOCK_N` are powers of two, with page size <= `BLOCK_N`
  and `BLOCK_N` divisible by page size.

For ineligible shapes—including single-token decode—the original page
lookup remains in use. The **4K workload may not activate this optimization**
because of the 8K KV-length threshold; 16K full or chunked prefill is the
primary target.

**Important:** the existing vLLM plugin normally selects `TRITON_ATTN`.
This PR does not change that selection. Setting
`FLAGGEMS_ILUVATAR_PAGED_KV=grouped` has no effect unless the serving
path actually runs the FlagGems varlen attention kernel.

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

Run the **unchanged competition benchmark** against the v4 server. To
isolate the page-lookup change, keep the model, server settings, request
workload, and **attention backend identical** between the two runs.

If explicitly evaluating the FlagGems attention backend, use the same
`VLLM_FL_USE_FLAGGEMS_ATTN=1` setting in *both* runs. Switching from
default `TRITON_ATTN` to FlagGems between runs would invalidate an
attribution of any speedup to grouped page lookup.

In the vLLM server environment, run baseline first:

```bash
cd /root/Optimization/v4
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
export VLLM_FL_USE_FLAGGEMS_ATTN=1
export FLAGGEMS_ILUVATAR_PAGED_KV=baseline

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

From another terminal, run the **unaltered** official throughput test:

```bash
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

Restart the same server with only
`FLAGGEMS_ILUVATAR_PAGED_KV=grouped` changed, and repeat the **same**
official benchmark. Record the 4K and 16K throughput results separately;
account for prefix-cache hits or other run-to-run workload differences.
Verify the server actually used FlagGems attention before attributing
any result to this kernel change.

Finally, run the competition's **`math_500` accuracy test**. Promotion
requires an allowed accuracy result and an actual improvement on the
official throughput workload.

## Competition compliance and status

- No quantization, speculative sampling, or model-hyperparameter changes.
- No benchmark-script modification and no replacement of the principal
  operator-selection logic.
- No direct upstream FlagOS merge.
- No v1/v2/v3/v4_preplay changes.

**Status:** grouped paged-KV is implemented in v4, opt-in and disabled by
default. Official 4K/16K throughput and `math_500` results for this PR
are pending. Do not claim an end-to-end throughput improvement yet.
