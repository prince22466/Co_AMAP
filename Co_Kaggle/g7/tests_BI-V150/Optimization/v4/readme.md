# v4 — BI-V150 grouped paged-KV lookup in FlagGems FlashAttention

## Goal and implementation

Optimize MiniCPM5-2B long-context prefill on BI-V150 **inside the existing
FlagGems Triton varlen attention operator**. This is a focused port of the
page-grouped lookup tested in `v4_pre1`, **without importing the unsuccessful
tile-sweep experiment**.

The original `virtual_to_cache_offset` maps each KV token to a physical page
by loading `page_table[token // page_size]`. The experimental kernel instead
loads one page-table entry per *page* within an aligned KV tile and expands
the result to each token using `tl.gather`. K/V tensor loads, causal masks,
softmax and matrix multiplication remain unchanged.

This is **operator-level memory-access optimization**, not a change to kernel
selection. Neither vLLM's backend selection nor the FlagGems `BLOCK_M`,
`BLOCK_N`, warp/stage heuristics are changed. No competition benchmark script,
model hyperparameter, quantization option, speculative sampling setting,
or upstream FlagOS merge is involved.

## Files changed

| Path | Change |
| --- | --- |
| `FlagGems/src/flag_gems/ops/flash_kernel.py` | Adds `virtual_to_cache_offset_grouped` and opt-in specialization to both paged-KV load loops in the **same** `flash_varlen_fwd_kernel` |
| `FlagGems/src/flag_gems/ops/flash_api.py` | Enables specialization at both existing varlen forward launch sites; optional compiled-kernel report |
| `FlagGems/src/flag_gems/ops/iluvatar_grouped_paged_kv.py` | Conservative pure-Python eligibility gate, default disabled |
| `FlagGems/tests/test_iluvatar_grouped_paged_kv_v4.py` | CPU-only gating and randomized physical-page address-equivalence checks |
| `FlagGems/tests/test_biv150_grouped_paged_kv_v4.py` | GPU BF16 GQA correctness: regular/optimized varlen, shuffled physical pages, partial KV tail |
| `profile_grouped_paged_kv.py` | Standalone baseline/grouped GPU-event latency profiler; **not** the official competition benchmark |

## Activation and safety

```bash
# Original behavior (default)
unset FLAGGEMS_ILUVATAR_PAGED_KV
# Explicit control and experimental candidate
export FLAGGEMS_ILUVATAR_PAGED_KV=baseline
export FLAGGEMS_ILUVATAR_PAGED_KV=grouped
```

The `grouped` path is eligible only for **Iluvatar**, paged KV,
`max_seqlen_q >= 128`, `max_seqlen_k >= 8192`, without dropout or local
attention, and when both page size and KV tile size are powers of two with
`BLOCK_N % page_size == 0`. Decode (one query token), short prompts,
contiguous KV, unsupported page layouts, and other vendors retain the
original kernel. **Unset the flag outside controlled testing** until the
official workload and `math_500` gates pass.

For live evidence of which kernel actually executed, use
`FLAGGEMS_V4_GROUPED_REPORT=1` (disabled during timing). Each successful
varlen kernel launch logs `V4_GROUPED_PAGED_KV_KERNEL`, the actual tile,
registers, spills and `grouped_page_lookup=True/False`. A missing marker
means the FlagGems varlen code has **not been confirmed** to execute.

**Important:** vLLM's default `TRITON_ATTN` path does not run this FlagGems
kernel. When measuring **baseline vs grouped**, select the *same* FlagGems
attention backend in both runs via `VLLM_FL_USE_FLAGGEMS_ATTN=1`.
Do not compare `TRITON_ATTN` baseline against a FlagGems grouped run and
attribute the difference solely to page grouping. The framework's default
operator choice remains unchanged by this PR.

## Standalone BI-V150 validation

```bash
cd /root/Optimization/v4
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"

# Confirm the checkout is loaded, not system site-packages.
python3 -c 'import flag_gems; print(flag_gems.__file__)'

# CPU-only safety and address-equivalence tests.
python3 -m pytest --noconftest -q \
  FlagGems/tests/test_iluvatar_grouped_paged_kv_v4.py

# GPU correctness (4 cases, including partial 8,193-token final page).
python3 -m pytest --noconftest -vv -s \
  FlagGems/tests/test_biv150_grouped_paged_kv_v4.py

# Full 16K BF16 GQA prefill (16 query heads, 2 KV heads).
python3 profile_grouped_paged_kv.py \
  --seq-len 16384 --query-len 16384 --heads 16 --kv-heads 2 \
  --head-dim 128 --shuffle-pages --warmup 10 --iterations 30 \
  --output profiling/v4_grouped_full16k.csv

# 2K-query / 16K-context chunked prefill.
python3 profile_grouped_paged_kv.py \
  --seq-len 16384 --query-len 2048 --heads 16 --kv-heads 2 \
  --head-dim 128 --shuffle-pages --warmup 10 --iterations 30 \
  --output profiling/v4_grouped_chunked16k.csv
```

The profiler always runs exactly two modes, **baseline** and **grouped**,
using identical BF16 Q/K/V and shuffled physical page table; it verifies the
executed specialization and refuses tile/warp differences. It checks
finite outputs, maximum absolute error and BF16 `rtol=0.03, atol=0.03`.
CUDA events measure an execution *envelope*, not direct hardware cycle or
physical DRAM counters. Use the BI-V150 vendor profiler for actual page-table
transactions, cache hits, HBM bandwidth and achieved occupancy.

## Historical evidence from v4_pre1 (not yet reproduced under v4)

| Workload | Baseline GPU-event median | Grouped median | Speedup |
| --- | ---: | ---: | ---: |
| 16K query / 16K KV | 114.5281 ms | 97.5166 ms | **1.1744×** |
| 2K query / 16K KV | 27.8980 ms | 23.4170 ms | **1.1914×** |

The **previous `v4_pre1` implementation** passed four BI-V150 GPU correctness
tests (regular/optimized paths, shuffled page mappings, non-full final page)
and had zero register spills. Those observations motivate this port, but
**do not establish that this new `v4` branch has passed the GPU tests**.
The new `v4` branch needs the same device-side verification.

## Official performance and accuracy gates

Start vLLM using the normal competition command with the **same FlagGems
backend for both runs**:

```bash
cd /root/Optimization/v4
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
export VLLM_FL_USE_FLAGGEMS_ATTN=1
export FLAGGEMS_ILUVATAR_PAGED_KV=baseline
# Diagnostic only for a short check (disable for timed benchmark):
export FLAGGEMS_V4_GROUPED_REPORT=1

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 --served-model-name minicpm \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --gpu-memory-utilization 0.85 --max-model-len 131072
```

Confirm the server logs contain `V4_GROUPED_PAGED_KV_KERNEL ... grouped_page_lookup=False`
for long-context prefill. Restart with **only**
`FLAGGEMS_ILUVATAR_PAGED_KV=grouped`, confirm `...=True`, and then
**unset** `FLAGGEMS_V4_GROUPED_REPORT` in both timed comparison runs.

Run the **unmodified** competition benchmark in another terminal:

```bash
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

Compare served throughput, account for prefix-cache hits in the 4K case,
and run the official `math_500` acceptance test before enabling by default.
The official competition result must not be inferred from standalone
attention-kernel speedup. Keep the PR in **Draft** until these checks pass.

## Status

Implementation and tests are proposed. No BI-V150 GPU test or official
benchmark has yet run against this **new v4 branch**. The original
kernel and framework dispatch remain the default unless explicitly enabled.
