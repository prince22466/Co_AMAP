# v5 — Decode paged-KV attention optimization (BI-V150)

## Goal and scope

Optimize **single-token autoregressive decode** for MiniCPM5-2B on
Iluvatar BI-V150, particularly the 16K-context throughput workload.

The baseline `v5` directory was copied unchanged from `v0` in PR #163.
This PR changes **only `Optimization/v5/`**.

The standard FlagOS Iluvatar configuration prefers `flagos` for
`attention_backend`, whose default delegates to vLLM's `TRITON_ATTN`.
Therefore editing FlagGems' varlen attention alone would **not** affect
standard serving. Instead, this PR keeps the same vLLM Triton unified
attention backend interfaces, with a plugin-owned copy of the vLLM 0.24.0
kernel and a small kernel-level decode optimization.

## Kernel optimization

In the stock Triton unified-attention paged-KV loop, every KV-token
lane performs the logical mapping:

```text
token position -> KV logical page -> block_table[page] -> physical block
```

For an aligned decode tile, adjacent token positions usually share a page:

- For 3D decode, a 16-token KV tile usually fits in one 16-token KV page.
  The optimized kernel loads one scalar page index.
- For 2D decode, a 32-token KV tile spans two 16-token KV pages.
  The kernel loads two physical page indices and broadcasts them to
  the 32 KV lanes with Triton `tl.gather`.

This lowers *logical page-table/address-generation work* per tile. It
does not eliminate K/V tensor reads or change softmax, attention masking,
numeric precision, scheduling, or KV cache layout. The final partial
page uses a masked page-table load in the grouped path.

The optimization is compiled in **without an enabling flag** for the
following conditions:

- `max_seqlen_q == 1` (only decode, including 2D and segmented 3D decode)
- BF16 Q/K/V, 16 query heads, 2 KV heads, head dimension 128
- `KVQuantMode.NONE` (no quantized KV cache)
- Power-of-two paged KV block size, with the block size dividing the
  current KV tile size or vice versa
- No Triton tensor descriptors (`use_td=False`)

All other cases—including prefill, alternate head counts, and quantized
cache modes—use the **original per-token page-table lookup**. Baseline
`BLOCK_M`, `BLOCK_Q`, `TILE_SIZE`, warps, stages, 2D/3D selection,
operator functionality, and forward attention behavior remain unchanged.
The separate v1 prefill tile tuning is **not copied**.

## Files changed

| File | Purpose |
| --- | --- |
| `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_decode_v5.py` | Existing Triton unified attention kernel plus decode page-index reuse |
| `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_decode_v5.py` | Thin backend adapter preserving vLLM metadata and KV-cache behavior |
| `vllm-plugin-FL/vllm_fl/dispatch/backends/flaggems/flaggems.py` | Return the optimized backend by default on Iluvatar only; preserve existing FlagGems override and non-Iluvatar fallback |
| `vllm-plugin-FL/tests/functional_tests/test_biv150_decode_attention_v5.py` | Default/fallback dispatch and BF16 paged-KV correctness tests |
| `readme.md` | Design, scope, testing instructions, and status |

No changes to competition benchmarks, installed upstream vLLM, FlagGems,
v1-v4 code, quantization settings, or operator-selection policy. The
configured FlagOS backend preferences and all existing fallbacks remain
in place. Only the Iluvatar `TRITON_ATTN` implementation gains an optimized
plugin subclass.

## Installation and testing on BI-V150

The v5 `vllm-plugin-FL` **must be installed** for the new attention
backend to be imported; merely copying the directory is insufficient.
Use the same package-installation procedure as for v0/v1, replacing the
workspace package paths with `Optimization/v5`.

Run syntax and targeted tests from the plugin root:

```bash
cd /root/Optimization/v5/vllm-plugin-FL
python3 -m pytest -q -s --noconftest \
  tests/functional_tests/test_biv150_decode_attention_v5.py
```

Launch the usual server **without any v5 opt-in setting**:

```bash
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

Use the **unchanged competition throughput script**:

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

The `/workspace/vllm-plugin-FL` benchmark location above is the
competition-provided script, **not** a modified v5 benchmark script.
Also run the competition's `math_500` correctness evaluation.

For an apples-to-apples baseline, test v0's original Triton attention
implementation with identical server and benchmark settings and a fresh
server process. Report end-to-end 4K and 16K served-throughput plus
isolated decode kernel timings, not just a standalone kernel speedup.

## Validation status

- Source-level change prepared for review.
- The plugin's functional test suite covers unaligned/shuffled KV pages,
  4K/16K contexts, 2D and 3D decode, and prefill/shape fallback.
- **No BI-V150 GPU test, 4K/16K throughput improvement, or MATH-500
  accuracy result is claimed at PR creation time.**
- Accept only if hardware correctness passes, MATH-500 stays within the
  competition's allowed accuracy range, and throughput improves.

## Competition compliance

No quantization enabling, speculative sampling, benchmark tuning,
unoptimized operator swapping, removal of major backend selection logic,
or upstream FlagOS merge.
