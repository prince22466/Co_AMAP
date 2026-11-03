# v8 — Integration of MiniCPM5-2B BI-V150 optimizations v1–v7

**Status: draft integration; not GPU-validated as a combined build.** The
original v0–v7 folders are unchanged. v8 starts from the v7 source tree and
imports compatible operator-level changes from v1–v6. This is NOT a claim that
seven separate performance results add up, nor that the integration passes
MATH-500 or improves end-to-end serving throughput.

## What is integrated

| Origin | Code present in v8 | Activation |
| --- | --- | --- |
| v1 | BF16 GQA prefill query-tiling kernel and adapter | `VLLM_FL_BIV150_V8_PREFILL=v1` (initial default) |
| v2 | Alternate larger KV-tile prefill kernel and adapter | `...PREFILL=v2` (exclusive with v1/v4) |
| v3 | FlagGems/vendor fused RMSNorm with residual ABI guard | `VLLM_FL_BIV150_RMS_V3=1` |
| v4 | vLLM Triton grouped-page prefill kernel and adapter | `...PREFILL=v4` (disabled by default) |
| v5 | Decode page-index reuse kernel and adapter | `VLLM_FL_BIV150_V8_DECODE=v5` |
| v6 | Triton KV-cache write tiling in FlagGems | `FLAGGEMS_BIV150_KV_V6=1` (disabled by default) |
| v7 | Iluvatar MM/Linear grouped scheduling and heuristics | Present in v7-derived FlagGems source; initially always enabled |

The first, abandoned v4 FlagGems attention backend replacement and its
`flash_api.py`, `flash_kernel.py`, and `iluvatar_grouped_paged_kv.py`
edits are deliberately **not** ported. That separate path caused a major
backend-level serving regression and is not needed for the selectable v4 Triton
operator. No upstream framework code, official benchmark, or model settings
were changed.

## Attention conflict resolution

v1, v2, v4 and v5 independently modified the same
`vllm_fl/dispatch/backends/flaggems/flaggems.py` dispatch hook. v8 preserves
FlagOS's existing CUDA/MLA/sparse checks, explicit
`VLLM_FL_USE_FLAGGEMS_ATTN=1` override, original
`AttentionBackendEnum.TRITON_ATTN` fallback and non-Iluvatar behavior.
Only the Iluvatar custom Triton backend is composed. The v8 adapter inherits
vLLM Triton metadata and cache-update logic.

- For pure decode batches (`max_query_len==1`), optional v5 is selected;
  otherwise the original Triton operator is used.
- For prefill or mixed batches, exactly one of v1/v2/v4 is selected;
  no simultaneous v1+v2 tuning is claimed.
- v4 keeps its per-sequence guard and vLLM fallback for unsupported/short
  prefill; v1/v2 keep their own kernel eligibility/tile selection.
- In mixed batches, v5 **does not** optimize decode rows independently. They
  use the selected prefill operator. Full per-sequence hybrid fusion would
  require another measured kernel change.
- A blank/invalid selector raises rather than silently picking a kernel.
  Changing env vars requires a vLLM process restart.

## Initial safe/conservative runtime configuration

```bash
cd /root/Optimization/v8
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=fl
unset VLLM_FL_USE_FLAGGEMS_ATTN
export VLLM_FL_BIV150_V8_PREFILL=v1
export VLLM_FL_BIV150_V8_DECODE=none
export VLLM_FL_BIV150_RMS_V3=0
export FLAGGEMS_BIV150_KV_V6=0

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

To evaluate **the full non-v4 candidate** (v1 prefill + v3 RMSNorm + v5
decode + v6 KV tiling + v7 GEMM), restart vLLM after setting:

```bash
export VLLM_FL_BIV150_V8_PREFILL=v1
export VLLM_FL_BIV150_V8_DECODE=v5
export VLLM_FL_BIV150_RMS_V3=1
export FLAGGEMS_BIV150_KV_V6=1
```

This is an *experimental combination*, not a validated winner. Revisit
v6/FlagGems import provenance and verify the server really calls the modified
`reshape_and_cache_flash` operator; its v6 isolated tile gains did not
produce a convincing end-to-end throughput improvement.

## Verification gates — execute on BI-V150

1. Confirm imports resolve to `/root/Optimization/v8/` for **both** packages;
   an older installed `vllm_fl` or `flag_gems` invalidates the comparison.
   Confirm `vllm_fl` selected backend and actual per-phase kernel launches.
2. Run `tests/functional_tests/test_biv150_stack_v8.py`,
   `tests/functional_tests/test_biv150_rmsnorm_v3.py`, FlagGems'
   `tests/test_biv150_kv_cache_tiling_v6.py` and
   `tests/test_iluvatar_gemm_linear_v7.py`. Validate all of v1/v2/v4/v5
   against original vLLM attention on real GPU, with shuffled pages,
   nonaligned block mappings, mixed batches, short prefill, 2D/3D decode
   and unsupported-fallback cases. The old individual-version dispatch
   tests require adaptation; do not claim that they were run here.
3. Run the **unchanged competition benchmark** from a separate shell:

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

4. Perform paired OFF/ON restarts holding hardware, vLLM flags, weights,
   warmup, prefix-cache state and benchmark unchanged. Isolate v1+v7 first,
   then v3, v5, v6 individually, then interactions; compare 4K **and** 16K
   total/output tokens/s, TTFT, TPOT, GPU kernels, CPU idle gaps, KV-cache
   waiting/preemptions. A/B v4 and v2 separately, never assume microbenchmarks
   transfer to serving.
5. Run the competition's unmodified MATH-500 evaluation after selecting a
   performance candidate. v1's historical 98.1% on 105 Level-3 questions
   is not an accuracy result for v8.

## Historical selection rationale

- v1 archived full-workload throughput: 2149.82 total tok/s (4K),
  1209.43 total tok/s (16K); reported 98.1% on a 105-question MATH-500
  subset. These were not tests of v8.
- v4 grouped prefill was ~23–25% slower in the isolated 16K Triton
  attention kernel A/B; disabled by default.
- v6/v7 development *half-run* results are not comparable as a matched
  official benchmark to v1. v7's confirmed MM kernel gain is small relative
  to the 16K attention-dominated GPU trace. Do not infer stacked speedup.

## Competition restrictions

No model hyperparameter changes, quantization, speculative sampling, official
benchmark tuning, deletion of major operator-selection logic, or merges from
the latest FlagOS inference framework branch.
