# v2 — BI-V150 long-prefill KV tiling experiment

**Status: draft candidate; no measured speedup or accuracy result claimed.**

v2 is an independent copy of the FlagOS vLLM plugin. It ports v1's actual
BF16 MiniCPM GQA paged-attention kernel and introduces an additional kernel
tile optimization for the competition's 4K/16K prefill-heavy workloads.

## Operator change (v1 → v2)

| Path (inside `vllm-plugin-FL`) | Purpose |
| --- | --- |
| `vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150_v2.py` | Plugin-owned fused paged-attention Triton kernel, based on v1; for supported long BF16 GQA prefill, increase KV tile `TILE_SIZE_PREFILL: 32 → 64` while retaining `BLOCK_M=32` and grouped-query reuse |
| `vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150_v2.py` | Adapter preserving the vLLM 0.24.0 attention metadata and KV-cache update paths |
| `vllm_fl/dispatch/backends/flaggems/flaggems.py` | Explicit `VLLM_FL_BIV150_ATTN_V2=1` dispatch opt-in, without changing default or existing FlagGems selection |
| `tests/functional_tests/test_biv150_attention_v2.py` | Opt-in/fallback, tile-selection boundaries, and BF16 numerical checks against pinned vLLM baseline |

The 64-token KV tile applies only when **all** v1 MiniCPM prefill guards
hold (BF16, 16 Q heads / 2 KV heads, head size 128, causal full attention,
unquantized KV cache, no incompatible features, and no tensor descriptors),
plus `max_seqlen_q >= 128` and `max_seqlen_k >= 2048`.
This changes the kernel's K/V dot and online-softmax loop work per iteration;
it is **not** a bare operator swap. Shorter prefill retains v1 tiling,
while decode and unsupported layouts retain baseline tiling.

## Reproduce and validate on BI-V150

Run the v2 package in a separate environment or process from v1;
`VLLM_PLUGINS=fl` enables the plugin and the v2 flag chooses the kernel.

```bash
cd /workspace/Optimization/v2/vllm-plugin-FL
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=fl
export VLLM_FL_BIV150_ATTN_V2=1
unset VLLM_FL_USE_FLAGGEMS_ATTN
unset VLLM_FL_BIV150_ATTN_V1

# Run focused functional tests on the competition GPU first.
python3 -m pytest -q tests/functional_tests/test_biv150_attention_v2.py

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

In a second shell, use the **unchanged** competition/official throughput
script and exact workload cases used for v1, including 4K and 16K:

```bash
cd /workspace
python3 /workspace/Optimization/v2/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

If the v2 directory is installed elsewhere, adjust filesystem paths only,
not benchmark logic or workload parameters.

Compare v2 against **v1 on the same device**, with identical server settings,
warmup policy, model, concurrency, and measured throughput. Check both cases
independently; reject/regate v2 if the 16K kernel regresses or miscompiles.
Also run the competition MATH-500 accuracy test at
`/workspace/evalscope-datasets/math_500/test.jsonl` through the official
accuracy evaluation workflow. Do not equate attention tensor tolerance with
MATH-500 correctness.

## Competition restrictions

Do not change model hyperparameters, enable quantization or speculative
decoding, tune benchmark scripts, delete framework operator-selection logic,
or merge the latest FlagOS inference-framework branch. `FlagGems/` is
unchanged. No throughput or MATH-500 pass has yet been measured for v2.
