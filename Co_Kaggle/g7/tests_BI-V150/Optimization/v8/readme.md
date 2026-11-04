# v8 — new start point for further optimization

**Goal:** v8 first copy everything of v1, and stack other applicable optimization efforts from v2 to v7.  
v2 will be skipped, because it is same direction of v1 but with different parameters and less performative.

## Automatic v1 + v5 attention activation

On Iluvatar, `FlagGemsBackend.attention_backend()` now automatically
selects `BIV150V1V5AttentionBackend` (the combined Triton adapter).
Within its `forward` method, **v1 handles prefill and mixed batches**,
and **v5 handles invocation-wide pure decode**. The original tuned v1
attention kernel is unchanged; v5's page lookup optimization is a
separate, shape-gated Triton kernel.

Neither `VLLM_FL_BIV150_ATTN_V1` nor a new v5 enable flag is needed.
No model or official benchmark changes are made, and the original
framework operator-priority policy is preserved. Non-Iluvatar devices
retain `TRITON_ATTN`, and the explicit
`VLLM_FL_USE_FLAGGEMS_ATTN=1` override remains available.

Start v8 with its `FlagGems/src` and `vllm-plugin-FL` on
`PYTHONPATH` and `VLLM_PLUGINS=fl`, without an attention-specific
toggle. Kernel correctness and official throughput/accuracy must be
validated on BI-V150 before promotion.

## v5 pure-decode stack (experimental; source only)

v8 now **combines v1 prefill with v5 decode** through
`attention_biv150_v1_v5.BIV150V1V5AttentionBackend` on Iluvatar.
This is automatic; do not set `VLLM_FL_BIV150_V8_DECODE` or other new
activation environment variables.

- `attn_metadata.max_query_len == 1`: use the original v5 Triton
  unified-attention launcher and its kernel-level paged-KV page-index
  reuse, for eligible BF16 16Q/2KV GQA / head-size-128 decode only.
  v5 itself guards tensor descriptors, quantization, unsupported layouts,
  and incompatible page/tile geometries, falling back to per-token lookup.
- `max_query_len > 1` (including mixed prefill/decode): keep the
  unchanged v1 attention adapter and tuned prefill kernel.
- The unchanged explicit FlagGems override, CUDA/MLA/sparse guards and
  non-Iluvatar `TRITON_ATTN` fallback retain priority. No benchmark,
  model, FlagGems, v1-v7, or framework operator-selection edits.

We preserve separate original v1 and v5 kernels/adapter files as a
source-comparable A/B baseline. The only new selection is inside the
existing vLLM Triton attention operator's forward method.

### BI-V150 validation required before merging

```bash
cd /root/Optimization/v8
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
python3 -m pytest -q -s --noconftest \
  vllm-plugin-FL/tests/functional_tests/test_biv150_attention_v1.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_v1_v5_stack.py
```

Verify production workers load this v8 adapter and trace which kernel
actually runs for pure decode versus mixed/prefill. With identical model,
server startup, prefix-cache conditions and fresh restarts, A/B against
the previous v8=v1 baseline using the **unchanged** official benchmark:

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

Compare total/output tok/s, TTFT, TPOT, GPU decode/prefill split,
GPU idle gaps, KV-cache occupancy, waiting and preemptions; repeat
`math_500`. Historical v5 microbenchmarks improved isolated decode
calls by roughly 10–24% across tested cases, but 16K serving remained
capacity/queueing-constrained, and those tests do not validate this stack.
**No BI-V150 combined-kernel test, matched serving A/B, or accuracy
score for v8+v5 is claimed. Do not merge on source inspection alone.**

## v6 paged-KV write tiling stacked onto v1+v5 (experimental)

The original **v6** FlagGems Triton `reshape_and_cache_flash_kernel` and its
GPU correctness test are copied exactly into v8, while the v1 and v5 attention
kernels and the existing operator-selection policy remain unchanged.

**Production activation:** the v8 combined attention implementation now calls
the v6 kernel for compatible writes of **at least 256 tokens**, using four
tokens per CTA and masked per-token paged-slot indices. The integration invokes
the existing v6 Triton kernel directly to avoid accidentally selecting the
FlagGems C-extension path. It requires NHD-contiguous 4D K/V page views,
FP16/BF16 keys, values and cache, contiguous slot indices, and unquantized
(`auto`) KV storage. All smaller writes, HND layouts, nonstandard strides,
FP8/per-token-head quantization, and encoder cases retain the inherited vLLM
Triton KV-cache update. No environment toggle is required.

**Important:** standalone v6 `reshape_and_cache_flash` is not the default
KV-cache writer for v8's v1+v5 attention backend. Copying the FlagGems file
alone would not activate v6 under that backend. This PR adds the narrowly
guarded update hook to the combined adapter; neither v1 nor v5 attention
forward paths are replaced.

### Verify v6 on BI-V150 before accepting the stack

```bash
cd /root/Optimization/v8
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl

# Separate processes: both repositories define a top-level Python "tests" package.
(
  cd FlagGems
  python3 -m pytest -q -s --noconftest \
    tests/test_biv150_kv_cache_tiling_v6.py
)
(
  cd vllm-plugin-FL
  python3 -m pytest -q -s --noconftest \
    tests/functional_tests/test_biv150_v1_v5_v6_stack.py \
    tests/functional_tests/test_biv150_attention_v1.py \
    tests/functional_tests/test_biv150_v1_v5_stack.py
)
```

Inspect a serving-worker GPU trace to confirm that eligible prefill batches
execute `reshape_and_cache_flash_kernel` with `TOKENS_PER_PROGRAM=4` and
that decode/fallback still use the original vLLM writer. The inherited paths
should not select the v6 JIT. Compare official 4K/16K throughput with a fresh
v8=v1+v5 baseline under identical resources/cache conditions and run
`math_500`; do not change the benchmark script. Archived standalone v6
half-run throughput (4K: 1815.76, 16K: 902.94 total tok/s) is **not** a
controlled A/B for the new v8 stack. Verify compiled kernel performance and
accuracy on BI-V150 before merging.
