# v8 — new start point for further optimization

**Goal:** v8 first copy everything of v1, and stack other applicable optimization efforts from v2 to v7.  
v2 will be skipped, because it is same direction of v1 but with different parameters and less performative.

## Automatic v1 attention activation

In v8, `FlagGemsBackend.attention_backend()` chooses the existing
`BIV150TritonAttentionBackend` automatically when vLLM's current platform
reports `vendor_name == "iluvatar"`. No
`VLLM_FL_BIV150_ATTN_V1` environment variable is required.

This uses the already tuned v1 Triton unified-attention kernel; it does not
change the model, the official benchmark, or the framework's per-operator
priority policy. Non-Iluvatar devices retain standard
`TRITON_ATTN`. The existing explicit
`VLLM_FL_USE_FLAGGEMS_ATTN=1` override remains available.

Start v8 with its `FlagGems/src` and `vllm-plugin-FL` on
`PYTHONPATH` and `VLLM_PLUGINS=fl`; no attention-specific toggle is needed.
The selector tests check automatic selection, platform fallback, and the
explicit FlagGems override. GPU correctness and official throughput/accuracy
benchmarks must be run on BI-V150 before accepting this candidate.

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
