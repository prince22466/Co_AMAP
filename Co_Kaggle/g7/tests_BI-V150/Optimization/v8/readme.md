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

## v7 GEMM / Linear kernel stack (experimental)

v8 additionally carries the **existing FlagGems Iluvatar MM and Linear kernel**
changes from standalone v7, preserving the above **v1 prefill/mixed + v5
pure-decode attention stack**. The four v7 files are copied unchanged:

- `FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/mm.py`:
  grouped M-tile scheduling for large-M GEMMs (GROUP_M=8), keeping the
  original row-major mapping for M <= 128, and edge-masked loads
  instead of modulo-wrapped addresses.
- `FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/linear.py`:
  edge-masked loads and FP32 accumulator forwarding to `tl.dot`, with
  the existing `allow_tf32=False` and optional bias behavior.
- `FlagGems/src/flag_gems/runtime/backend/_iluvatar/heuristics_config_utils.py`:
  add the matching `EVEN_M` and `EVEN_N` Linear compile-time heuristics.
  These are required by the updated Linear kernel signature.
- `FlagGems/tests/test_iluvatar_gemm_linear_v7.py`:
  hardware-gated FP16/BF16 correctness checks for decode-like and
  prefill-like shapes, tails, transposed weights, bias, `mm_out`, and 1D/3D inputs.

These changes affect the *implementation of existing* Iluvatar operators.
There is no v7 activation switch, no replacement operator backend, and no
change to FlagOS/vLLM dispatch priorities, attention selection, model settings,
quantization, speculative decoding, or the official benchmark script.
Other platform implementations are unchanged. Standalone v7 remains intact.

### Required BI-V150 validation

```bash
cd /root/Optimization/v8
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl

python3 -m pytest -q -s --noconftest FlagGems/tests/test_iluvatar_gemm_linear_v7.py vllm-plugin-FL/tests/functional_tests/test_biv150_attention_v1.py vllm-plugin-FL/tests/functional_tests/test_biv150_v1_v5_stack.py
```

Also run the existing FlagGems `test_mm.py`, `test_linear.py`, and
`test_addmm.py` suites in the same supported GPU environment. Verify in
production GPU traces that the new MM/Linear kernels are executed, since vLLM
may use native GEMMs on some paths. Compare identical v8(v1+v5) and
v8(v1+v5+v7) server runs on both **unchanged** official 4K/16K
cases `[[4096,1024,64,256],[16384,1024,64,128]]`, then run `math_500`.

**This port has not been GPU-tested or benchmarked in this PR.**
Do not claim a serving throughput or accuracy improvement until the
hardware checks and paired official measurements pass.
