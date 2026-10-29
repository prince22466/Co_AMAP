# v4 — Built-in grouped paged-KV lookup for 16K prefill (Triton attention)

## Why this follow-up exists

The original v4 PR (#161) made FlagGems the **default attention backend**
on Iluvatar so its grouped page-lookup kernel would run without an enable
flag. That unintentionally replaced vLLM's `TRITON_ATTN` implementation
for **decode as well as prefill**. The resulting official 16K run reported
**237.83 total tokens/s**, **13.99 output tokens/s**, a **9,368.85 s**
duration and extreme TTFT/ITL. This is a blocking serving regression.

A previously observed **1.1744×** grouped-kernel speedup on full 16K
attention (and **1.1914×** on 2K/16K chunked attention) was measured
inside **FlagGems v4_preplay**. It did **not** establish that replacing the
entire vLLM attention backend would be faster.

## Revised design: optimize the existing Triton operator only for prefill

The standard vLLM-FL dispatcher remains in place. On Iluvatar, the FlagOS
attention-backend hook now returns a subclass of vLLM's existing
`TritonAttentionBackend`, rather than substituting FlagGems attention.
The subclass **inherits the standard vLLM metadata builder and KV-cache
update implementation**.

- **Decode-only requests:** invoke `TritonAttentionImpl.forward()`
  directly, using the original vLLM kernel (including CUDA-graph decode).
- **4K prefill and unsupported shapes:** likewise invoke the original
  `TritonAttentionImpl.forward()`.
- **Eligible 16K prefill:** call a plugin-owned copy of the **vLLM
  v0.24.0** unified Triton attention operator, with only the page-address
  lookup optimized. The original `BLOCK_M`, `BLOCK_Q`, KV tile and
  attention math/softmax are retained; v1's tile-sweep experiment is not
  included.
- **Mixed prefill+decode batches:** the kernel checks
  `cur_batch_query_len` and `seq_len` **for each sequence**. The decode
  sequences use the original page-table load, even if another sequence
  activates grouped prefill in that same kernel launch.

For a KV tile holding 32 tokens with a page size of 16, the optimized
Triton kernel loads two physical page IDs and expands them to the token
lanes with `tl.gather`, replacing 32 repeated per-token page-index
lookups. It preserves the original K/V loads, masks, dot products,
softmax and output.

This is **a new port of the grouped-addressing idea into the vLLM
Triton operator**, not a claim that the former FlagGems kernel's
measured speedup transfers automatically.

## Changes in this PR

| File | Purpose |
| --- | --- |
| `vllm-plugin-FL/vllm_fl/dispatch/backends/flaggems/flaggems.py` | Undo the automatic FlagGems backend replacement. Default Iluvatar FlagOS path uses a vLLM Triton-derived optimized backend; keep existing dispatch selection, fallbacks and explicit FlagGems override |
| `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_grouped_v4.py` | Thin vLLM Triton backend adapter: original decode/short-prefill fallback, custom prefill forward only for eligible shapes |
| `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_grouped_v4.py` | vLLM v0.24.0 Triton unified attention with grouped page loads gated by shape and per-sequence phase; no v1 tile changes |
| `vllm-plugin-FL/tests/functional_tests/test_biv150_grouped_prefill_v4.py` | GPU correctness checks against original vLLM Triton on shuffled physical pages, partial pages, long prefill and mixed prefill/decode |
| `readme.md` | Rationale, code map, verification and competition gates |

The previous FlagGems grouped implementation from #161 remains in the
source tree but is **not the automatically selected serving attention
backend**. No unrelated operators, benchmark scripts or model settings
are modified.

## Automatic eligibility

The custom Triton prefill path requires the following:

- Iluvatar BI-V150 via the existing FlagOS dispatcher;
- BF16, head dimension 128 and GQA ratio 8 (MiniCPM5-2B geometry);
- batch max query length ≥128 and max KV length ≥8192;
- no KV quantization, and no unsupported page/tile alignment;
- no sliding-window attention or multimodal-prefix masking;
- 2D prefill, with the existing vLLM KV tile strictly larger than the
  cache page size (both power-of-two aligned).

Within such a batch, a sequence must itself have **query length ≥128**
and **KV length ≥8192** to use grouped page addressing. Others use
the original page lookup. This is especially important for continuous
batching, which can mix decode and prefill work.

## Server and test commands

Install/load the modified `v4/vllm-plugin-FL` package and confirm the
server process imports it (not an older installed copy). Standard
serving requires **no v4 enable environment variable**:

```bash
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

On the BI-V150 machine, run focused **correctness** checks (not a
standalone performance profiler):

```bash
cd /root/Optimization/v4/vllm-plugin-FL
python3 -m pytest --noconftest -vv -s \
  tests/functional_tests/test_biv150_grouped_prefill_v4.py
```

Then run the **unchanged official 4K/16K serving benchmark**:

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

Verify the actual selected backend is
`BIV150GroupedPrefillAttentionBackend`. Compare separately with the
original, unmodified `TRITON_ATTN` serving run. Record **total
tokens/s**, output tokens/s, TTFT, TPOT and KV-cache preemptions for
both 4K and 16K.

Finally, run the competition's `math_500` accuracy evaluation. Do not
submit on the basis of kernel microbenchmarks alone.

## Competition compliance / status

No quantization, speculative decoding, model hyperparameter edits,
official benchmark script edits, or direct merge from the latest FlagOS
branch. We preserve the framework's core operator-selection logic and
make a real memory-access improvement in the existing Triton operator.

**Status: candidate / Draft PR.** GPU compile, correctness, 4K/16K
serving throughput and `math_500` have not yet been run for this new
Triton implementation. The older FlagGems measurements are historical
evidence only.
