# v8 — v1 tiled attention + automatic v4 grouped paged-KV lookup

v8 builds on v1's optimized Triton attention. v2's competing tile parameters
are not included; the previously reverted v3 RMSNorm experiment stays out.

## Built-in v1 + v4 activation on Iluvatar

On the Iluvatar platform, `FlagGemsBackend.attention_backend()`
**automatically selects** `BIV150V1GroupedV4AttentionBackend`. There
is **no v4 startup environment variable**: the legacy
`VLLM_FL_BIV150_GROUPED_V4` has no effect on selection.

The new Triton kernel combines two concrete operator optimizations:
- v1's BF16/GQA query tiling (`BLOCK_M=32` for eligible prefill).
- v4's grouped page-ID lookup: load each KV-cache page ID once per tile
  and expand to token lanes with Triton `tl.gather`.

This implementation preserves vLLM's Triton backend metadata, KV-cache write,
operator interfaces and existing FlagOS priority. The original v1 kernel and
adapter remain unchanged for A/B verification and rollback. Non-Iluvatar
platforms retain `TRITON_ATTN`; the existing explicit
`VLLM_FL_USE_FLAGGEMS_ATTN=1` override still takes precedence.

### Eligibility and fallbacks

Grouped page lookup is applied **inside the combined kernel**, only if the
original v1 BF16, causal, nonquantized MiniCPM GQA specialization is eligible
(16 Q heads, 2 KV heads, head dimension 128, no unsupported sinks,
multimodal-prefix, sliding window, ALiBi or softcap), and:
- the batch has 2D prefill, max query length >=128, max KV length >=8192;
- page size and KV tile are power-of-two aligned, and KV tile > page size;
- the individual sequence has query length >=128 and KV length >=8192.

Short prefill, decode, unsupported layouts and mixed-batch decode sequences
retain **v1-equivalent original page lookup** and computation. This is
kernel-level conditional specialization, not a new framework attention policy.

## Start vLLM without an attention-specific toggle

```bash
cd /root/Optimization/v8
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
unset VLLM_FL_USE_FLAGGEMS_ATTN VLLM_ATTENTION_BACKEND

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

The `unset` only removes earlier explicit backend overrides; it is not
required to activate v4. Check the **server worker** import path and confirm
the selected class `BIV150V1GroupedV4AttentionBackend`. A GPU trace is
needed to establish whether long-prefill kernels specialize to grouped
page lookup, rather than assuming this from a selector log.

## BI-V150 validation (not yet run)

```bash
cd /root/Optimization/v8
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
python3 -m pytest -q -s --noconftest \
  vllm-plugin-FL/tests/functional_tests/test_biv150_attention_v1.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_v1_grouped_v4.py
```

For an A/B test, run the **pre-integration v8/v1 baseline** (main at
PR #207 creation, `f059f21695d32c71e494adabff5a0cb3457cf259`)
against this automatic v1+v4 branch. Do **not** attempt to A/B using
`VLLM_FL_BIV150_GROUPED_V4`, which is deliberately ignored. Use
separate server restarts, the same server configuration, GPU conditions
and prefix-cache state, and the unchanged competition benchmark:

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

Compare total/output tokens/s, TTFT, TPOT/ITL, GPU idle gaps and KV
preemptions. Run `math_500` accuracy before promoting to production.

## Historical risk and status

v4's first attempt replaced the default whole-model Triton attention backend
with FlagGems and severely regressed serving. The later v4 Triton operator
rewrite avoided that design but has no confirmed end-to-end speedup. Integrating
it with v1 **does not establish** a throughput gain.

**Status: source-level integration; BI-V150 compilation, correctness, 4K/16K
serving A/B, and MATH-500 accuracy remain mandatory.**
