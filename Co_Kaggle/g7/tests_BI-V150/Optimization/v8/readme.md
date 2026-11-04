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

## Experimental v4 grouped paged-KV lookup stacked onto v1

The default **remains v1**. This experimental candidate combines v1's tuned
query tiling (`BLOCK_M=32` for eligible BF16 GQA prefill) with the actual
v4 kernel-level grouped physical-page-table load: load one page ID per
KV-cache page, then expand with Triton `tl.gather`. The original v1
attention kernel and its backend are retained byte-for-byte; the combined
candidate lives in its own plugin-owned kernel and adapter. No FlagGems
backends or framework priority policy are replaced.

**Activation (BI-V150 Iluvatar only):**

```bash
# Default control: tuned v1 kernel only.
unset VLLM_FL_BIV150_GROUPED_V4

# Experimental candidate: v1 tiles + v4 grouped paged-KV loads.
export VLLM_FL_BIV150_GROUPED_V4=1
```

Restart all serving workers after switching the environment variable.
The existing `VLLM_FL_USE_FLAGGEMS_ATTN=1` override still takes precedence;
non-Iluvatar platforms keep the standard Triton backend.

The experimental grouped lookup activates only for decoder-attention
2D prefill with v1's eligible BF16 GQA 16-query-head/2-KV-head,
head-dimension-128 geometry, causal/non-quantized cache, context
`max_seqlen_k >= 8192`, `max_seqlen_q >= 128`, a supported
power-of-two page size, and `TILE_SIZE_PREFILL > block_size` with
page-aligned tiling. Each sequence additionally needs query length >=128
and KV length >=8192; decode sequences in mixed batches use ordinary
per-token page-table addressing. Unsupported cases retain v1 behavior.

**Why experimental:** v4's initial whole-backend FlagGems replacement
caused a severe official serving regression. The revised v4 Triton
prefill-only implementation in `Optimization/v4` was not yet validated
by 4K/16K serving or MATH-500. Earlier kernel-only observations do not
establish a benefit when combined with v1. Keep this off unless GPU A/B
shows a real total-throughput improvement.

**BI-V150 correctness gate:**

```bash
cd /root/Optimization/v8
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
python3 -m pytest -q -s --noconftest \
  vllm-plugin-FL/tests/functional_tests/test_biv150_attention_v1.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_v1_grouped_v4.py
```

Restart the same v8 serving configuration once with the flag unset and
once set to 1. Verify actual selected backend in serving logs, then run
the unchanged competition 4K/16K benchmark
`[[4096,1024,64,256],[16384,1024,64,128]]`.
Record total/output throughput, TTFT, ITL/TPOT, GPU idle time, and
KV-cache preemptions under identical resource and prefix-cache conditions.
Only consider promotion after MATH-500 accuracy passes.

**Status:** CPU/static source review only. No BI-V150 compile, kernel
correctness run, serving A/B, or MATH-500 result has been obtained for
this integrated v1+v4 candidate.
