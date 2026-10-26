# v1 — BI-V150 attention prefill optimization

**Goal:** Improve MiniCPM5-2B BF16 inference throughput on BI-V150, especially the 16K prefill workload. This is an **untested candidate**; throughput and MATH-500 accuracy must be verified before merging.

## Changes in PR #150

All paths below are relative to `vllm-plugin-FL/`.

| File | Change | Function and purpose |
| --- | --- | --- |
| `vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py` | **Added** | Plugin-owned vLLM 0.24.0 Triton unified-attention operator. Tests larger prefill query tiles (`BLOCK_M=32`, `BLOCK_Q=4` versus `16`/`2`) for MiniCPM BF16 causal GQA (16 query heads, 2 KV heads, head size 128) to improve query/KV reuse. Other configurations retain baseline tile selection. |
| `vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150.py` | **Added** | Backend adapter connecting the optimized operator to vLLM's existing attention metadata and KV-cache flow. |
| `vllm_fl/dispatch/backends/flaggems/flaggems.py` | **Modified** | Adds explicit `VLLM_FL_BIV150_ATTN_V1=1` opt-in for BI-V150. With the flag unset, the original `TRITON_ATTN` selection remains unchanged. |
| `tests/functional_tests/test_biv150_attention_v1.py` | **Added** | Tests opt-in/fallback dispatch and compares kernel outputs against baseline, including chunked prefill near 16K context. |
| `BIV150_V1_EXPERIMENT.md` | **Added** | Documents activation, validation, baseline/v1 A/B benchmarking, MATH-500 checking, and rollback. |

`FlagGems/` and upstream installed vLLM are **unchanged**. The experiment does not alter model hyperparameters or official benchmark scripts and does not enable quantization or speculative decoding.

**Next step:** Run syntax and GPU correctness tests on BI-V150; then compare unchanged 4K/16K throughput and MATH-500 accuracy before accepting the candidate. See [experiment guide](vllm-plugin-FL/BIV150_V1_EXPERIMENT.md).
