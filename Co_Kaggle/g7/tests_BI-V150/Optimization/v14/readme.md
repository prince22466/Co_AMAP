# v14 — device-independent refactor of v12

v14 applies the same integration pattern as [v13](../v13/) (the device-independent refactor of v10) to [v12](../v12/). It is intended to be functionally equivalent to v12 on BI-V150. The original v12 folder remains unchanged.

## Implementation

- The unified attention kernel is relocated to `vllm-plugin-FL/vllm_fl/dispatch/backends/triton_unified_attention.py` with identical source content (Git blob `3488c183652bd90d7191555e0452c7e77d1bad0c`).
- The shared adapter is `vllm-plugin-FL/vllm_fl/dispatch/backends/triton_attention.py`.
- `PlatformFL.get_attn_backend_cls` first runs the original FlagOS dispatch. Only a selected stock dense Triton backend maps to the shared optimized implementation. There is no GPU-vendor gate for this mapping; non-Triton, custom, MLA and sparse selections retain their original behavior.
- Eligible BF16 GQA prefill retains BLOCK_M=64, BLOCK_Q=8 (eight query heads per KV head), and num_stages=1. Eligible pure decode retains page-table lookup reuse. All workload and feature gates, existing hardware-specific upstream branches, and numerical computations are preserved.
- FlagGems, benchmark scripts, and model/serving parameters are unchanged. No new optimization, quantization, speculative decoding, or sparse approximation is introduced.
- The v12-targeted autoresearch workflow and historical findings remain in [v12](../v12/autoresearch/); they are not duplicated as an active v14 workflow.

## Validation

Run the three functional test files from this version's plugin root with its own FlagGems and plugin on PYTHONPATH:

```bash
cd /root/Optimization/v14
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
python3 -m pytest -q -s --noconftest \
  vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_mixed_paging.py
```

The tests cover dispatch preservation, shared adapter binding, BF16 GQA attention, decode paging, mixed paging and the one-stage prefill launch. Use fresh matched v12/v14 runs on BI-V150 with unchanged competition benchmarks (4K/16K) and MATH-500, then validate on other compatible Triton platforms. v12's [handoff](../v12/autoresearch/HANDOFF.md) records unresolved dense-attention reference discrepancies and provisional performance results; the refactor does not resolve them. No new GPU, official throughput, accuracy, or cross-device results are claimed.
