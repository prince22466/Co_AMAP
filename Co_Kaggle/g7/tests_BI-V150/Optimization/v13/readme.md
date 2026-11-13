# v13 (device-independent refactor of v10)

v13 is the reviewed device-independent attention refactor of v10. v13 should be functionally the same as v10. The original v10 folder remains unchanged.

## Implementation

- The single unified attention kernel is relocated to `vllm-plugin-FL/vllm_fl/dispatch/backends/triton_unified_attention.py` with identical source content (Git blob `d12c2ab2d8c1a68a3cd44bbb99f5f13bb0ba6fbc`).
- The shared adapter is `vllm-plugin-FL/vllm_fl/dispatch/backends/triton_attention.py`.
- `PlatformFL.get_attn_backend_cls` first runs the original FlagOS dispatch. Only a selected stock dense Triton backend maps to the shared optimized implementation. No GPU-vendor gate; non-Triton, custom backends, MLA and sparse selections are preserved.
- Eligible BF16 GQA prefill retains BLOCK_M=64, BLOCK_Q=8 (for eight query heads per KV head), num_stages=2. Eligible pure decode retains page-table lookup reuse. All workload and feature gates and numerical computations are preserved.
- FlagGems, the benchmark scripts and model/serving parameters are unchanged.

## Validation goal

Performance and correctness parity with v10 on BI-V150, then validation on other compatible Triton platforms. This is a refactor, not a new optimization or a performance-gain claim. The historical [v10 results](../../profiling/v10/summary.txt) are reference values only (4K 2372.38 and 16K 1872.70 total tok/s; MATH-500 Level 3 97.1%). Use fresh matched v10/v13 runs for parity. No v13 GPU results are claimed.

See [program.md](program.md) for exact validation commands and [AGENTS.md](AGENTS.md) for workspace and competition constraints. Historical v10 research remains in [v10](../v10/); its v9-targeted autoresearch workflow is not copied into v13.
