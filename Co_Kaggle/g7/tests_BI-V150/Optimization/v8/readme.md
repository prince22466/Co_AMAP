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
