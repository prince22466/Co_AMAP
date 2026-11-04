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

## Stack v3 — automatic fused RMSNorm (BI-V150)

v8 integrates the v3 fused RMSNorm kernel without a per-run activation
variable. On `current_platform.vendor_name == "iluvatar"`, the existing
FlagGems RMSNorm selects the specialized kernel only for eligible dense
inference inputs (contiguous FP16/BF16/FP32, hidden size 128–4096). The
original FlagGems `gems_rms_forward` remains the automatic fallback for
ineligible shapes, devices, shared-storage residuals, and autograd.

The kernel retains FlagGems' **in-place residual-add contract** and
FP32 accumulation/dtype rounding. Iluvatar vendor RMSNorm uses the same
kernel for eligible inputs but remains **out of place**. The v1 optimized
attention backend, framework operator-selection priority, model settings,
and official benchmarks are unchanged.

To test on BI-V150, from `v8/vllm-plugin-FL` run:
`python3 -m pytest -q --noconftest tests/functional_tests/test_biv150_rmsnorm_v3.py`

Verify a production OOT trace, unchanged official 4K/16K throughput, and
MATH-500 accuracy before accepting the stack. No speedup is claimed yet.
