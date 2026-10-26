# BI-V150 / MiniCPM5-2B — v1 prefill query tiling experiment

**Status:** candidate for GPU validation; no throughput improvement or MATH-500
accuracy has been established yet.

## Scope

- Base plugin: FlagOS `vllm-plugin-FL` at commit `13eb9be69`.
- Attention source base: `vllm-project/vllm` tag `v0.24.0`
  (`vllm/v1/attention/ops/triton_unified_attention.py`).
- First candidate: `BLOCK_M=32` and `BLOCK_Q=4` instead of `16`/`2` for
  BF16, causal, non-sliding-window prefill with 16 Q heads, 2 KV heads,
  head size 128, and unquantized KV. All other configurations preserve the
  original tiling decisions. The internal Triton attention kernel, KV loading,
  QK/PV dot products and softmax remain the existing implementation, with
  the enlarged query tile changing the compiled work granularity and reuse.
- Unmodified: `FlagGems/`, original installed upstream vLLM source, global
  dispatch priority, official benchmark scripts, quantization and sampling.

This is *operator-level kernel tiling* inside a versioned implementation;
it is not a request to switch to a faster unmodified operator.

## Files

- `vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py`:
  independent pinned vLLM 0.24.0 source copy with v1 BF16-GQA prefill tiling.
- `vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150.py`:
  subclass of the upstream attention backend/impl; same forward logic as
  vLLM 0.24.0, but invokes the plugin-owned kernel module.
- `vllm_fl/dispatch/backends/flaggems/flaggems.py`:
  explicit opt-in to the BI-V150 backend, otherwise original selection.
- `tests/functional_tests/test_biv150_attention_v1.py`:
  dispatch fallback and numerical equivalence checks on GPU.

## Activate without modifying site-packages

Run from the cloned `Co_AMAP` repository root:

```bash
PLUG="$PWD/Co_Kaggle/g7/tests_BI-V150/Optimization/v1/vllm-plugin-FL"
export PYTHONPATH="$PLUG${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=fl

# Check which copy of vllm_fl Python imports.
python -c 'import vllm_fl; print(vllm_fl.__file__)'

# Original baseline dispatch; no opt-in.
VLLM_FL_BIV150_ATTN_V1=0 python -c \
  'from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend; print(FlagGemsBackend().attention_backend())'

# Candidate dispatch (requires an Iluvatar BI-V150 GPU).
VLLM_FL_BIV150_ATTN_V1=1 python -c \
  'from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend; print(FlagGemsBackend().attention_backend())'
```

`VLLM_FL_USE_FLAGGEMS_ATTN` must be unset or zero when enabling v1. The opt-in
flag is for experiment selection only. Without it, the default attention backend
and FlagOS fallback chain remain unchanged. We do not modify the installed vLLM
kernel file.

## Compile and test

```bash
cd "$PLUG"
python -m py_compile \
  vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150.py \
  vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py \
  vllm_fl/dispatch/backends/flaggems/flaggems.py \
  tests/functional_tests/test_biv150_attention_v1.py

python -m pytest -q tests/functional_tests/test_biv150_attention_v1.py
```

The GPU tests require running on BI-V150. CPU-only CI may test the dispatch
cases but skip numerical GPU kernels. The expensive 16K-context test deliberately
emulates a 2K prefill chunk starting at position 14,336.

## End-to-end A/B procedure

Restart the server for each variant (do **not** change server arguments,
prompts, request concurrency, dtype, KV-cache dtype, model weights, or benchmark
scripts between A and B).

```bash
# A. Baseline, separate shell; other serve flags identical to B.
VLLM_FL_BIV150_ATTN_V1=0 vllm serve /workspace/MiniCPM5-2B \
    --served-model-name minicpm --host 0.0.0.0 --port 9031 \
    --trust-remote-code

# Stop A completely, then run B using the SAME flags.
VLLM_FL_BIV150_ATTN_V1=1 vllm serve /workspace/MiniCPM5-2B \
    --served-model-name minicpm --host 0.0.0.0 --port 9031 \
    --trust-remote-code
```

The commands above illustrate the *matching-arguments requirement*, not an
official competition startup configuration; use your actual tested serving
flags unchanged in both runs. In particular, previously used `--enforce-eager`
or GPU-memory flags affect performance and must not be varied in the comparison.

For both variants: run the **unaltered** competition 4K/16K workloads, capture
total tokens/s, output tokens/s, prefill/decode latency, runtime errors, and GPU
trace hot kernels. Re-run with fresh/uncached prompts as appropriate to avoid
prefix-cache bias. Then run MATH-500 with unchanged evaluation settings and
check its required accuracy threshold. Do not accept v1 solely on kernel-time
share or diagnostic wall time.

## Recovery

No upstream files are overwritten. To restore default attention behavior,
stop the candidate server, set `VLLM_FL_BIV150_ATTN_V1=0` (or unset it),
and restart using exactly the same `PYTHONPATH` and serve command.
