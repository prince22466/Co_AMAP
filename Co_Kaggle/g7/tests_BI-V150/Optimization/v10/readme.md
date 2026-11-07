# BI-V150 attention optimization (v8)

This folder contains one production attention backend and one unified Triton
attention launcher, with phase-aware kernel specializations integrated into
their original modules. There are no separate version-specific adapters or
duplicate Triton kernel implementations.

## Implementation

- `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150.py`:
  `BIV150TritonAttentionBackend` reuses vLLM's metadata, KV-cache updates,
  and backend interface. Its `forward` method invokes the single launcher.
- `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py`:
  contains the original attention kernel, tuned BF16 GQA prefill tiles, and
  a compile-time `DECODE_PAGE_LOOKUP` specialization for eligible pure decode.
- `vllm-plugin-FL/vllm_fl/dispatch/backends/flaggems/flaggems.py`:
  automatically selects the BI-V150 backend on Iluvatar. Other vendors keep
  the Triton fallback. `VLLM_FL_USE_FLAGGEMS_ATTN=1` retains priority.

Pure decode is identified by `max_seqlen_q == 1`. Paged-KV lookup reuse is
restricted to supported BF16 head/shape/quantization configurations and
otherwise uses the original per-token lookup path. Prefill (including mixed
batches) retains the tuned tile configuration when eligible; unsupported
prefill shapes retain the original kernel settings. The two compile-time
optimizations are independent and never active for the same invocation.

## Validation

```bash
cd /root/Optimization/v8
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
python3 -m pytest -q -s --noconftest \
  vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py
```

On BI-V150, verify actual kernel selection, correctness, throughput, and
accuracy with the unchanged official 4K/16K benchmark and MATH-500 cases.
Source inspection and CPU-only checks do not establish hardware performance.
