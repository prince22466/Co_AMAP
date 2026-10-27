# v3 — BI-V150 fused residual-add + RMSNorm experiment

**Status: unmeasured experimental candidate; keep the PR as draft until hardware
correctness, throughput and MATH-500 are verified.**

## Independent optimization direction

v1 changes the prefill attention query tile (16 → 32); v2 changes the long
prefill attention KV tile (32 → 64). **v3 optimizes a different hot-path
operator:** the RMS normalization used around transformer blocks.

The stock `vendor/iluvatar/impl/normalization.py` uses several PyTorch
operations and temporary tensors to add the residual, reduce the row variance,
compute reciprocal RMS and apply the weight. v3 implements a **one-launch
Triton kernel** which:
- Loads each token row once and conditionally adds the residual.
- Returns a rounded residual tensor when requested, preserving the interface.
- Reduces the sum of squares in FP32 and computes reciprocal RMS.
- Scales by the learned weight and writes the normalized output.

The fast path supports contiguous inference tensors (FP16/BF16/FP32, hidden
size 128–8192, weight and input of the same dtype). Other cases keep the
original backend/reference implementation. No quantization, speculative
sampling or model hyperparameter modifications are introduced.

The opt-in applies within both the **Iluvatar vendor RMSNorm** and the
**FlagGems RMSNorm** path (only on the Iluvatar platform for the latter);
the framework's backend selection remains unchanged. This is an actual
fused operator implementation, not switching to another unmodified operator.

## Changed paths (relative to vllm-plugin-FL)

| Path | Purpose |
| --- | --- |
| `vllm_fl/dispatch/backends/vendor/iluvatar/impl/normalization_biv150_v3.py` | Inference-shape guards and fused operation launcher |\n| `vllm_fl/dispatch/backends/vendor/iluvatar/impl/normalization_biv150_v3_kernel.py` | New fused residual-add/RMSNorm Triton kernel |
| `vllm_fl/dispatch/backends/vendor/iluvatar/impl/normalization.py` | Opt-in vendor dispatch with original fallback |
| `vllm_fl/dispatch/backends/flaggems/impl/normalization.py` | Opt-in for FlagGems dispatch on BI-V150 only |
| `tests/functional_tests/test_biv150_rmsnorm_v3.py` | Numerical and fallback regression tests |

## BI-V150 validation

Use the **v3 source tree** when starting the vLLM process. The benchmark
script is the original competition-provided copy at `/workspace`; do not
edit or tune it.

```bash
cd /root/Optimization/v3/vllm-plugin-FL
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=fl
export VLLM_FL_BIV150_RMS_V3=1

python3 -m pytest -q tests/functional_tests/test_biv150_rmsnorm_v3.py

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

In a **second terminal**, run the unchanged competition script:

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

For an A/B comparison, restart the server with
`VLLM_FL_BIV150_RMS_V3=0` and keep all other server/benchmark settings
unchanged. Confirm from operator profiling that the OOT RMSNorm implementation
was actually selected; otherwise the end-to-end result does not exercise
this kernel. Record both 4K and 16K throughput and regression separately.
Run the official MATH-500 accuracy evaluation on
`/workspace/evalscope-datasets/math_500/test.jsonl`; numerical kernel unit
tests are not a substitute for model accuracy.

**Expected scope:** previous 16K profiling attributed most GPU kernel time to
attention, so v3 might help normalization-heavy shapes without improving the
main 16K score substantially. Retain v3 only if BI-V150 measurements show
net throughput gains with permissible accuracy. Set
`VLLM_FL_BIV150_RMS_V3=0` (or unset it) for immediate rollback.

## Competition compliance

No benchmark script changes, quantization, speculative decoding, changed
hyperparameters, removed operator-selection logic, or merge of an upstream
FlagOS development branch. `FlagGems/` is untouched.
