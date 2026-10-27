# v3 — BI-V150 fused RMSNorm operator experiment (review fixes)

**Status:** source corrected and ready for BI-V150 operator correctness/performance
evaluation. GPU numerical equivalence, official 4K/16K throughput and MATH-500
accuracy are not claimed as passed; keep the PR draft until verified.

## Independent optimization

v1/v2 tune paged-attention tiling. v3 instead tests a BI-V150-specific
RMS normalization Triton kernel. **The default Iluvatar FlagGems RMSNorm
implementation is already fused**; v3 competes against that real kernel,
not against an unfused PyTorch reference. Potential improvements include
not allocating/writing the auxiliary inverse-RMS buffer in the non-residual
case and different row-wise memory/reduction geometry. Net speedup is
an empirical question, particularly for the attention-dominated 16K case.

The v3 kernel retains the FlagGems arithmetic order:
- For residual-add, compute `x + residual` in FP32, use that full-precision
  sum for RMS statistics, and round only when storing residual.
- For RMSNorm without a residual, round the normalized value to input dtype
  *before* multiplying by weight, as in FlagGems.
- Match FlagGems' **in-place residual-add ABI**: return the same input
  `x` and `residual` tensors with updated contents. The vendor's standalone
  reference path retains out-of-place output semantics.
- Accumulate variance in FP32; use a single Triton program per dense row.

The specialized path is opt-in through `VLLM_FL_BIV150_RMS_V3=1`, with
conservative fallback for noncontiguous/misaligned/aliased inputs, unsupported
dtype/shape, 128–4096 hidden-size range, non-CUDA tensors, and autograd.
A real `nn.Parameter` can now pass in `torch.inference_mode()` and
`torch.no_grad()`, while still falling back when gradients are recorded.
The existing FlagGems/vendor/reference backend selection is untouched.

Files changed under `vllm-plugin-FL/`:
- `vllm_fl/dispatch/backends/vendor/iluvatar/impl/normalization_biv150_v3_kernel.py` — single-launch Triton kernel
- `vllm_fl/dispatch/backends/vendor/iluvatar/impl/normalization_biv150_v3.py` — eligibility, launch and in-place contract
- `vllm_fl/dispatch/backends/vendor/iluvatar/impl/normalization.py` — vendor opt-in preserving out-of-place return
- `vllm_fl/dispatch/backends/flaggems/impl/normalization.py` — default FlagGems opt-in preserving in-place ABI
- `tests/functional_tests/test_biv150_rmsnorm_v3.py` — guard, reference, FlagGems-equivalence, aliasing, real OOT dispatch checks

## Step 1: Validate on BI-V150

Start a new shell with **the v3 source on PYTHONPATH**, not the baseline
plugin. The deployment path must match your actual checkout.

```bash
cd /root/Optimization/v3/vllm-plugin-FL
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=fl
export VLLM_FL_BIV150_RMS_V3=1

python3 -m pytest -q tests/functional_tests/test_biv150_rmsnorm_v3.py
```

The GPU test suite includes FP32/FP16/BF16 and residual/non-residual comparisons
against **actual FlagGems**, verifies in-place mutations and returned object
identity, checks vendor out-of-place compatibility, and exercises
`RMSNormFL.forward_oot` with a true Parameter via the default dispatcher.
Any Triton compile failure, numerical mismatch, or dispatch assertion
means stop and fix before performance benchmarking.

## Step 2: Start the v3 server

```bash
cd /root/Optimization/v3/vllm-plugin-FL
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=fl
export VLLM_FL_BIV150_RMS_V3=1
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

Check the v3 source import path and verify by profiler/dispatch tracing that
the fused kernel runs in the server, not just the unit tests.

## Step 3: Official unchanged 4K/16K benchmark

In another terminal, use the competition's original script at `/workspace`.
Do **not** modify the benchmark implementation, workload or arguments.

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

For A/B, stop and restart the server with
`VLLM_FL_BIV150_RMS_V3=0`, holding serving and official benchmark settings
fixed. Record 4K and 16K throughput separately; check memory allocations,
operator runtime, and regressions. Run the official MATH-500 evaluation on
`/workspace/evalscope-datasets/math_500/test.jsonl` at identical settings.
Select v3 only if measured throughput improves with acceptable accuracy.

## Competition compliance

No model hyperparameter changes, quantization, speculative sampling,
benchmark changes, deleted fallback/selection logic, or merge of the newest
FlagOS framework. `v3/FlagGems/` and v1/v2 are unchanged.
