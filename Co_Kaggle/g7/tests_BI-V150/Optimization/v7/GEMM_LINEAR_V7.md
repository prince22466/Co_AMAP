# v7 — Iluvatar GEMM / Linear kernel optimization candidate

**Status:** Candidate; BI-V150 correctness, kernel timing, and official 4K/16K
serving throughput have **not** yet been measured for this change. Do not merge
on the basis of code review alone.

## Scope and hypothesis

This PR modifies the **existing** FlagGems Iluvatar kernels at
`FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/{mm,linear}.py`.
No model hyperparameters, quantization, speculative decoding, vLLM dispatch,
attention/scheduler logic, or competition benchmark scripts are changed.

* **MM — L2-friendly grouping:** The pre-existing `GROUP_M=8` parameter was
  not used by `mm_kernel`. Group program IDs into nearby M tiles before
  advancing across N, improving the potential reuse of A/B tiles in L2.
  Grouping degenerates naturally when decode's `M=1` produces one M tile.
* **MM and Linear — edge-safe address generation:** Avoid `% M` / `% N`
  remapping of out-of-range tiles. Mask invalid rows and columns rather than
  loading duplicate edge data. Compile-time `EVEN_M`/`EVEN_N` specializations
  keep interior loads unmasked when the full dimension is tile-aligned.
* **Linear — accumulate into dot:** Pass the existing fp32 accumulator directly
  to `tl.dot` rather than producing a temporary dot result then performing an
  additional vector add. Preserve `allow_tf32=False` and the existing bias
  behavior.

These are **kernel-level implementation changes**, not a backend swap.
Performance improvement is a hypothesis until BI-V150 data confirms it.
Any change in floating-point accumulation order may affect low-order bits.

## Correctness first

On BI-V150, activate v7's FlagGems using the same `PYTHONPATH` setup as the
baseline, then run from `v7/FlagGems`:

```bash
python3 -m pytest -q -s tests/test_iluvatar_gemm_linear_v7.py
```

The dedicated tests exercise FP16/BF16, prefill-like M=4096/16384,
decode-like M=1, ragged M/N/K, weight transpose, bias, `mm.out`, and
1D/3D Linear input. They import the Iluvatar implementation directly so a
Torch or vendor fallback cannot hide a kernel failure. These tests are
hardware-gated and are not claimed as passed.

Also run the existing FlagGems `test_linear.py`, `test_mm.py`, and
`test_addmm.py` suites with the regular BI-V150 environment.

## Performance and dispatch verification

1. Run **unmodified** v0 and v7 under identical environment, compiler cache
   state, model, concurrency, server arguments, and benchmark input.
2. Profile the real vLLM serving execution. Verify that the changed
   `mm_kernel` / `linear_kernel` actually execute; vLLM may also call native
   GEMM implementations. If the modified kernels are not on the measured hot
   path, there is no justified throughput claim.
3. Record kernel duration, GEMM fraction of GPU time, occupancy/register
   pressure, and GEMM-related L2/cache metrics (where supported). Separate
   decode GEMM from full prefill GEMM and prefix-cache-heavy runs.
4. Run the provided official benchmark without modifications:

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

The benchmark assumes the existing server is running with the corresponding
v0 or v7 checkout. Collect per-case total tok/s, request/s, TTFT, and
profiling evidence. Evaluate `math_500` with the competition's unchanged
accuracy procedure before considering this candidate for submission.

## Rollback / acceptance

Accept only if this kernel change is confirmed on the serving hot path,
does not regress correctness or competition accuracy, and yields repeatable
throughput gain (particularly on the 4K/16K workloads) without a significant
regression in the other case. Otherwise reject or revise the candidate. The
v0-v6 folders remain untouched.
