# Groups 12–14: Attention, Shared Memory/Async, Atomics (BI-V150 / Triton)

Standalone GPU performance laboratories, extending Groups 1–11. No changes to
MiniCPM, vLLM, FlagGems, quantization, speculative decoding, model
hyperparameters, official benchmark scripts, or operator-dispatch decisions.

## Quick start

From `Co_AI_Inference/Triton_Learning` on the **target GPU**:

```bash
bash run_groups_12_14.sh smoke
bash run_groups_12_14.sh full
PROFILE=1 METRICS=time_duration bash run_groups_12_14.sh full
```

`smoke` performs independent CPU reference checks and correctness-only GPU
launches. `full` additionally measures GPU-event latency across three trials,
saves CPU/GPU version metadata, and emits compiler stages when available.
Use `OUT_DIR=/some/path` for results, `CONTEXTS=4096,16384` for Group 12's
contexts, or `PROFILE=1` for **optional** isolated `ixkn-cli` captures. Native
counter runs are separate from the uninstrumented latency experiment. Inspect
the actual numeric counter rows and matching kernel name before attributing
a bottleneck; a successful process exit or `status=ok` alone is not evidence
that meaningful hardware counters were collected.

Can run the mathematical invariant tests on a CPU-only machine with PyTorch:

```bash
python3 test_groups_12_14_math.py
```

These CPU checks do not validate GPU lowering or kernel synchronization.
Every GPU experiment also independently compares the **full output** to a CPU
reference before timing. Failures produce `status=error`, write the CSV and
make the corresponding group command exit nonzero. Unavailable optional
features produce `status=unsupported` and are never silently replaced by
another implementation.

## Group 12: paged attention and KV-cache

`group12/triton_paged_attention_lab.py` implements exact **single-query,
GQA decode** against permuted pages of K/V, using stable streaming online
softmax and a CPU float64 reference. The GPU kernel reads the same Q, K/V,
and block table for each A/B variant, with no sparse approximation.

| Lab | Baseline | Candidate | Main question |
| --- | --- | --- | --- |
| A01 | token16: per-token block-table lookup | page16: one page-table load per 16-token page | Is page-granularity indexing advantageous? |
| A02 | page16 | token32: two-page tile with per-token lookup | Combined tile-width and lookup trade-off; not a pure tile-size isolation |

Default contexts: 4K and 16K; page size fixed to 16 for controlled comparisons;
head dimension 128, 8 query heads and 2 KV heads. The **smoke** runner additionally
checks a non-multiple-of-page context to exercise masking. Both variants
execute all attention scores, no drop in long-range information.

```bash
python3 group12/triton_paged_attention_lab.py --lab A01 --contexts 4096,16384 \
  --dump-asm results/asm12 --out results/a12.csv
python3 group12/triton_paged_attention_lab.py --lab A01 --variant page16 \
  --contexts 4096 --single-run --out results/page_probe.csv
```

**Scope limitation:** These are decode-like single-query kernels, not
prefill FlashAttention, mixed prefill+decode serving, or full-model throughput
tests. The A02 comparison changes two mechanisms at once. Benchmark timings
also do **not** include allocation, K/V construction, or host-side page-table
creation; they represent steady-state kernel execution.

## Group 13: shared memory and explicit async copy

`group13/triton_shared_async_lab.py` deliberately distinguishes
compiler-managed staging from explicit shared-memory/async instructions.

| Lab | Mechanism | Expected on the current BI-V150 Triton build |
| --- | --- | --- |
| H00 | Core Triton tiled transpose, sizes 16 and 32, compiler stages and shared-byte metadata | Runs, but shared traffic is **not guaranteed** |
| H01 | Gluon explicit shared allocation, store, CTA barrier, load; plain vs swizzled layout | Likely **unsupported** because previous Group5 S04 reported missing Gluon IR bindings |
| H02 | Core Triton synchronous copy vs opt-in NVIDIA Ampere+ Gluon `cp.async` | Synchronous baseline runs; NVIDIA async path is **unsupported on BI-V150** |
| H03 | Explicit Gluon `cp.async` producer/consumer pipeline: one SMEM buffer versus two, with prefetch/fill/drain and waits | **Unsupported on BI-V150** with this backend; available only after explicit NVIDIA Ampere+ opt-in |

```bash
python3 group13/triton_shared_async_lab.py --lab H00 --single-run \
  --out results/h00.csv
python3 group13/triton_shared_async_lab.py --lab H01 --require-gluon \
  --out results/h01.csv
# NVIDIA Ampere+ only, on a verified compatible system:
python3 group13/triton_shared_async_lab.py --lab H02 --enable-nvidia-async \
  --require-nvidia-async --out results/h02.csv
python3 group13/triton_shared_async_lab.py --lab H03 --enable-nvidia-async \
  --require-nvidia-async --out results/h03.csv
```

For H01/H02-async/H03, `shared_bytes > 0` and correct output are required for a
successful implementation. H03 attempts real multi-buffered prefetch/compute overlap only on NVIDIA Ampere+; successful\ncompilation alone does not prove overlapping execution. Actual async instructions, memory overlap, and
bank-conflict reduction require generated ISA and hardware counter inspection.
Do **not** claim hardware async-copy results on BI-V150 from source-level
Triton pipeline hints. Groups 5 and 6 already cover barrier overhead and
compiler-managed pipelining; this suite does not repeat their full experiments.

## Group 14: atomics, contention, multi-CTA reduction

`group14/triton_atomic_sync_lab.py` uses exact int32 increments and a
CPU integer reference, avoiding floating-point atomic ordering ambiguity.

| Lab | Comparison | Fairness notes |
| --- | --- | --- |
| B01 | Same number of global atomics, different hot-bin counts (1/8/64 by default) | Same input work; destination distribution changes |
| B02 | Per-lane global atomics vs a CTA-local reduction followed by one global atomic | **Identical** mapping from every element to output bins |
| B03 | One global atomic per CTA vs 2-kernel scratch partials + final reduction | Measures entire operation, including atomic-output initialization and second launch |

```bash
python3 group14/triton_atomic_sync_lab.py --lab all \
  --n 65536 --bins 64 --out results/b14.csv
python3 group14/triton_atomic_sync_lab.py --lab B02 --variant cta \
  --single-run --out results/b02_probe.csv
```

Atomic runs reinitialize their output on **every measured invocation**, so
the reset is included in the total event duration. Two-pass reduction writes
its output without a reset, and measures both kernels. This is an end-to-end
algorithm comparison, not an attempt to isolate the intrinsic cost of an
atomic instruction. No unsafe spin-wait inter-CTA barrier is used.

## Interpretation and next steps

For credible conclusions, compare only workloads with identical inputs and
logical operations, use idle-device repetitions, inspect the compiled ISA,
and corroborate resource and memory claims with supported BI-V150 counters.
Numerical accuracy is checked independently of timing. A successful 10–30%
kernel microbenchmark speedup does **not** imply an equivalent increase in
MiniCPM 4K/16K total tokens/s.

The remaining major unimplemented mechanisms are full prefill/mixed
attention, BI-V150-native explicit asynchronous producer/consumer pipelines
(if supported by a future backend), and GPU contention across concurrently
executing serving kernels. Those require dedicated integration and profiling
rather than pretending that a portable fallback validates native async.
