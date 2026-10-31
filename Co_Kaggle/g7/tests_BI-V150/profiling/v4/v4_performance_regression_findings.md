# v4 performance regression — root-cause investigation (BI-V150)

**Date:** 2026-10-31  
**Model / workloads:** MiniCPM5-2B, BI-V150, official serving 4K/16K with 1024 output tokens, concurrency 64  
**Code history:** [original v4 PR #161](https://github.com/prince22466/Co_AMAP/pull/161); [prefill-only Triton follow-up PR #165](https://github.com/prince22466/Co_AMAP/pull/165)  
**Finding:** The grouped physical-page lookup **in the v4 vLLM Triton attention kernel** is a reproducible source of **~23–25% longer 16K prefill attention kernel time**. Its specific contribution to the observed **5.06% end-to-end 16K serving throughput decline** has not yet been quantified by a serving-level on/off test.

## Executive conclusion

v4 set out to accelerate **16K prefill**, while preserving decode performance. Two different integrations must not be conflated:

1. **Original v4 (#161):** enabled FlagGems attention as the default BI-V150 vLLM attention backend. This replaced attention behavior for **both prefill and decode**, despite the grouped page lookup targeting long-prefill. The first reported 16K serving run showed only **237.83 total tok/s**, **13.99 output tok/s**, duration **9,368.85 s**, and mean TTFT **2,222.23 s**. This was a broad backend-switch regression; these numbers must **not** be attributed solely to the grouped page-index instruction.
2. **Updated v4 (#165):** uses a subclass of the existing vLLM Triton backend, preserves the normal decode forward path, and applies grouped page loads only to eligible long-prefill sequences. Numerical correctness tests passed (five scenarios), but the official 16K serving test still dropped from **861.88** to **818.29 tok/s** (−5.06%). Direct GPU attention-kernel A/B tests now isolate **grouped lookup itself** as substantially slower in this Triton implementation.

**Decision:** The current grouped-lookup implementation in v4 is a failed *performance* optimization on this target. Do not interpret the five passing numerical tests as performance validation. Preserve v1 as the better measured serving candidate until a new 16K prefill approach beats baseline under the official workload.

## Official end-to-end serving evidence

Metrics below are the repository's **mean of the final three successful runs** for each workload, not a single run or the best run. All tests use 1024 output tokens/request and concurrency 64.

| Workload | Metric | Baseline | Updated v4 (#165) | Relative change |
| --- | --- | ---: | ---: | ---: |
| 4K / 256 prompts | Total throughput (tok/s) | 1,881.95 | 1,976.75 | **+5.04%** |
| 16K / 128 prompts | Total throughput (tok/s) | 861.88 | 818.29 | **−5.06%** |
| 16K | Output throughput (tok/s) | 50.70 | 48.14 | −5.05% |
| 16K | Benchmark duration (s) | 2,585.31 | 2,723.02 | +5.33% |
| 16K | Mean TTFT (ms) | 636,512.47 | 675,740.44 | +6.16% |
| 16K | Mean TPOT (ms) | 518.00 | 543.51 | +4.92% |

Sources checked directly in the repository:

- [Baseline summary CSV](../../4k16K_summary_benchmark.csv)
- [Baseline raw runs](../../4k16K_rawrun_benchmark.csv)
- [v4 summary CSV](4k16k_summary.csv)
- [v4 raw runs](4k16k_rawruns.csv)
- [v4 summary](summary.txt)

v4's four 16K serving runs: **815.69, 818.49, 818.35, 818.04 tok/s**. The baseline's four runs were **860.82, 862.06, 861.89, 861.69 tok/s**. The non-overlap makes the regression convincing under the recorded run conditions, although unrelated configuration/environment differences cannot be excluded from summary CSVs alone.

For context, [v1's published summary](../v1/summary.txt) reports **1,209.43 tok/s** on 16K (**+40.32%** vs baseline), and reports a passing MATH-500 evaluation on the tested subset. v4's updated 16K result is **~32.3% lower than v1**. An updated-v4 `math_500` evaluation was **not performed**.

## Direct BI-V150 GPU-event A/B experiment

The following GPU-event results were supplied from two runs of a standalone local probe at `/root/Optimization/v4/probe_v4_prefill_gpu.py` on 2026-10-31. The probe compared original vLLM Triton unified attention against v4 on **the same tensors**, using BF16, page size 16, head size 128, 16 Q heads and 2 KV heads. It warmed up the kernels, checked outputs against original Triton (`rtol=0.03, atol=0.03`) and recorded three GPU-event timing rounds per comparison. **Correctness: PASS** for both reported workloads in both A/B runs.

This is **operator-only GPU time**: no end-to-end model GEMMs, vLLM scheduler, queuing, KV-cache preemption, or serving concurrency.

| Case | Original Triton (ms) | v4 grouping **ON** (ms) | Grouping-ON penalty | v4 grouping **OFF** (ms) | Grouping-OFF penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| **16K full prefill**: Q=16,384, KV=16,384 | 298.441 | **366.124** | **+22.68%** | **298.423** | **+0.11%**¹ |
| **16K chunked prefill**: Q=2,048, KV=16,384 | 70.481 | **88.014** | **+24.88%** | **70.532** | **−0.03%**¹ |

¹ Grouping-OFF percentage uses the **original Triton median from the same OFF experiment**, not the original median from the earlier ON experiment. Those paired OFF-run baseline medians were **298.088 ms** (full) and **70.556 ms** (chunked). The ON-run paired baselines were **298.441 ms** and **70.481 ms**, respectively.

Three-round data as printed by the probe:

| Test | Original Triton rounds (ms) | v4 rounds (ms) |
| --- | --- | --- |
| Full, grouping ON | 297.9593, 298.4411, 298.6918 | 366.6180, 366.1237, 356.9597 |
| Chunked, grouping ON | 70.4167, 70.4960, 70.4812 | 88.0095, 88.0145, 90.6806 |
| Full, grouping OFF | 298.8679, 297.8557, 298.0885 | 298.4231, 298.7467, 298.0530 |
| Chunked, grouping OFF | 70.5897, 70.5562, 70.4246 | 70.5315, 70.5096, 70.5348 |

**Controlled difference:** The second probe run temporarily changed only the `GROUPED_PAGED_LOOKUP` launch specialization from

```python
GROUPED_PAGED_LOOKUP=(
    not use_3d
    # ... existing eligibility conditions ...
),
```

to

```python
GROUPED_PAGED_LOOKUP=(
    False and not use_3d
    # ... identical remaining conditions ...
),
```

in `Optimization/v4/vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_grouped_v4.py`.

Both runs compared with **the same original Triton operator** and passed the output-equality check. With grouping forcibly disabled, the custom kernel's latency became indistinguishable from original Triton (≈0%). This strongly isolates the **grouped lookup specialization** as the source of the standalone prefill kernel slowdown, rather than the surrounding cloned vLLM operator, its original tile choices, or the backend adapter.

The user's shell diagnostic installed a `trap` that restores the original kernel file after the test; the ON/OFF modification was **not a permanent repository change**. The profiler's line `v4 grouped-lookup eligibility (predicted): True` is a **shape-based prediction**: it still prints during the forcibly disabled experiment and is **not proof the grouped kernel was active** in that run.

**Provenance limitation:** The GPU A/B output and probe script were provided/run locally and are not yet included as original log artifacts in this repository. This document transcribes the measurements; it is not a claim that the raw GPU trace or an independent CI run is archived here.

## Why a page-load reduction became slower

The prefill Triton kernel processes KV tiles and computes the physical KV page for each token lane. The original lookup loads the physical page IDs using vector addresses:

```python
physical_block_idx = tl.load(
    block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
).to(tl.int64)
```

The v4 modification loads one page ID for each physical page in the tile and then broadcasts the IDs to token lanes via `tl.gather`:

```python
page_lane = tl.arange(0, TILE_SIZE // BLOCK_SIZE)
physical_pages = tl.load(...)
physical_block_idx = tl.gather(
    physical_pages, offs_t // BLOCK_SIZE, axis=0
)
```

**Measured:** This specialization slows the full and chunked 16K prefill operator by approximately 23–25% on BI-V150.

**Plausible, not proven at instruction level:**

- Repeated page-table addresses may already be served efficiently by the cache / vector-memory load machinery.
- `tl.gather` can add lane movement, register pressure or generated instructions that exceed the cost of the original repeated loads.
- Page-index calculation and an additional per-sequence prefill guard inside the KV loop may add work.

A kernel disassembly / compiler intermediate-code review and per-kernel counters would be needed to distinguish these mechanisms. We currently have **timing causality**, not measured cache-hit, occupancy, memory-traffic or instruction-count evidence. Do **not** claim that any one mechanism has been established.

The earlier `v4_preplay` **FlagGems** attention kernel benefited from grouping (114.5281→97.5166 ms for Q16K/KV16K and 27.8980→23.4170 ms for Q2K/KV16K). Those are **different operator/kernel implementations**, not comparable absolute timings with the vLLM Triton kernel. The positive FlagGems microbenchmark was insufficient to justify the prior whole-backend replacement, and its speedup cannot be assumed to transfer to vLLM Triton.

## Why the end-to-end regression is only ~5%

The 16K official serving metric counts **both prompt and output tokens**, and the full runtime includes prefill attention, decode attention, model GEMMs, other kernels, request scheduling, queuing, and any KV-cache preemption. A ~23–25% slower **prefill-attention kernel** therefore need not produce a 23–25% lower **total serving throughput**.

Both TTFT (+6.16%) and TPOT (+4.92%) worsened in updated v4. Higher TTFT is consistent with more expensive prefill but also includes waiting/queuing; higher TPOT may reflect competition between prefill and decode during continuous batching even if decode kernel code itself is unchanged.

**Do not assert the entire −5.06% serving regression is caused by grouped lookup without the corresponding serving-level ON/OFF A/B**, with all other arguments identical. The GPU-level evidence makes it the leading causal explanation.

## Recommended next actions

1. **Do not ship grouped-lookup v4 as a performance win.** Keep the PR #165 decode-preserving backend structure available for investigation, but do not submit an unmeasured grouped specialization.
2. **End-to-end causal confirmation:** restart the same updated v4 server with grouped lookup compiled OFF; rerun the **unmodified** official 16K benchmark. Compare total tok/s, output tok/s, TTFT, TPOT and KV-cache preemptions against grouped ON and original baseline. Do not edit the competition benchmark script.
3. **If pursuing prefill again:** optimize tile-level KV reuse, Q tiling, GPU occupancy/register pressure or a more efficient scalar/page load **within the Triton operator**, with numerical equivalence checks and GPU-event A/B latency tests first. Do not change the framework's main operator-selection policy as a shortcut.
4. **Protect decode and 4K:** retain inherited original Triton decode and ordinary prefill fallback. Validate mixed prefill/decode workloads.
5. **Only promote candidates after** >baseline 16K *serving* throughput, no material 4K regression and competition `math_500` accuracy ≥0.95.

**Competition compliance:** No quantization, speculative sampling, benchmark script adjustment, unoptimized operator switching, deletion of main selection logic or direct merge of latest FlagOS code is proposed by this report.

## Reproduction reference

Environment with updated v4 Python source:

```bash
cd /root/Optimization/v4
export PYTHONPATH="$PWD/vllm-plugin-FL:$PWD/FlagGems/src:${PYTHONPATH:-}"

# GPU event microbenchmarks — standalone local diagnostic, NOT the official serve benchmark
python3 probe_v4_prefill_gpu.py --case 16k-full-prefill
python3 probe_v4_prefill_gpu.py --case 16k-chunked-prefill
```

Repeat with only the grouped specialization disabled as described above, restarting the Python process after the temporary source change. Restore the source afterward. The probe script is assumed to be present locally; it is **not yet tracked in this repository**.

For the official serving benchmark after starting the standard v4 server, retain the unmodified competition command and cases:

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```
