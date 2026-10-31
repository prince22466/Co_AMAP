# v5 decode optimization and 16K serving bottlenecks — BI-V150 findings

**Date:** 2026-10-31 (analysis; machine-side logs use their own timestamps)  
**Target:** MiniCPM5-2B, Iluvatar BI-V150, vLLM 0.24.0, FlagGems/vLLM-FL v5  
**Competition workloads:** `[[4096,1024,64,256],[16384,1024,64,128]]`  
**Optimization scope:** Triton unified-attention **decode paged-KV physical-page lookup**, not long-prefill optimization.

## Executive findings

1. **The v5 decode kernel is faster in direct, matched-input GPU tests.** All **30/30** tested configurations (4,096/16,384/17,408 KV lengths × batches 1/8/16/32/64 × 2D/3D) passed the benchmark's numerical comparison and were faster than upstream vLLM. Reported attention-call latency reductions range from **10.09% to 23.72%** (unweighted average approximately **14.4%**). Output error printed as `0.00000`, which is *rounded reporting*, not a claim of bitwise identity.
2. **v5 improves the recorded end-to-end 4K and 16K serving metrics over the self-run baseline:** 4K **2,092.46 vs 1,881.95 total tok/s (+11.19%)**; 16K **937.09 vs 861.88 total tok/s (+8.73%)**. These are comparisons of published run summaries, **not a controlled same-process kernel on/off experiment**, and should not be described as fully isolated decode-caused gains.
3. **16K KV-cache capacity is a hard constraint at concurrency 64.** Server allocation was **522,960 KV tokens**. Given the model's measured/config-derived **43,008 bytes/token (42 KiB/token)**, this is approximately **20.95 GiB** of allocated KV storage. At 64 simultaneous 16K prefixes alone, demand would be **1,048,576 tokens (42 GiB)**; after 1,024 generated tokens/request, **1,114,112 tokens (~44.63 GiB)**. With no prefix sharing, roughly **31 full 16K prefixes** or **30 full 17,408-token contexts** fit at once. The actual scheduler can use partial/chunked contexts, so these are capacity bounds, not a prediction of exact instantaneous running-request count.
4. **KV saturation and queueing were observed, not inferred solely from capacity.** During a monitored v5 serving run, `vllm:kv_cache_usage_perc` peaked at **0.9997**, waiting requests peaked at **63**, running requests peaked at **64**, and `vllm:num_preemptions_total` increased by **8**. Those extrema/counter changes span the *whole monitored 4K+16K interval* and are not safely attributable exclusively to 16K. Later 16K server snapshots commonly showed **29–31 running requests**, high cache usage (roughly 93–100%), and a waiting queue. Snapshot prefix-cache hit rate was **0.0%**.
5. **Prefill/decode overlap is a major observed serving symptom.** Engine logs repeatedly reported ~**11–20 generated tok/s** during intervals with ongoing prefill throughput (~1,638 prompt tok/s), compared with ~**243–254 generated tok/s** in some intervals with **0 prompt tok/s**. These are engine-wide, windowed throughput reports—not per-kernel timings or a controlled A/B. They show interference/correlation but do **not** isolate scheduling versus GPU prefill contention versus disabled v5 specialization.
6. **v5's specialization has a significant eligibility limitation.** The code enables `DECODE_PAGE_LOOKUP` only when **`max_seqlen_q == 1`** (plus strict shape/dtype/cache checks). A fused attention call containing any prefill with `max_seqlen_q > 1` falls back to the original page lookup for that invocation, including its decode tokens. The 3D path is also disabled by mixed prefill, missing segment buffers, threshold excess, or batch-invariance settings. **How often this actually occurs in official serving remains unmeasured.**
7. **The 16K workload remains much slower end-to-end than 4K**, despite decode improvement: 16K output throughput **55.13 vs 418.49 tok/s**, mean TTFT **588.44 vs 11.61 s**, and mean TPOT **475.30 vs 139.82 ms**. This is consistent with long-prefill work, KV memory pressure, waiting/preemption and decode contention; there is insufficient evidence to apportion exact causal shares.
8. **v1 remains the stronger published 16K serving candidate.** Its 16K summary reports **1,209.43 tok/s**, versus v5 **937.09 tok/s** (v5 **22.52% lower** than v1), consistent with prioritizing long-context prefill performance rather than assuming decode-only acceleration will dominate total throughput. This comparison is across version summaries; other environment differences have not been ruled out.

## 1. Implementation: what v5 actually changes

The v5 implementation is a **specialization of the existing vLLM Triton unified-attention operator** within `vllm_fl.dispatch.backends.vendor.iluvatar.impl.triton_unified_attention_decode_v5`, exposed through `BIV150DecodeAttentionBackend`. It does not switch to a different attention algorithm or alter the competition benchmark.

Inside a KV tile, upstream code loads a physical-page index for token lanes. Eligible v5 decode uses fewer page-table loads and reuses each loaded page index for the tokens belonging to that physical page:

- **2D decode:** for standard 16-token KV pages and a 32-token tile, load two page entries then distribute them to token lanes (the `tl.gather` path).
- **3D segmented decode:** for a 16-token tile fitting one page, load one page index and broadcast it. Overall 3D attention-call latency includes segmented softmax reduction.
- **Prefill and unsupported shapes:** retain the original vectorized page-table lookup.

The host-side `decode_page_lookup` gate requires: `max_seqlen_q == 1`, 16 query heads, 2 KV heads, head size 128, power-of-two page size compatible with the chosen tile, BF16 Q/K/V, `KVQuantMode.NONE`, and tensor-descriptor mode off. The `use_3d` decision separately requires prepared segment buffers, permitted sequence count, decode-only batch, and no batch-invariance restriction.

**Code references:** [v5 decode kernel](../../Optimization/v5/vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_decode_v5.py), [v5 attention backend](../../Optimization/v5/vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_decode_v5.py), [FlagGems dispatch](../../Optimization/v5/vllm-plugin-FL/vllm_fl/dispatch/backends/flaggems/flaggems.py).

**Runtime verification caveat:** The standalone microbenchmark printed v5 source imports under `/root/Optimization/v5/`, native vLLM under `/usr/local/lib/python3.12/site-packages/vllm/`, and `FlagGems vendor: iluvatar`. Separately, importing the backend in a verified shell selected `BIV150DecodeAttentionBackend`. The serving log excerpt searched for backend-selection text but did not show a matching line; a server-process `PYTHONPATH` capture or explicit backend selection log should still be archived to prove each *serving* measurement used the intended source.

## 2. Matched-input decode microbenchmark (local terminal evidence)

Standalone script: `/root/Optimization/v5/bench_v5_decode.py`; local output: `/tmp/v5_decode_full.csv` and `/tmp/v5_decode_full.log`. **These raw microbenchmark files are not currently archived in this profiling directory.** Timings below are the reported CUDA-Graph-replay attention-call measurements with warmup, `--iterations 40 --repeats 3`; one attention layer, identical paged-KV inputs per comparison. The performance result is **kernel/attention-call evidence, not full serving throughput**.

| KV context | Batch | Mode | Upstream (ms) | v5 (ms) | Latency reduction |
| ---: | ---: | :---: | ---: | ---: | ---: |
| 4,096 | 1 | 2D | 0.733 | 0.571 | 22.07% |
| 4,096 | 1 | 3D | 0.070 | 0.054 | 21.87% |
| 4,096 | 32 | 2D | 1.129 | 0.999 | 11.50% |
| 4,096 | 32 | 3D | 0.798 | 0.695 | 12.98% |
| 4,096 | 64 | 2D | 2.146 | 1.930 | 10.09% |
| 4,096 | 64 | 3D | 1.602 | 1.380 | 13.85% |
| 16,384 | 1 | 2D | 2.873 | 2.224 | 22.57% |
| 16,384 | 1 | 3D | 0.251 | 0.191 | 23.72% |
| 16,384 | 8 | 2D | 3.736 | 3.199 | 14.39% |
| 16,384 | 8 | 3D | 0.746 | 0.659 | 11.56% |
| 16,384 | 16 | 2D | 3.788 | 3.254 | 14.10% |
| 16,384 | 16 | 3D | 1.483 | 1.315 | 11.30% |
| 16,384 | 32 | 2D | 4.480 | 3.893 | 13.10% |
| 16,384 | 32 | 3D | 2.960 | 2.578 | 12.91% |
| 16,384 | 64 | 2D | 8.657 | 7.732 | 10.69% |
| 16,384 | 64 | 3D | 5.917 | 5.137 | 13.19% |
| 17,408 | 1 | 2D | 3.735 | 3.006 | 19.51% |
| 17,408 | 1 | 3D | 0.309 | 0.247 | 20.22% |
| 17,408 | 32 | 2D | 4.756 | 4.122 | 13.35% |
| 17,408 | 32 | 3D | 3.135 | 2.736 | 12.73% |
| 17,408 | 64 | 2D | 8.945 | 7.970 | 10.89% |
| 17,408 | 64 | 3D | 6.274 | 5.451 | 13.12% |

Other batches (8 and 16 at 4K/17,408) also passed and showed lower latency; see the terminal benchmark output or archive the local CSV for all 30 rows. The results do **not** imply the 3D path is necessarily selected at batch 64 in production; 3D eligibility is a separate runtime decision.

A nominal 42-layer model extrapolation of the 16K/B32/3D saving is **42 × (2.960 − 2.578) = ~16.0 ms** of attention time per hypothetical full decode step, *if* per-layer conditions and execution are comparable. This is illustrative only, not measured total-model step time.

## 3. Official-workload serving results

Runs use the unchanged competition workloads. The summaries average the **last three successful runs**, excluding the first; all individual runs in [v5 raw results](4k16k_rawruns.csv) were marked SUCCESS.

| Metric | 4K baseline | 4K v5 | 16K baseline | 16K v5 |
| --- | ---: | ---: | ---: | ---: |
| Total throughput (tok/s) | 1,881.95 | **2,092.46** | 861.88 | **937.09** |
| Output throughput (tok/s) | 376.39 | **418.49** | 50.70 | **55.13** |
| Mean TTFT (ms) | 12,309.34 | **11,609.70** | 636,512.47 | **588,437.58** |
| Median TTFT (ms) | 3,513.29 | 3,356.29 | 687,934.06 | 634,741.13 |
| P99 TTFT (ms) | 74,084.96 | 69,800.41 | 1,233,906.07 | 1,135,598.54 |
| Mean TPOT (ms) | 156.12 | **139.82** | 518.00 | **475.30** |
| Duration (s) | 696.75 | 626.69 | 2,585.31 | 2,377.82 |

Relative to baseline: **4K total throughput +11.19%, 16K +8.73%, 4K mean TPOT −10.44%, 16K mean TPOT −8.24%, 16K mean TTFT −7.55%**.

**Repeatability:** last-three 16K total-throughput runs were **937.14, 937.15, 936.98 tok/s**; the corresponding 4K runs were **2,085.57, 2,041.12, 2,150.69 tok/s**. The very stable 16K numbers suggest reproducibility within this setup; they do not establish causal attribution.

**Comparative status:** [v1](../v1/4k16k_summary.csv) publishes **2,149.82 tok/s (4K)** and **1,209.43 tok/s (16K)**, so v1 remains the strongest published 16K serving candidate. v5 is a validated *decode direction*, not the current best 16K system-wide solution.

**Sources:** [v5 summary](4k16k_summary.csv), [v5 individual runs](4k16k_rawruns.csv), [v5 summary note](summary.txt), [self-run baseline summary](../../4k16K_summary_benchmark.csv), [v1 summary](../v1/4k16k_summary.csv).

## 4. The real 16K bottlenecks: evidence and limits

### 4.1 KV-cache saturation, capacity-limited effective concurrency — **directly observed**

- 42 transformer layers × 2 KV heads × 128 head dimensions × 2 (K and V) × 2 bytes/BF16 = **43,008 bytes/token**, or **42 KiB/token**.
- Reported server GPU KV cache: **522,960 tokens ≈ 20.95 GiB** (not the full nominal device memory).
- At concurrency 64 and context 16,384, full-context KV demand is **64 × 16,384 = 1,048,576 tokens ≈ 42 GiB**.
- At context 17,408, full-context KV demand is **1,114,112 tokens ≈ 44.63 GiB**.
- The scheduler therefore cannot hold all 64 full long contexts simultaneously without sharing or changes in memory budget. In one monitoring interval the KV usage reached **99.97%**, and late 16K logs frequently showed roughly **30 active requests** with a nonempty waiting queue.

**Implication:** Capacity, not simply a slow page-index calculation, is a large determinant of long-context concurrency and completion time. Avoid attributing the entire TTFT to the attention kernel.

### 4.2 Waiting and preemptions — **observed, cause shares not isolated**

- Monitoring over both workload phases captured a peak of **63 waiting requests**, peak **64 running**, and **8** additional `vllm:num_preemptions_total` events.
- Late 16K snapshots showed waiting declining from the 20s to 0 while running stayed around 29–31 and cache usage frequently exceeded 95%.
- The preemption counter is from the full monitoring period. The log excerpt does **not** determine which workload triggered each event or how much wall time they cost.
- High mean/median/P99 TTFT at 16K—**588.4 s / 634.7 s / 1,135.6 s**—is consistent with heavy waiting plus costly prefill, but TTFT alone does not distinguish them.

### 4.3 Prefill/decode competition — **observed correlation**

Selected 16K server log intervals showed:

| Observed interval type | Prompt throughput | Generation throughput |
| --- | ---: | ---: |
| Prefill active, ~30 requests running | ~1,638 prompt tok/s | commonly ~11–20 output tok/s |
| No prompt processing, same general late-run phase | 0 prompt tok/s | some windows ~243–254 output tok/s |

This is a **windowed engine-log contrast**, not a controlled equal-batch experiment; request ages, active set, KV lengths and scheduling states differ. It is consistent with long prefill monopolizing GPU time and starving decode, but **does not prove a particular kernel is the bottleneck**. Investigate GPU busy/idle gaps, per-layer prefill attention, GEMMs, and scheduler step time.

### 4.4 Mixed prefill/decode disables the v5 fast path — **code-confirmed, occurrence unmeasured**

Because `DECODE_PAGE_LOOKUP` is gated on the **invocation-wide** `max_seqlen_q == 1`, a mixed call with any prefill typically falls back to baseline page indexing. Separately, the 3D segmented path is unavailable during mixed prefill or when its other guards fail.

**Important distinction:** This code path explains *why a decode-only microbenchmark might overstate the production benefit*, but without per-invocation tracing we cannot say how many actual decode tokens lose the specialization, nor how much of the measured 16K throughput is caused by that mechanism.

### 4.5 Total tok/s conceals a severe output-generation problem — **measured**

The official 16K run processes **2,097,152 prompt tokens** and **131,072 generated tokens**; its **937.09 total tok/s** includes both. Output throughput is only **55.13 tok/s**. The equivalent 4K output throughput is **418.49 tok/s**, and 16K mean TPOT is **475.30 ms** versus 4K **139.82 ms**.

**Implication:** Inspect output throughput, TPOT/ITL and real decode launch share alongside total throughput; a total-token throughput improvement alone does not establish that long-context generation is fast.

### 4.6 Other candidates: GEMM, launch/scheduler gaps, paged-KV traffic — **not isolated by v5 evidence**

Earlier project profiling identified attention (especially 16K prefill), GEMM/Linear and GPU scheduling gaps as optimization opportunities. The v5 experiment does **not** quantify GEMM time, GPU idle time, CPU scheduling overhead, occupancy, cache bandwidth, or physical-page-lookup share. These remain valid **investigation targets**, not additional root causes proven by the v5 monitoring run.

## 5. Recommended investigation priority (no benchmark or rule changes)

1. **Protect the confirmed v5 decode win.** Retain existing operator selection and standard fallback; keep the 30-case BF16 correctness/performance matrix. Archive `/tmp/v5_decode_full.csv` and its execution metadata in this directory if available.
2. **Prove serving-time code activation.** Capture vLLM EngineCore PID environment (`PYTHONPATH`, plugin variables), dispatch/backend identification, vLLM version and KV-cache block count for every official run. A correct interactive-shell import is not proof that an existing server process loaded the same code.
3. **Separate 4K from 16K scheduler metrics.** Collect time-aligned metrics per official workload: GPU KV usage, running/waiting requests, preemption deltas, prompt/output tok/s, and prefill/decode GPU time. Archive raw logs or CSV traces.
4. **Measure actual 2D/3D and `DECODE_PAGE_LOOKUP` eligibility frequency.** In an isolated diagnostic run (not by editing the competition benchmark), record how many attention invocations/tokens take the v5 fast path, particularly during mixed prefill/decode. Preserve operator-selection logic.
5. **Profile genuine 16K prefill, including chunked prefill and GEMM.** Benchmark prefill attention separately with identical inputs, correlate scheduler steps with GPU activity, then test kernel-level tiling/data-reuse changes only after validating correctness.
6. **Treat KV capacity as a first-class constraint.** Measure effective concurrent full-context sequences and preemption/recompute costs; optimize KV handling at kernel/memory-access level where feasible. Do not misrepresent adjusting benchmark concurrency or parameters as a competition optimization.
7. **Compare a new independent v1+v5 kernel candidate only after isolation tests.** v1 currently leads at 16K; combining valid prefill and decode kernel improvements may help but is not guaranteed. Re-run unchanged official 4K/16K cases and **MATH-500 accuracy** for each promoted candidate.

## 6. Status and methodological limits

- **Kernel result:** verified in supplied local console logs; 30/30 numerical checks reported PASS, no microbenchmark regressions; raw `/tmp` benchmark artifact not committed here.
- **Serving result:** published per-run and summary CSVs support positive *association* of v5 version with 4K/16K throughput improvement over the original reference. The data do not by themselves prove the **entire** gain is caused by the decode lookup instructions.
- **Scheduler result:** observations are from user-supplied terminal/server logs, not yet archived as full raw traces in this directory; counter deltas and extrema span both cases unless split.
- **16K root-cause ranking:** KV-cache saturation and waiting are measured constraints. Prefill/decode contention is strongly suggested by interval logs. The mixed-batch fast-path gate is proven by code; its production cost is **unmeasured**. No exact percent attribution across these mechanisms is currently justified.
- **Accuracy gate:** `summary.txt` still marks **v5 MATH-500 accuracy >=0.95 as TBD**. Kernel-output agreement is not a substitute for the competition's model-level accuracy evaluation.
- **Competition compliance:** No quantization or speculative sampling, no benchmark-script edits, no operator-only switching without optimization, no removal of primary operator-selection logic, and no direct merge of upstream FlagOS changes are proposed.

**Bottom line:** v5 successfully reduces *decode attention* latency, but 16K serving is primarily constrained at the system level by insufficient KV capacity for 64 full long contexts, observable queueing, and strong prefill/decode interference. Measure real mixed-batch activation and prefill cost before attributing the remaining throughput gap to decode itself.
