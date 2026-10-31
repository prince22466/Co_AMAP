# v5 decode optimization and 16K serving bottlenecks — BI-V150 findings

**Date:** 2026-10-31 (analysis; machine-side logs use their own timestamps)  
**Target:** MiniCPM5-2B, Iluvatar BI-V150, vLLM 0.24.0, FlagGems/vLLM-FL v5  
**Competition workloads:** `[[4096,1024,64,256],[16384,1024,64,128]]`  
**Optimization scope:** Triton unified-attention **decode paged-KV physical-page lookup**, not long-prefill optimization.

## Executive findings

1. **The v5 decode kernel is faster in direct, matched-input GPU tests.** All **30/30** tested configurations (4,096/16,384/17,408 KV lengths × batches 1/8/16/32/64 × 2D/3D) passed the benchmark's numerical comparison and were faster than upstream vLLM. Reported attention-call latency reductions range from **10.09% to 23.72%** (unweighted average approximately **14.4%**). Output error printed as `0.00000`, which is *rounded reporting*, not a claim of bitwise identity.
2. **16K KV-cache capacity is a hard constraint at concurrency 64.** Server allocation was **522,960 KV tokens**. Given the model's measured/config-derived **43,008 bytes/token (42 KiB/token)**, this is approximately **20.95 GiB** of allocated KV storage. At 64 simultaneous 16K prefixes alone, demand would be **1,048,576 tokens (42 GiB)**; after 1,024 generated tokens/request, **1,114,112 tokens (~44.63 GiB)**. With no prefix sharing, roughly **31 full 16K prefixes** or **30 full 17,408-token contexts** fit at once. The actual scheduler can use partial/chunked contexts, so these are capacity bounds, not a prediction of exact instantaneous running-request count.
3. **KV saturation and queueing were observed, not inferred solely from capacity.** During a monitored v5 serving run, `vllm:kv_cache_usage_perc` peaked at **0.9997**, waiting requests peaked at **63**, running requests peaked at **64**, and `vllm:num_preemptions_total` increased by **8**. Those extrema/counter changes span the *whole monitored 4K+16K interval* and are not safely attributable exclusively to 16K. Later 16K server snapshots commonly showed **29–31 running requests**, high cache usage (roughly 93–100%), and a waiting queue. Snapshot prefix-cache hit rate was **0.0%**.
4. **Prefill/decode overlap is a major observed serving symptom.** Engine logs repeatedly reported ~**11–20 generated tok/s** during intervals with ongoing prefill throughput (~1,638 prompt tok/s), compared with ~**243–254 generated tok/s** in some intervals with **0 prompt tok/s**. These are engine-wide, windowed throughput reports—not per-kernel timings or a controlled A/B. They show interference/correlation but do **not** isolate scheduling versus GPU prefill contention versus disabled v5 specialization.


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



## 3. The real 16K bottlenecks: evidence and limits

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
