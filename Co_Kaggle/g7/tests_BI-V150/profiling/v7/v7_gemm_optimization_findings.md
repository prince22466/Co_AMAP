# v7 GEMM optimization findings — production shapes, kernel gains, and 16K bottlenecks

**Updated:** 2026-11-03  
**Model/hardware:** MiniCPM5-2B / Iluvatar BI-V150 / FlagGems / vLLM-FL  
**Competition workloads:** `[[4096,1024,64,256],[16384,1024,64,128]]`  
**Status:** post-merge technical findings for [v7 PR #177](https://github.com/prince22466/Co_AMAP/pull/177). Operator-level gains **confirmed**; a competitive end-to-end v0→v7 throughput gain **not confirmed**.

## Executive summary

1. **v7 improves real large-M GEMM, but not every shape.** For the four BF16 MM shapes observed during MiniCPM serving at `M=2048`, sum of per-shape median kernel times fell from **2.6516 to 2.4545 ms** (**7.43% less GEMM time**, under one-call-per-shape weighting). The wide `(2048,2048,12288)` case improved by **17.79%** in latency; `(2048,6144,2048)` regressed by **5.91%**.
2. **Grouped MM tile ordering is the primary large-shape win.** Forcing `GROUP_M=1` *inside the same v7 operator* eliminates most of the wide-projection benefit: `(2048,2048,12288)` measured **1405.57 µs** with group 1 versus **1150.38 µs** with group 8 (separate GPU-timing runs). Other shapes regress under grouping; a selective policy remains an experiment, not committed code.
3. **These MM kernels do execute in vLLM serving.** Two rank-0 GPU traces contain **336** and **1008** `mm_kernel` calls. The sum of the four measured v7 MM latencies predicts their cumulative GPU time closely (206.18 vs 206.29 ms; 618.53 vs 618.25 ms). This supports representativeness of the measured shapes, but is not proof of per-invocation shape or source-commit attribution.
4. **Long-prefill attention is the major measured GPU compute bottleneck.** The v0 16K prefill diagnostic attributes **93.19%** of summed GPU kernel time to attention; a v7-heavy rank-0 trace attributes **94.2%**. The estimated benefit of the observed MM acceleration is only ~**0.37%** of summed GPU kernel time in that latter trace. This is not an end-to-end throughput estimate.
5. **KV-cache capacity and waiting are confirmed in earlier v5 serving telemetry**, not in a loaded v7 experiment: 522,960 allocated KV tokens, peak usage 99.97%, max 63 waiting requests and 8 preemptions over a monitored combined workload. A v7 monitor showing all zeros ran without an established active load; it does **not** refute v5 saturation.
6. **No serving win has been demonstrated.** The two-run v7 *development half-run* averages were **1634.43 tok/s (4K)** and **854.05 tok/s (16K)**, below the separately optimized v6 half-run; this is **not a matched v0/v7 A/B** and does not replace the unmodified official benchmark.

## 1. Scope, architecture, and implementation

v7 is derived from v0 and modifies existing Iluvatar FlagGems operator implementations; it does not switch framework backends, modify model hyperparameters, enable quantization or speculative decoding, or change competition benchmark scripts. [PR #177](https://github.com/prince22466/Co_AMAP/pull/177) is merged; this report documents follow-up measurements, not a new kernel patch.

- [`mm.py`](../../Optimization/v7/FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/mm.py): Triton `mm_kernel` and `_launch_mm`. A 2-D output tile grid is mapped from program IDs. The **v7 launch policy** is `GROUP_M=1 if M <= 128 else 8`. For `GROUP_M=1`, it uses row-major `pid_m=pid//grid_n`, `pid_n=pid%grid_n`; for group 8, it groups up to eight M-tiles per program-ID group across N-tiles. It retains original autotuning/dispatch, split-K handling, dtype/precision pathways, boundary masking and alignment hints.
- [`linear.py`](../../Optimization/v7/FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/linear.py): preserves grouped Linear tiling (`GROUP_M=8`) and bias behavior; adds M/N edge masks and feeds the fp32 accumulator to `tl.dot(a,b,acc,allow_tf32=False)`. The dominant code path may differ between eager/compiled and prefill/decode executions.
- [v7 implementation overview](../../Optimization/v7/GEMM_LINEAR_V7.md) gives the original proposal and validation plan. Its pre-test status text is historical; the results below supersede that status.

### Matrix structure observed in serving

The reversible launch-time instrumentation records each **distinct** `(op,M,K,N,dtype,A/B strides)` **once per worker process**, **not call frequency**. It reported **146 unique records**: **144 MM** across **four projection families** and **two Linear** shapes. [Shape inventory](v7_observed_gemm_shapes.csv) transcribes the user-reported table into a machine-readable file.

| Operator | M observed | K | N | Tensor layout | Meaning supported by evidence |
|---|---|---:|---:|---|---|
| MM | 2048 and 1–256 (multiples of 8 plus 4, 2, 1) | 2048 | 12288 | A=(K,1), B=(1,K) | Wide projection; large-M and smaller-batch invocations |
| MM | same | 2048 | 2560 | same | Projection |
| MM | same | 2048 | 2048 | same | Projection |
| MM | same | 6144 | 2048 | same | Reduction from wider K |
| Linear | 1 and 256 | 2048 | 130560 | A=(2048,1), B=(1,2048) | Likely vocabulary/output projection; layer identity not traced |

All these records report `torch.bfloat16` and transposed/column-major B. `M=2048` is compatible with chunked prefill, but the shape recorder alone does **not** prove the scheduler's precise chunk-size setting, count of calls at each M, or phase attribution. Real shape frequency/phase must be collected separately if needed.

## 2. Correctness and production-path validation

- BI-V150 direct vendor operator suite: **58 passed, 0 failed, 3 warnings, 143.90 s**, from `python3 -m pytest -q -s tests/test_iluvatar_gemm_linear_v7.py`. Warnings concerned deprecated `use_cuda_graph` and event-timing fallback when the replay benchmarker was unavailable; neither was an assertion failure.
- Imports of both `flag_gems.runtime.backend._iluvatar.ops.mm` and `...ops.linear` resolved under `/root/Optimization/v7/FlagGems/src/` in the test environment. That establishes import provenance, not by itself server-process provenance.
- Rank-0 traces captured `mm_kernel`, `linear_kernel`, and `kernel_unified_attention` executions. The accompanying `async_llm` traces had CPU `python_function` events and zero GPU kernel events; that is a **trace ownership**, not a no-GPU-execution, result.
- **No v7 `math_500` evaluation** or complete unchanged-official serving v0/v7 benchmark result is archived here. Passing direct BF16/FP16 operator tests cannot substitute for model-level accuracy validation.

### Two rank-0 GPU traces (user-reported; profiling is diagnostic)

| Metric | Trace A | Trace B |
|---|---:|---:|
| Summed GPU kernel time | 1103.549 ms | 13472.449 ms |
| Attention | 862.32 ms (78.1%), 84 calls | 12693.03 ms (94.2%), 252 calls |
| GEMM/Linear | 208.81 ms (18.9%) | 676.43 ms (5.0%) |
| `mm_kernel` | 206.29 ms, 336 calls | 618.25 ms, 1008 calls |
| `linear_kernel` | 2.49 ms, 3 calls | 57.54 ms, 70 calls |
| Approx. `mm_kernel` time/call | 613.96 µs | 613.34 µs |

GPU kernel-time shares are **summed durations**, not full vLLM wall-clock time, request latency, occupancy, bandwidth, queueing or CPU/GPU idle gaps. Both sets give exactly four MM calls per attention call; mapping each to the four M=2048 shapes is strongly suggested by timing but remains **aggregate inference**.

## 3. Serving development benchmarks: no established end-to-end gain

Both v6 and v7 published two-run averages under a locally named `benchmark_throughput_serve_half.py` development script. The v7 raw runs are [here](4k16k_rawruns%28half_run%29.csv), summary [here](4k16k_summary%28half_run%29.csv); the v6 comparison is [here](../v6/4k16k_summary%28half_run%29.csv). The names and run counts are reported as recorded; **do not substitute these results for unchanged official scoring**.

| Workload (1024 decode, C=64) | v6 total tok/s | v7 total tok/s | v7 relative to v6 | v6 mean TTFT | v7 mean TTFT |
|---|---:|---:|---:|---:|---:|
| 4K; 256 prompts | 1815.76 | 1634.43 | -9.99% | 15,622.11 ms | 15,585.93 ms |
| 16K; 128 prompts | 902.94 | 854.05 | -5.41% | 605,222.89 ms | 640,304.64 ms |

All four v7 raw trials report success, but 4K individual total-throughput runs were **1464.75** and **1804.12 tok/s**—a substantial spread. v6 is a **different optimization candidate**, not v7's parent; a v6→v7 percentage is *not* a controlled estimate of the change from v0. The archived v0 baseline [official summary](../../4k16K_summary_benchmark.csv) was run under a different comparison protocol and should not be silently combined with the v7 half-run. Official, same-environment v0/v7 ON/OFF testing remains open.

## 4. Operator-only A/B experiments on BI-V150

Tests ran BF16 with B transposed; user supplied median GPU timings (7 repeats × 40 iterations in the MM experiment). Both versions used separate Python processes and passed the probe's correctness check. The backend lacked a replay benchmarker and used event timing. `v7_gemm_ab_results.csv` stores **34 paired comparisons** in microseconds, including the non-production proxy shapes, production shapes, and forced-group-1 control; medians in different experiment groups must not be conflated as a paired simultaneous experiment.

### 4.1 Original proxy-shape MM and Linear tests (exploratory)

| MM shape (M,K,N) | v0 µs | v7 µs | v7 speedup |
|---|---:|---:|---:|
| (1,2560,2560) | 44.91 | 44.82 | 1.002× |
| (32,2560,6912) | 75.21 | 76.05 | 0.989× |
| (64,2560,2560) | 38.33 | 37.93 | 1.011× |
| (256,2560,2560) | 94.50 | 99.33 | 0.951× |
| (2048,2560,2560) | 387.99 | 373.67 | 1.038× |
| (4096,2560,6912) | 1907.02 | 1677.55 | 1.137× |

| Linear shape (M,K,N) | v0 µs | v7 µs | v7 speedup |
|---|---:|---:|---:|
| (1,2560,2560) | 27.12 | 27.02 | 1.004× |
| (8,2560,2560) | 27.77 | 27.60 | 1.006× |
| (32,2560,6912) | 75.22 | 76.66 | 0.981× |
| (64,2560,2560) | 48.71 | 46.84 | 1.040× |
| (128,2560,2560) | 39.65 | 38.57 | 1.028× |

The largest exploratory MM result (M=4096,K=2560,N=6912) was **1.137×** faster, but `K=2560/N=6912` are **not** among the captured production MM projection families; therefore it must not be substituted for a serving-shape speedup. Linear improvements were generally small and one proxy shape regressed.

### 4.2 Four real M=2048 MM shapes — v0 vs v7 default

| M,K,N | v0 median (µs) | v7 median (µs) | v7 latency change |
|---|---:|---:|---:|
| 2048,2048,12288 | 1399.27 | 1150.38 | -17.79% |
| 2048,2048,2560 | 298.35 | 297.79 | -0.19% |
| 2048,2048,2048 | 235.87 | 245.73 | +4.18% |
| 2048,6144,2048 | 718.10 | 760.58 | +5.91% |
| **Sum, once per shape** | **2651.60** | **2454.48** | **-7.43%** |

One-call-per-shape sum is an **illustrative per-layer kernel-time total**, not a timing of an entire model layer or a distribution-weighted throughput metric. The largest win occurs for the wide N=12288 projection; two narrower shapes regress.

### 4.3 Representative real small-M MM shapes — v0 vs v7 default

| M,K,N | v0 median (µs) | v7 median (µs) | v7 latency change |
|---|---:|---:|---:|
| 1,2048,12288 | 80.12 | 80.20 | +0.10% |
| 8,2048,12288 | 86.03 | 88.32 | +2.66% |
| 32,2048,12288 | 86.82 | 86.55 | -0.31% |
| 64,2048,12288 | 92.26 | 93.29 | +1.12% |
| 128,2048,12288 | 159.63 | 158.08 | -0.97% |
| 256,2048,12288 | 258.66 | 212.20 | -17.96% |
| 128,2048,2560 | 53.18 | 53.03 | -0.28% |
| 256,2048,2560 | 70.18 | 75.30 | +7.30% |
| 128,2048,2048 | 31.11 | 30.55 | -1.81% |
| 256,2048,2048 | 54.77 | 54.61 | -0.29% |
| 128,6144,2048 | 88.35 | 88.08 | -0.31% |
| 256,6144,2048 | 175.12 | 165.73 | -5.36% |

The M=256 wide projection improves by **17.96%** in latency, but the M=256, K=2048, N=2560 case regresses **7.30%**. At small M many variations are ~1–3% or include obvious event-timing outliers; their significance requires paired repeated testing.

**Accuracy caveat:** The probe reported `max_abs_error=1.0` against PyTorch for the **two** small-M `K=6144,N=2048` shapes at M=128 and 256 **in both v0 and v7**; the probe still printed PASS. This is not evidence of a v7-specific error, but the absolute-error distribution, relative error and BF16 tolerance should be checked. Do not interpret PASS as bitwise equality.

### 4.4 Controlled implementation change: v7 forced `GROUP_M=1`

The operator file was temporarily changed to pass `GROUP_M=1` and restored using an EXIT trap. This **does not** change the framework operator-selection logic. The comparator below is the previous v7 default `GROUP_M=8` for M>128 (separate run), versus forced group 1.

| Real shape M,K,N | v7 group 1 (µs) | v7 group 8 (µs) | group 8 latency change |
|---|---:|---:|---:|
| 2048,2048,12288 | 1405.57 | 1150.38 | -18.16% |
| 2048,2048,2560 | 298.02 | 297.79 | -0.08% |
| 2048,2048,2048 | 245.68 | 245.73 | +0.02% |
| 2048,6144,2048 | 723.28 | 760.58 | +5.16% |
| 256,2048,12288 | 278.67 | 212.20 | -23.85% |
| 256,2048,2560 | 76.40 | 75.30 | -1.43% |
| 256,6144,2048 | 176.59 | 165.73 | -6.15% |
| **M=2048: sum of four** | **2672.55** | **2454.48** | **-8.16%** |

In the same forced-group-1 session, the **v0 four-MM sum was 2668.08 µs**, so v7 without grouping was essentially equal to v0 at this grouped measure. The wide-projection result isolates grouped ordering as a major contributor to v7's improvement. Nevertheless, different group=8/group=1 sessions mean this is **strong operator-level causal evidence, not a randomized interleaved A/B**.

Two residual regressions **persist when group 1 is forced**: (2048,2048,2048) measured 236.03 µs in v0 vs 245.68 µs in group-1 v7; (256,2048,2560) measured 70.98 vs 76.40 µs. These cannot be blamed solely on grouped ordering. Candidate causes such as emitted instructions, edge handling or tuner selection have **not** been isolated.

### 4.5 Reconciliation with actual GPU traces

The four v7 default M=2048 kernel medians sum to **2.454480 ms**. At four MM calls per attention invocation:

| | Trace A (84 attention calls) | Trace B (252 attention calls) |
|---|---:|---:|
| Predicted summed MM time, 4 shapes × calls | 206.176 ms | 618.529 ms |
| Observed summed `mm_kernel` time | 206.290 ms | 618.250 ms |
| Hypothetical v0→v7 MM saving, identical MM calls | 16.558 ms | 49.673 ms |
| Saving / observed total GPU kernel time | 1.50% | 0.37% |

This close reconciliation links microbenchmarked GEMM to real serving GPU traces. The hypothetical savings assume the same call shapes/counts and ignore scheduling, overlaps, compile/setup cost and other changes; they are **not** independently measured v0/v7 serving speedups.

## 5. Confirmed bottlenecks, evidence scopes, and open hypotheses

| Finding | Evidence and scope | What is **not** established |
|---|---|---|
| **16K prefill attention dominates GPU kernel time** | [v0 16K diagnostic](../v0/profile_runs/hot_kernels_1.csv): attention **93.1942%**; v7 heavy rank-0 trace **94.2%** | Share of entire wall-clock time; exact memory vs compute sub-bottleneck |
| **16K decode splits time between attention and GEMM** | [v0 decode diagnostic](../v0/profile_runs/hot_kernels_2.csv): attention **52.3274%**, Linear **40.3741%** | Same distribution in every mixed serving schedule |
| **4K decode GEMM/Linear matters** | [v0 4K decode diagnostic](../v0/profile_runs/hot_kernels_4.csv): Linear **64.5015%**, attention **23.9183%** | Full 4K serving bottleneck; [4K prefill diagnostic](../v0/profile_runs/hot_kernels_3.csv) was prefix-cache-sensitive |
| **16K KV capacity is insufficient for all 64 complete contexts without sharing** | [v5 investigation](../v5/v5_decode_16k_bottleneck_findings.md): **522,960 KV tokens** allocated, ~**43,008 bytes/token**; 64 × 16,384 = **1,048,576** prefix tokens (before generation) | A precise maximum sustainable batch size with partial contexts, prefix sharing or different config |
| **KV-cache saturation and waiting occurred in v5 serving** | v5 monitor: max cache use **99.97%**, **63 waiting**, **8 preemptions** across combined 4K+16K monitoring; later 16K snapshots ~29–31 running with waiting | Attribution of every reported peak/preemption exclusively to 16K, or direct proof of same symptoms in v7 |
| **Prefill/decode throughput interference is observed** | v5 engine windows: ~**11–20 output tok/s** alongside ~1638 prefill tok/s, versus some ~**243–254 output tok/s** windows with zero prefill activity | Proof whether causation is scheduler, GPU attention pressure, cache thrash, or missing decode specialization |
| **v7 GEMM contributes little to attention-heavy GPU time** | v7 trace B: MM/Linear **5.0%**, attention **94.2%**; actual-shape MM speedup predicts ~**0.37%** kernel-time saving | Entire run throughput improvement or the limit under every workload |
| **v4 grouped paged-KV lookup regressed 16K attention** | [v4 investigation](../v4/v4_performance_regression_findings.md): isolated ~**23–25%** slower prefill attention kernel with grouped lookup, despite separate FlagGems microbenchmark gains | That grouped **GEMM** tile ordering is bad: it is a different kernel and optimization |

### Still unproven / do not label as confirmed bottlenecks

- GPU idle gaps, CPU launch latency and vLLM scheduler synchronization cost have **not** been quantified for the v7 workload.
- Within 16K attention, dominant **memory traffic vs compute vs register pressure/occupancy** remains to be isolated with kernel-level profiling; attention's time share alone cannot determine its hardware bottleneck.
- The standalone v7 KV monitoring script produced zeros while no concurrent 16K benchmark was established. This is an *idle or missing-load observation*, not evidence that v7 solved cache saturation.
- Per-shape invocation frequency, actual server-process source SHA, and full official v0/v7 end-to-end throughput/accuracy comparison remain unverified.

## 6. Decision and reproducible next tests

**Decision:** Keep v7's existing grouped-MM implementation as a **shape-dependent, operator-validated optimization**, not a proven competition throughput win. Do not assume a blanket `GROUP_M=8` policy is best for narrower shapes. Prioritize the measured attention bottleneck and 16K KV capacity effects ahead of generic GEMM tile tuning.

1. **Keep code unchanged until a paired candidate proves better.** In a *new* kernel-only experiment, test grouped ordering selectively for wide (M,K,N) cases and group 1 for (2048,6144,2048); use the same operand shapes/layout and stable temperature/cache state, alternating A/B measurement order. Verify all relevant BF16/FP16 outputs and the `K=6144` maximum-error discrepancy. Do not change framework operator-selection logic.
2. **Profile attention under uncached 16K prefill and realistic mixed prefill/decode.** Separate Q/KV tiling, memory traffic, occupancy and launch/scheduling gaps; compare kernels with matched tensors before considering a serving-level acceptance test.
3. **Monitor live 16K load.** Capture `vllm:kv_cache_usage_perc`, running/waiting, preemptions and throughput *while* the unmodified 16K test is active, so v5's KV/scheduling evidence can be checked under v7.
4. **Validate any promoted candidate** using the competition's **unchanged** `benchmark_throughput_serve.py` and exact `[[4096,1024,64,256],[16384,1024,64,128]]` workloads, matched v0/v7 server configs, plus the unchanged `math_500` accuracy evaluation. No benchmark-script tuning, quantization, speculative sampling, backend switching without operator optimization, or wholesale FlagOS branch merges.

### Reproduce official benchmark (after removing diagnostic instrumentation and profiling)

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

## 7. Sources and provenance

- **Repository evidence:** [v7 development raw runs](4k16k_rawruns%28half_run%29.csv), [v7 development summary](4k16k_summary%28half_run%29.csv), [v6 development summary](../v6/4k16k_summary%28half_run%29.csv), [v0 diagnostic traces/kernels](../v0/profile_runs), [v4 root-cause report](../v4/v4_performance_regression_findings.md), [v5 root-cause report](../v5/v5_decode_16k_bottleneck_findings.md), [v7 source implementation](../../Optimization/v7/GEMM_LINEAR_V7.md).
- **User-reported BI-V150 experiment outputs:** [PR #177 58-pass test / module validation](https://github.com/prince22466/Co_AMAP/pull/177#issuecomment-6574954807), [early MM A/B](https://github.com/prince22466/Co_AMAP/pull/177#issuecomment-6602478620), [real-shape A/B](https://github.com/prince22466/Co_AMAP/pull/177#issuecomment-6603185988), [grouped-ID control experiment](https://github.com/prince22466/Co_AMAP/pull/177#issuecomment-6603291017). Full sample-level output was provided in the optimization chat; the accompanying CSV here records medians (not the individual event repeats).
- **Machine-readable tables created for this report:** [A/B measurements](v7_gemm_ab_results.csv) and [once-per-shape inventory](v7_observed_gemm_shapes.csv). The shape inventory is reconstructed from the user-reported report listing; it is not a time-weighted count, nor an independently recollected raw server log.
- **Verification limits:** GPU timings, correctness result logs, v7 trace summaries and shape dump were reported from the user's BI-V150 container; they were not independently re-executed by this PR author. Claims labeled “confirmed” mean directly observed in those supplied measurements, not separately reproduced in CI.

