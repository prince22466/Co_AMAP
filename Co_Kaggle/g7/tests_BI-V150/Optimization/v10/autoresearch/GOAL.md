# v10 autoresearch contract — +6% relative to v9 half-run

**Only allowed execution/edit workspace:** `/root/Optimization/v10`. Do not modify any v0–v9 tree or `/workspace` inference sources. Use v10's inherited, optimized v9 attention kernel as the starting code; no new candidate changes are included in this setup PR.

## Frozen reference and success conditions

From `Co_Kaggle/g7/tests_BI-V150/profiling/v9/summary.txt`, the *reported* v9 two-run half-run means are:

| Case | v9 Total tok/s | Exact +6% threshold | Minimum if displayed to 2 decimals |
| --- | ---: | ---: | ---: |
| 4K `[4096,1024,64,256]` | 2256.31 | 2391.6886 | **2391.69** |
| 16K `[16384,1024,64,128]` | 1518.06 | 1609.1436 | **1609.15** |

**Goal: 4K at least 2391.6886 OR 16K at least 1609.1436 Total tok/s, REPRODUCIBLY, in the unchanged half-run, with all requests successful and correct kernels.** Both workloads are always measured and disclosed. These FIXED thresholds do not shift with candidate selection. v9 figures are reported anchors, not necessarily a properly matched current-machine A/B control; obtain/verify current v10 control to establish causality and disclose run-to-run drift. Independent confirmation matters, particularly for noisy 4K runs.

**Intermediate checkpoint:** repeatable +5% in at least one case versus last accepted matched v10 control -> evidence-backed PR, then continue. Final +6% versus fixed v9 anchor -> final PR even if last incremental gain is under +5%; then stop. Numerical checkpoint / final score without valid request completion, independent confirmation, operator execution, or correctness is only a candidate.

**Accuracy:** MATH-500 deferred to the human operator. Passing kernel tests and these half-runs does NOT claim accuracy >=95% or official full-run competition success.

**Benchmark:** unchanged read-only `/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py` with `RUNS=2`, `SKIP_FIRST=0`, exact two cases. Neither clone nor tune benchmark scripts. Keep original benchmark inputs/serving flags fixed. The comparator `autoresearch/evaluate.py` only validates result CSVs.

## Current priorities (change only with fresh production evidence)

1. 4K **decode** small-M `mm` / `linear`, measured under real serving workloads.
2. Mixed prefill/decode attention execution and page-lookup/data-movement efficiency; isolate production frequency and causal impact.
3. Residual 16K prefill attention memory dependencies, K/V reuse, softmax layout and compiler pipeline *beyond* v9's stage-2 win.
4. KV-cache wait/preemption and GPU gaps, only if new v10 telemetry demonstrates an operator-level addressable bottleneck.

Use the screening funnel in `../AGENTS.md` and exact commands in `../program.md`. Correctness -> matched kernel A/B -> production trace -> short diagnostic serving A/B -> documented go/no-go -> valid two-workload half-run -> independent comparison -> PR. One half-run can take 90–120 minutes; do not begin if <90 minutes remain, and budget more for repeats. Record a real deadline or mark it unknown rather than guessing. No unrelated source edits, quantization, speculative decoding, benchmark optimization, dispatch-only switching, or merge of latest FlagOS.

Record source hashes, original CSVs, production trace, both throughputs, TTFT/TPOT, successful requests, tested shapes, current best SHA, PR URLs and `outstanding_accuracy_check=true` in `.autoresearch-work/session-report.json`. All large artifacts ignored; concise findings tracked.
