# v10 Codex autoresearch

The v10 research workspace is **`/root/Optimization/v10` in the existing Remote-SSH IDE**. The repository mirror is `Co_Kaggle/g7/tests_BI-V150/Optimization/v10`. No separate CLI/SSH bridge is required.

Read [AGENTS.md](../AGENTS.md), [GOAL.md](GOAL.md), [program.md](../program.md), and [EXPERIMENTS.md](EXPERIMENTS.md). v10 begins with the v9 two-stage BF16 GQA prefill attention improvement and mixed-paging regression test. The former v9 half-run summary is a reported *performance reference*, not independently verified matched raw control data.

## Success means +6% of fixed v9 half-run

The unmodified v9 half-run reported 4K **2256.31**, 16K **1518.06** Total tok/s. v10 must reproducibly reach **>=2391.6886 in 4K OR >=1609.1436 in 16K** (display as 2391.69 / 1609.15). Measure both cases, with two complete successful raw runs per case. The goal stays fixed even as v10 progresses. For causal testing, compare candidate to the last accepted v10 control using truly matched raw CSVs; get a fresh starting v10 baseline if available historical CSVs are inadequate.

Every repeatable **>=5% gain relative to the last accepted matched v10 baseline** warrants a progress PR, even below the final +6% goal. Smaller incremental gains can accumulate. Final +6% goal merits a final PR and stop even if the last improvement is less than 5%. Reproducibility, correctness and genuine kernel execution are mandatory; one positive numeric average is not enough.

## Work priorities

- Current #1 hypothesis: **4K decode small-M GEMM/Linear**, after profiling v10 production shapes and actual time share. Historic v7 4K decode Linear dominance was isolated GPU evidence, not an asserted current end-to-end bottleneck.
- #2: **mixed prefill/decode attention**, including loss of pure-decode page-lookup fast path in mixed calls. Measure how often mixed calls occur and test actual kernel/data-movement fixes.
- #3: **16K prefill attention** residual data dependency, shared-memory conflicts, and K/V reuse while preserving v9 stage-2 improvement.
- #4: KV-cache pressure, preemption, CPU/GPU idle only when current telemetry supports an operator-level remedy. Re-rank targets after the first new serving trace.

Use a short, profiled, exact-input, correctness-gated A/B funnel instead of immediately running long scored tests. A whole two-case half-run historically takes **1.5–2h**; require >=90 min available to START, and reserve ~300+ min for a fresh baseline plus repeats. Keep timestamped budget ledger. Do not adjust test workloads, serving parameters, framework dispatch or benchmark code to improve score.

## Artifacts and validation

- Dedicated vLLM server and read-only provided half-run: see [program.md](../program.md).
- Read-only CSV checker: `python3 autoresearch/evaluate.py --baseline-summary ... --baseline-raw ... --candidate-summary ... --candidate-raw ...`. It checks shapes, exactly two successful full-request raw repetitions, summary/raw agreement, both throughputs and TTFT; it distinguishes `numeric_5pct_gain_observed` (current control) from `numeric_6pct_vs_v9_goal_observed` (fixed reported v9 reference).
- `python3 -m unittest discover -s autoresearch/tests -v` verifies CSV comparison logic without a GPU.
- GPU attention/decode/mixed-paging tests plus any modified FlagGems MM/Linear tests are necessary before promotion.
- Write logs, CSV, traces and scratch probes under ignored `.autoresearch-work/`. Log concise hypothesis/outcome in [EXPERIMENTS.md](EXPERIMENTS.md) and post source SHA plus evidence links in each PR. Never auto-merge.

MATH-500 and official full-run evaluation are human-owned downstream checks. The development half-run does not establish either. Codex in the IDE is an interactive agent and does not start unattended merely by checking in this framework.
