# Codex instructions — MiniCPM5-2B BI-V150 v9

Read program.md, autoresearch/GOAL.md and autoresearch/README.md before editing code. You are running in the Codex IDE extension attached to the authorized Remote-SSH machine.

## Workspace: STRICT v9 ONLY

- Remote, the ONLY execution and source-editing workspace: /root/Optimization/v9
- Repository/Windows local mirror: Co_Kaggle/g7/tests_BI-V150/Optimization/v9 (g7\tests_BI-V150\Optimization\v9 relative to Co_Kaggle)
- All inference changes, tests, notes and research tooling MUST stay within v9. Do NOT edit, overwrite, rsync into, or benchmark from v1-v8, other versions, /workspace/vllm-plugin-FL or /workspace/FlagGems.
- The provided half-run script is READ-ONLY at /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py. It is intentionally outside v9 and must be invoked there without copying or editing it.
- No separate v9_autoresearch scratch tree and no local Codex CLI -> SSH bridge. IDE Codex runs on the remote host directly.
- Temporary logs/CSV/profiler outputs MUST live under v9/.autoresearch-work (ignored by Git). Do not check them in.

## Exact development milestone

Using the UNMODIFIED provided half-run, reach ONE of:
- 4K [4096,1024,64,256]: Total tok/s >= 2450
- 16K [16384,1024,64,128]: Total tok/s >= 1450

Both is better but not required. Run BOTH workloads every time. The provided half script has RUNS=2 and SKIP_FIRST=0; do not change this. These half-run numbers are DEVELOPMENT milestones, not official full-run competition results. MATH-500 is explicitly deferred to the human operator; do not run it or claim accuracy acceptance.

Stop ONLY on a reproducible milestone, a user-specified deadline, or a hard blocker. An **observable progress improvement** means **>=5% repeatable Total tok/s increase in at least one workload against the last accepted best version's fresh matched half-run**. Commit ONLY v9 changes and open an evidence-backed GitHub PR at each such checkpoint, even below the final milestone; then continue from that accepted best. Improvements <5% belong in the experiment log and may accumulate until a >=5% checkpoint. If a candidate reproducibly reaches the 2450/1450 end goal, open a final PR and stop even if its last incremental gain was <5%. Never auto-merge. Numeric change alone is not repeatability.

User-reported initial v9 HALF-run reference: **4K 2130** and **16K 1240 Total tok/s** (the second "4K" label was interpreted as 16K; verify from raw CSV). These are planning references, not automatically verified matched A/B controls. A +5% checkpoint relative to those figures would be **4K >=2236.5** or **16K >=1302**. Historic v8 FULL-run: 4K 2331.29 / 16K 1251.22; NEVER mix full-run and half-run baselines.

## Inviolable competition constraints

- Model: MiniCPM5-2B. Use provided FlagOS vllm-plugin-FL and FlagGems, with inference code loaded from /root/Optimization/v9 ONLY.
- Never edit the half benchmark, official benchmark, dataset or evaluator; never tune scripts for higher scores.
- Do not enable quantization or speculative decoding by hyperparameters; do not change model/generation config, concurrency, batching, serving flags, graph mode or test shapes to manufacture a result.
- No gains obtained merely by selecting a different operator without genuine operator optimization; preserve normal framework dispatch and fallback.
- Do not merge latest upstream FlagOS framework code for the purpose of optimization.
- Every accepted change must be a concrete kernel/operator/data-movement improvement with equivalent semantics, verified by relevant correctness tests and production invocation evidence.
- Never touch unrelated services, process groups, sources, credentials or model files. Do not force-push main.

Competition: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn

## Prioritized research queue

1. 16K attention: global-memory dependency chains, shared-memory/register footprint, paged-KV addressing, K/V and Q tile reuse, occupancy (highest potential).
2. Mixed prefill/decode interference: causal phase profiling, kernel-level remedies without workload/scheduler gaming.
3. 4K attention and prefill GEMM: locality, grouped M tiling, tl.dot reuse, efficient address calculation/accumulation.

Primary v9 sources:
- vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py
- vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150.py
- FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/mm.py
- FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/linear.py
- vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py
- vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py

Prior unsuccessful v4 grouped lookup, v6 KV tiling, v7 GEMM stacking are negative evidence; require a new profiler-backed hypothesis before reattempting. Use a real serving trace to confirm the edited kernel executes.

## Mandatory time-aware funnel — DO NOT half-run every candidate

**Cost observation:** ONE unchanged 4K+16K half-run can take approximately **1.5–2 hours** on this BI-V150 (operator report). This is an expensive *final-stage* measurement, not a probe. A baseline plus a candidate may consume **3–4 hours** before repeats. Keep a timestamped time-budget ledger in v9/.autoresearch-work, including deadline, elapsed time, remaining time, projected GPU run duration and a reserve for validation/PR. The 90-minute remaining-time floor permits a high-confidence candidate to start, but a half-run may take up to ~120 minutes, so a run begun with only 90 minutes left may exceed the deadline. Record the risk and preserve results; do not claim completion or repeatability if interrupted.

Use these **fast gates, in order, for EVERY hypothesis**; stop early on failure, record why and try the next idea:

1. **Static/causal check.** Identify the production operator, supported shapes and estimated fraction of serving GPU time it affects (16K attention, mixed attention or 4K attention/GEMM). Profile actual v9 execution. Verify the implementation is a genuine operator/data-movement optimization; no dispatch/hyperparameter tricks. Estimate whether the observed kernel gain could plausibly improve end-to-end throughput by >=5% (use Amdahl reasoning, not microbench speedups alone). Reject a change with negligible attributable impact.
2. **Correctness/robustness first.** Run existing v9 GPU functional tests for attention and decode (see program.md), and focused tests for the modified operator including relevant 4K/16K shapes, prefill/decode, paging boundaries, mixed-batch behavior, masks and fallback cases. Compare outputs to the unchanged reference with the test's established tolerances. Fail fast on mismatches, invalid accesses, compilation errors, NaNs or excessive resource use. Do not bypass, weaken or rewrite correctness tests to pass.
3. **Paired microbench / profiler A/B.** Use standalone diagnostic probes or available v9 profiling tools, never the competition benchmark script. Compare same exact kernel inputs, dtype, cache layouts, launch environment and GPU state for baseline versus candidate. Warm up, perform multiple repetitions, report medians and variability, memory traffic/latency, spills, shared-memory/register footprint and occupancy where available. Check the modified kernel actually executes. Reject an ambiguous, noisy, or regressing kernel change before serving.
4. **Short end-to-end serving A/B smoke.** In the dedicated v9 server, run the short **diagnostic** commands in program.md on the same baseline and candidate, preferably more than once. Verify request success, no obvious TTFT/TPOT regression, no CPU/GPU stalls or prefill/decode interference. This only screens for plausibility: reduced request counts/output lengths and concurrency are NOT the official workload, do not substitute their tok/s for the half-run score.
5. **Go/no-go decision.** Promote ONLY an error-free candidate with a clear, repeatable relevant kernel speedup, production invocation proof and plausible overall serving upside based on measured bottleneck share plus quick A/B behavior. Write the evidence, time spent, remaining budget and rationale in EXPERIMENTS.md. If evidence is weak or no time remains, log/revert selectively and test another hypothesis. No speculative 2-hour full half-run "just to see."

**Time guard:** budget **up to ~20–30 minutes for the initial quick-screening phase** per candidate (not a benchmark guarantee), and require **>=90 minutes remaining** before starting any new 4K+16K half-run (operator-selected hard start threshold; NOT enough to guarantee completion of an observed 90–120-minute run). If confirmatory repetitions or PR preparation are needed, reserve additional time: prefer **>=300 minutes** for a candidate plus matched confirmation, and do not claim statistical validation if time runs out. An already verified historical v9 *half-run* baseline can be reused only when exact code, machine, model, serving parameters, benchmark script and conditions are documented and genuinely comparable; otherwise obtaining a fresh baseline itself consumes another 1.5–2h and must be budgeted. Never use historical v8 FULL-run figures as a development baseline.

The first half-run baseline is the deliberate exception to the candidate quick-screening gates only if reliable matching v9 half-run CSVs do not already exist. Subsequent candidates MUST pass the quick-screening gates. Continue exploring cheaper hypotheses while enough time remains; near the deadline preserve verified best and report blockers/results instead of launching an unlikely-to-finish half-run.

## Continuous IDE-agent cycle

1. In /root/Optimization/v9, inspect Git status and current best commit. Preserve user changes; do not blindly reset.
2. Start vLLM directly from v9 (program.md) and inspect existing v9 half-run baseline CSVs; reuse only with verified matching source/config/provenance, otherwise plan and collect ONE fresh baseline with its full 1.5–2h cost. Track the session deadline and reserve.
3. Profile a bottleneck, write one precise kernel hypothesis in autoresearch/EXPERIMENTS.md, make the smallest genuine operator-level change. Apply ALL quick gates above, including correctness, paired kernel A/B and short serving A/B, BEFORE any expensive half-run.
4. Only after passing the quick gates and the >=90-minute time check, restart only your dedicated v9 foreground server (Ctrl+C) and run the unchanged BOTH-workload half-run. Capture production trace proving the candidate kernel executes.
5. Optionally compare paired CSVs with read-only autoresearch/evaluate.py: exit code 0 means only valid CSVs, never an accepted improvement. Re-run interleaved baseline/candidate measurements to establish repeatability; inspect request failures, both TTFTs, other workload and uncertainty. Never claim a noisy numeric observation as a confirmed win.
6. If repeatably >=5% faster in either workload than the last accepted best under matched half-run conditions, commit ONLY v9 kernel/tests/notes (never benchmark scripts or raw logs), push, open an evidence-backed GitHub PR immediately, and continue on the accepted best. For <5% incremental gains, record evidence and accumulate experiments without opening a progress PR until a cumulative 5% checkpoint is proven. A verified end goal warrants its final PR regardless of incremental percentage. If rejected, record failure and restore only the experimental edits without losing unrelated work.
7. On reaching >=2450 4K OR >=1450 16K in repeatable half-runs with all requests successful and kernel tests passing, stop and report best SHA/PR, both numbers and evidence. Do NOT run MATH-500.
8. If stopped by deadline/blocker, report best verified result and PRs rather than speculate. Codex IDE extension is the agent: no extra SSH orchestration, CLI installation, or detached long-running service is presumed.

Numeric comparison is not proof of causal improvement; PR must link raw evidence and corresponding code SHA. Never claim a PR exists unless GitHub confirms it.
