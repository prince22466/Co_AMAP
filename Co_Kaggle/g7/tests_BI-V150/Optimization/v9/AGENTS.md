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

Stop ONLY on a reproducible milestone, a user-specified deadline, or a hard blocker. When any kernel change yields an observable, repeatable total-tok/s improvement above measurement noise (even short of milestones), record it, commit ONLY v9 changes, and open a GitHub PR with raw evidence. Then continue from the last accepted best commit. Do not auto-merge. Do not insist on an arbitrary 1% floor; reproducibility determines whether a gain is real.

Historical v8 FULL-run reference (NOT a v9 half-run baseline): 4K 2331.29; 16K 1251.22. Never use these as fresh v9 half-run controls.

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

## Continuous IDE-agent cycle

1. In /root/Optimization/v9, inspect Git status and current best commit. Preserve user changes; do not blindly reset.
2. Start the v9-owned vLLM server with autoresearch/bench.py. First collect a FRESH half-run v9 baseline for BOTH workloads. Record SHA, exact commands and CSV/log paths.
3. Profile a bottleneck, write one precise kernel hypothesis in autoresearch/EXPERIMENTS.md, make the smallest genuine operator-level change.
4. Run focused GPU/correctness tests; restart only the server owned by autoresearch/bench.py; run BOTH unchanged half workloads. Capture trace showing candidate kernel reached production.
5. Validate paired CSVs with autoresearch/evaluate.py; if differences are small/noisy, run interleaved baseline/candidate repetitions under comparable GPU load. Always disclose request failures, TTFT, the other workload's change and uncertainty; avoid falsely declaring a noisy improvement.
6. If repeatably improved, commit ONLY v9 kernel/tests/notes (never benchmark scripts or raw logs), push, open an evidence-backed GitHub PR immediately, and continue on the accepted best. If rejected, record failure and restore only the experimental edits without losing unrelated work.
7. On reaching >=2450 4K OR >=1450 16K in repeatable half-runs with all requests successful and kernel tests passing, stop and report best SHA/PR, both numbers and evidence. Do NOT run MATH-500.
8. If stopped by deadline/blocker, report best verified result and PRs rather than speculate. Codex IDE extension is the agent: no extra SSH orchestration, CLI installation, or detached long-running service is presumed.

Numeric gate is not proof of causal improvement; PR must link raw evidence and corresponding code SHA. Never claim a PR exists unless GitHub confirms it.
