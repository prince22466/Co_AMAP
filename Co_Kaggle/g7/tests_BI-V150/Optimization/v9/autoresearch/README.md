# v9 autoresearch — remote Codex IDE

Codex is already installed in the remote SSH IDE. Work on **/root/Optimization/v9 ONLY**, mirrored locally at Co_Kaggle/g7/tests_BI-V150/Optimization/v9. There is no SSH bridge, local CLI supervisor, or bench.py wrapper.

**Direct, exact vLLM start, GPU tests, read-only half-run and comparison commands:** [program.md](../program.md).

The required read-only benchmark is /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py. Its actual settings are RUNS=2, SKIP_FIRST=0. This external file is an INPUT ONLY: never modify or copy it and never use /workspace/vllm-plugin-FL as the source for the serving process. Both [4096,1024,64,256] and [16384,1024,64,128] must run each time. Write all logs, original raw/summary CSVs and profiler outputs under v9/.autoresearch-work/ (Git ignored).

## Numeric comparison, not acceptance

autoresearch/evaluate.py is a small optional read-only CSV checker: it verifies exact shapes, two raw SUCCESS runs with complete requests per case, both summary rows, finite metrics and summary/raw agreement. It reports both Total tok/s and mean TTFT deltas, and whether candidate 4K >=2450 OR 16K >=1450 was observed.

A **zero exit status only means valid input CSVs**; it does not mean throughput improved, nor that a milestone was reproducibly reached. Output fields numeric_gain_observed and numeric_target_observed are observations only; reproducibility_verified is always false. The script does not certify optimized kernel invocation, functional correctness, official full-run performance, or accuracy.

For small or noisy differences, repeat interleaved baseline/candidate A/B on the same idle BI-V150. Check raw run variation, functional tests and production GPU traces. No arbitrary 1% gain floor, but never open a performance PR or stop solely because of an unconfirmed numerical fluctuation.

## Research protocol

Read [GOAL.md](GOAL.md), AGENTS.md and [EXPERIMENTS.md](EXPERIMENTS.md). Start from a fresh v9 half-run baseline, profile, edit one legitimate v9 kernel/operator hypothesis, run GPU tests, restart only your own dedicated v9 vLLM process, run both half workloads, evaluate, repeat when needed. Immediately open an evidence-backed GitHub PR for every reproducible gain, even below target, and continue from accepted best. Stop at repeatable 4K >=2450 OR 16K >=1450, a specified deadline, or a genuine blocker.

The user will run MATH-500 separately after the throughput goal. Never represent the development half-run as the official full-run or as accuracy acceptance. Include SHA or file hashes, exact command, benchmark hash, original CSV and logs, both TTFTs, production trace, test results, and PR URL in each finding. Do not commit bulky logs.

If /root/Optimization/v9 is not a Git worktree, transfer only accepted v9 changes into the tracked v9 local mirror for GitHub commits/PRs. Do not touch other Optimization versions or /workspace sources. The IDE extension alone does not promise autonomous background execution.
