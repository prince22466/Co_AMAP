# v9 autoresearch contract — remote Codex IDE, half-run

**ONLY source workspace:** /root/Optimization/v9 (mirrored locally at Co_Kaggle/g7/tests_BI-V150/Optimization/v9). No other Optimization version or framework installation may be modified.

**Performance objective (logical OR):**
- Half-run 4K [4096,1024,64,256]: Total tok/s >= 2450
- Half-run 16K [16384,1024,64,128]: Total tok/s >= 1450

Use the unchanged READ-ONLY script /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py: RUNS=2, SKIP_FIRST=0, both shapes in every measurement. It is NOT the official full-run. Do not change its code, run counts, parsing or input parameters.

**Defer MATH-500 completely.** The human runs it after the research goal. Passing a half-run is not passing official performance or accuracy evaluation. Still require kernel functional tests, no missing/failed benchmark requests, and production invocation evidence. Always record the other workload and TTFT; these are disclosures, not extra hard gates imposed over the user's OR target. For any small gain, repeat paired A/B beyond one half-run to exclude measurement noise.

Priority: (1) 16K attention memory dependencies/shared-memory/tiles/occupancy; (2) mixed prefill-decode interference and kernel causality; (3) 4K attention and prefill GEMM locality/tiling.

CYCLE: FRESH BASELINE -> PROFILE -> ONE KERNEL PATCH -> GPU TEST -> RESTART OWN v9 SERVER -> HALF-RUN BOTH -> SCORE -> REPEAT A/B AS NEEDED -> ACCEPT+COMMIT+PR OR REJECT+REVERT EXPERIMENT -> NEXT.

Stop on reproducible goal, concrete user deadline or blocker. Record every observable, repeatable serving improvement with a distinct evidence-backed PR even if short of 2450/1450, and continue. Do not auto-merge. Never accept switching operators, hyperparameter/benchmark tuning, quantization, speculative decoding, or merging new upstream FlagOS as the optimization. For failures, preserve findings and baseline; avoid restoring unrelated user changes.

Run commands documented in README.md; all results under v9/.autoresearch-work. A prior v8 full-run throughput value is not a valid half-run baseline. Numerical evaluator accepts valid two-run half CSVs, reports candidate values, goal status and percentage deltas; it does NOT prove causal repeatability. For PRs attach actual raw, paired repetitions, trace and SHAs (logs stay untracked).

## Session report (in .autoresearch-work/session-report.json)

Record best_commit, best_pr_urls, baseline_csvs, candidate_csvs, four_k_total_tok_s, sixteen_k_total_tok_s, goal_reached, tests_passed, kernel_trace_path, research_log, outstanding_accuracy_check=true, and notes. Never claim MATH-500 was run or accepted.
