# v9 autoresearch contract — remote Codex IDE, half-run

**ONLY source workspace:** /root/Optimization/v9 (mirrored locally at Co_Kaggle/g7/tests_BI-V150/Optimization/v9). No other Optimization version or framework installation may be modified.

**Performance objective (logical OR):**
- Half-run 4K [4096,1024,64,256]: Total tok/s >= 2450
- Half-run 16K [16384,1024,64,128]: Total tok/s >= 1450

User-reported initial v9 half-run reference: **4K 2130 Total tok/s; 16K 1240 Total tok/s** (the second "4K" label is interpreted as 16K, pending raw-CSV verification). A **5% observable gain** over these planning references corresponds to **4K 2236.5** or **16K 1302**. Re-establish a FRESH matched half-run for the current last accepted best before claiming a gain.

Use the unchanged READ-ONLY script /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py: RUNS=2, SKIP_FIRST=0, both shapes in every measurement. It is NOT the official full-run. Do not change its code, run counts, parsing or input parameters.

**Defer MATH-500 completely.** The human runs it after the research goal. Passing a half-run is not passing official performance or accuracy evaluation. Still require kernel functional tests, no missing/failed benchmark requests, and production invocation evidence. Always record the other workload and TTFT; these are disclosures, not extra hard gates imposed over the user's OR target. For any small gain, repeat paired A/B beyond one half-run to exclude measurement noise.

Priority: (1) 16K attention memory dependencies/shared-memory/tiles/occupancy; (2) mixed prefill-decode interference and kernel causality; (3) 4K attention and prefill GEMM locality/tiling.

## GPU-time control: expensive half-run only after fast gates

Observed duration of ONE unchanged 4K+16K half-run: **1.5–2 hours**. A fresh baseline AND candidate can require **3–4 hours** before any confirmation. Write an explicit deadline/time-remaining log. **Never start a half-run if <90 minutes remain**. A run started with 90–119 minutes remaining may exceed the deadline (observed duration can reach 120 minutes); the threshold is a start rule, not a guaranteed finish; reserve more (approximately 300 minutes or greater) if a baseline or repeat comparison is required. Do not spend the GPU budget repeatedly collecting full half-runs of weak hypotheses.

For EACH candidate, do this order **before** a half-run: (1) causal production profiler and bottleneck share; (2) v9 functional GPU/fallback correctness and relevant 4K/16K/mixed attention or GEMM edge cases; (3) warmed-up repeated A/B kernel microbench with timings, variance, registers/SMEM/occupancy/spills/memory traffic; (4) serving trace proving edited kernel was used; (5) short repeated identical serving A/B probes using direct diagnostic commands in program.md, ensuring successful requests and no obvious TTFT/TPOT regression. Cap initial screening effort to roughly **20–30 minutes per candidate** unless new evidence justifies deeper profiling. Use Amdahl reasoning to decide whether a >=5% end-to-end gain is even plausible. Reject bad, unproven or marginal candidates before the 2-hour benchmark. Screened tests DO NOT replace official half-run numbers.

Reuse a documented matching v9 half-run baseline only if original logs/CSVs, exact code SHA or hashes, benchmark hash, machine and serving config are comparable; otherwise budget to remeasure. Never claim confirmed repeatability based on a single candidate half-run if workload noise or changed conditions demand another run. If insufficient time for credible evidence, preserve best verified result and stop without inventing a PR or accuracy claim.

CYCLE: VERIFY/MEASURE MATCHED BASELINE -> PROFILE -> ONE KERNEL PATCH -> CORRECTNESS -> KERNEL A/B -> SHORT SERVING A/B -> GO/NO-GO + TIME CHECK -> HALF-RUN BOTH -> REPEAT IF NEEDED -> ACCEPT+PR OR REJECT -> NEXT.

Stop on reproducible goal, concrete user deadline or blocker. **Checkpoint progress requires a repeatable >=5% Total tok/s gain in at least one case relative to the last accepted best**, tested under comparable A/B conditions. Log smaller positive changes without an immediate progress PR; cumulative gains can qualify once they cross 5%. Open an evidence-backed PR for each verified 5% checkpoint even short of the target; always open a final PR for a reproducible 2450/1450 goal even when the last incremental step is smaller. Continue after progress checkpoints. Do not auto-merge. Never accept switching operators, hyperparameter/benchmark tuning, quantization, speculative decoding, or merging new upstream FlagOS as the optimization. For failures, preserve findings and baseline; avoid restoring unrelated user changes.

Run commands documented in README.md; all results under v9/.autoresearch-work. A prior v8 full-run throughput value is not a valid half-run baseline. Optional numeric comparator checks two-run half CSVs, reports >=5% numerical checkpoint observations, final-goal threshold observations and percentage deltas; exit 0 confirms valid CSVs only. It does NOT prove a speedup, goal reproducibility, kernel invocation or accuracy. For PRs attach actual raw, paired repetitions, trace and SHAs (logs stay untracked).

## Session report (in .autoresearch-work/session-report.json)

Record best_commit, best_pr_urls, baseline_csvs, candidate_csvs, four_k_total_tok_s, sixteen_k_total_tok_s, goal_reached, tests_passed, kernel_trace_path, research_log, outstanding_accuracy_check=true, and notes. Never claim MATH-500 was run or accepted.
