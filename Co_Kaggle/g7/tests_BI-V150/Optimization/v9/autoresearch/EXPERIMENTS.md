# v9 Codex IDE half-run experiment log

Remote only: /root/Optimization/v9. Input benchmark: /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py (unchanged). Development goal: 4K >=2450 OR 16K >=1450 Total tok/s. MATH-500 is deferred to the user.

All raw logs, traces, repeated A/B outputs and manifests live in v9/.autoresearch-work/. Record the exact commit, test and production-trace evidence for every change.

| ID | Base SHA | Kernel hypothesis / edit | Correctness tests | Half-run 4K A -> B tok/s | Half-run 16K A -> B tok/s | Repeatability / decision / PR |
| --- | --- | --- | --- | --- | --- | --- |
| 000 | v9 starting point | Establish fresh v9 half-run baseline; do not substitute historical v8 full-run values | Pending | Pending | Pending | Pending |

For each experiment note date/time, server SHA, source paths, benchmark SHA, exact commands, raw and summary CSVs, both TTFTs, any failed requests, observed kernel invocation and profiler evidence. Document rejected experiments to avoid repeating unsupported hypotheses. File a PR for each reproducible serving-level gain, even below the goal, and continue until goal/deadline/blocker. No MATH-500 gate or claim.
