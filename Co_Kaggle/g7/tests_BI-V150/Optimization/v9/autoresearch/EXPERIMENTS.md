# v9 Codex IDE half-run experiment log

Remote only: /root/Optimization/v9. Input benchmark: /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py (unchanged). Development goal: 4K >=2450 OR 16K >=1450 Total tok/s. User-reported initial v9 half-run references: 4K 2130, assumed 16K 1240 tok/s (verify raw CSV). 5% progress threshold vs these references: 4K 2236.5 or 16K 1302; thereafter 5% vs the latest accepted best. MATH-500 is deferred to the user.

## Required time/evidence ledger per candidate

Before editing, record: session deadline (with time zone), remaining GPU allocation, whether an authenticated matching v9 half-run baseline is available, source/serving/benchmark SHA or hashes, target bottleneck share and the expected end-to-end effect. For each hypothesis record QUICK gates (correctness/fallback GPU tests, paired warmed-up microbenchmark medians/variance, GPU resource/traffic evidence, real kernel invocation, short baseline/candidate serving A/B with both 4K/16K diagnostic cases, request success, TTFT/TPOT), plus a **go/no-go decision**. Budget ~20–30 minutes for initial quick screens. One complete 4K+16K half-run may take 1.5–2h; require >=150 min remaining and reserve ~300 min if baseline/repeated controls needed. Only promote high-confidence candidates. Reject early when plausible end-to-end gain is too small, reliability is poor, or time is insufficient. Mark diagnostics as NOT official half-run evidence; do not mislabel short-serving tok/s as scored 4K/16K.

All raw logs, traces and repeated A/B outputs live in v9/.autoresearch-work/. Record exact source commit or file hashes, benchmark hash, tests and production-trace evidence for every change. Use the direct commands in program.md; no benchmark wrapper.

| ID | Base SHA | Kernel hypothesis / edit | Correctness tests | Half-run 4K A -> B tok/s | Half-run 16K A -> B tok/s | Repeatability / decision / PR |
| --- | --- | --- | --- | --- | --- | --- |
| 000 | v9 starting point | User-reported half-run values; verify 16K label and raw CSV, establish matched best baseline | Pending | 2130 reported (unverified) | 1240 reported (assumed 16K, unverified) | Baseline not yet verified |

For each experiment note date/time, server SHA, source paths, benchmark SHA, exact commands, raw and summary CSVs, both TTFTs, any failed requests, observed kernel invocation and profiler evidence. Document rejected experiments to avoid repeating unsupported hypotheses. File a progress PR for each >=5% repeatable serving-level gain against last accepted best, even below goal; log smaller gains for accumulation. File a final PR upon reproducibly reaching the goal even for a <5% last step. Continue until goal/deadline/blocker. No MATH-500 gate or claim.
