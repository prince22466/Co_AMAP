# Codex instructions — MiniCPM5-2B BI-V150 v11

Read `program.md` and `autoresearch/EXPERIMENTS.md` before touching inference code. This file is the authoritative research contract; `program.md` is the command runbook. Codex is operating in the existing Remote-SSH IDE on the authorized BI-V150 machine.

## Workspace boundary — v10 only

- Remote source, tests, profiler artifacts and serving workspace: **/root/Optimization/v10**.
- Repository mirror: `Co_Kaggle/g7/tests_BI-V150/Optimization/v10`.
- Do not edit, overwrite, copy changes into, or benchmark FROM v0–v9, `/workspace/vllm-plugin-FL`, `/workspace/FlagGems`, or any unrelated checkout. The supplied read-only half-run script under `/workspace/vllm-plugin-FL/benchmarks/` is the sole exception: invoke it in place, without modifying it.
- All temporary logs, CSVs, native GPU traces and probes go inside `v10/.autoresearch-work/` (ignored by Git). Commit only accepted v10 code, regression tests, and concise reproducible findings.
- No separate SSH bridge, local Codex CLI/daemon, or automated time allocation is assumed. Preserve user edits and owned services; never kill unrelated jobs.

## Exactly defined target: +6% over FIXED reported v9 half-run

The user supplied v9 development half-run averages (two runs per workload):
- 4K `[4096,1024,64,256]`: **2256.31** Total tok/s.
- 16K `[16384,1024,64,128]`: **1518.06** Total tok/s.

The final goal is **logical OR**: candidate >= `2256.31 * 1.06 = 2391.6886` (display **2391.69**) for 4K, OR >= `1518.06 * 1.06 = 1609.1436` (display **1609.15**) for 16K. Achieving BOTH is better, not required. Test and disclose BOTH every time. The goal is always anchored to those FIXED v9 reference values, not to a moving v10 baseline. Do NOT accidentally use v9's old 2450/1450 thresholds, or interpret the old v9 16K target as already satisfying v10.

These v9 numbers originate from `profiling/v9/summary.txt`, without original raw CSVs attached there. The reported 4K individual repetitions were 2044.14 and 2468.48 tok/s, a spread much larger than 6%; a single two-run average crossing the fixed goal is NOT proof of speedup. Treat them as **reported fixed goal anchors**, not as a substitute for provenance-verified, matched A/B controls. First verify v10 starts with the exact stage-2 optimized v9 prefill kernel and v9 mixed-paging tests; establish a fresh v10 starting-code half-run A when comparable raw reference data are unavailable. Compare each candidate B to the current accepted v10 control A with identical model, server, GPU, source and benchmark inputs. A fresh control is necessary but not sufficient: repeat matched A/B before attributing a causal gain. It does NOT change the fixed v9 goal. Any mismatch or drift must be disclosed and investigated.

**Progress PR rule (distinct from final goal):** open a PR for a repeatable >=5% gain in EITHER workload compared to the last accepted matched v10 baseline, even if the fixed +6% v9 goal has not yet been reached. Smaller steps may accumulate until a repeatable >=5% checkpoint. A repeatable final +6% v9 goal with positive matched-v10 uplift deserves a final PR regardless of last incremental progress. Do not auto-merge. Numeric thresholds alone are neither reproducibility nor correctness. Disclose the other workload; do not automatically promote a material net-negative two-case competition tradeoff without user approval.

Use the UNMODIFIED `/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py` with `RUNS=2`, `SKIP_FIRST=0`, and the exact two workloads. These are development half-run results, NOT official full-run scores. MATH-500 accuracy evaluation is reserved for the human operator; do not run or claim it was passed. Preserve functional correctness in every experimental candidate.

## Prioritized hypothesis queue — REPROFILE v10 BEFORE selecting work

1. **4K decode GEMM/Linear:** first profile actual v10 4K prefill/decode/mixed stages. If material, optimize small-M real production `mm`/`linear` projection shapes (M and active-batch distribution, K/N, A/B layout, tile utilization, output tail masks, weight reuse, memory transactions, accumulation dependencies). Prior v7 traces put Linear at ~64.5% of *isolated* 4K decode summed GPU kernel time; that is historical evidence, NOT v10 end-to-end time share. Do not simply stack the v7 grouped-MM code: it was not a proven serving win.
2. **Mixed prefill/decode attention:** measure real serving frequency and time share first. Pure decode `DECODE_PAGE_LOOKUP` is gated on invocation-wide `max_seqlen_q == 1`; mixed calls may lose this optimization. Investigate operator-local data-movement/lookup/layout improvements *without replacing the framework's operator selection or silently bypassing fallbacks*. Distinguish GPU interference, scheduler overhead and KV-cache waiting through traces; historical correlation is not causation.
3. **16K prefill attention memory pipeline:** preserve v9's two-stage BF16 GQA specialization; inspect emitted instructions, paged-KV dependency chain, K/V reuse, score/softmax accumulator layout, register/shared-memory footprint, and bank behavior. v9 achieved ~31% direct prefill kernel acceleration with near-unchanged occupancy, so occupancy alone is not a proven optimization lever. v4 grouped page lookups regressed in direct tests: do not repeat without a genuinely new, measured kernel hypothesis.
4. **KV-cache waiting/preemption, GPU idle and launch gaps:** monitor live 4K/16K requests and investigate only if fresh v10 evidence identifies an addressable cause through legitimate operator or memory-path improvements. Never change scheduler policy, capacity, benchmarking inputs or serving flags merely to inflate score without an allowed kernel/operator optimization.
5. RMSNorm/SiLU/other small kernels only when current trace plus Amdahl bound supports meaningful total-throughput improvement.

Re-rank whenever fresh **v10** serving traces contradict the historic ordering. Report shares as GPU summed-kernel time versus true wall time accurately. Track GPU active/idle, memory counters and output tok/s/TPOT in addition to the official target metric Total tok/s.

Primary source candidates:
- `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py`
- `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150.py`
- `FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/mm.py`
- `FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/linear.py`

## Inviolable competition constraints

- No quantization or speculative sampling by changing hyperparameters. Do not change model, weights, generation config, concurrency, input/output lengths, prefill/decode counts, graph mode, vLLM serving flags or environment for score manipulation.
- Do not modify or tune the provided half-run, official benchmark, accuracy evaluator, dataset, or their workload definitions. Local standalone *diagnostic* probes are allowed only to screen kernels, not to replace scored tests.
- No operator-selection-only speedup, no removal of major dispatch/selection/fallback logic, no unsupported backend switch. Every optimization must improve real existing kernel/operator computation or data movement with equivalent semantics.
- No merging latest upstream FlagOS inference framework code as an optimization shortcut.
- No assertion weakening, test bypass, unverified accuracy claim or cherry-picked failed/incomplete benchmark result. Preserve the original 4K and 16K shape and request-success requirements.
- Work only in the permitted v10 code and test files. Never modify the deployed external benchmark path.
- Competition rules: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn

## Required time-aware funnel

A complete unchanged 4K+16K half-run has historically taken **~1.5–2 hours**. Do not consume that for every hypothesis. Record GPU deadline/allocation (or explicitly mark unknown), elapsed time, remaining time, server/source/benchmark hashes, current best SHA and raw control data.

For EACH hypothesis:
1. **Production causal profiling:** isolate actual v10 operator execution, shapes and attributable critical-path/time share; estimate maximum plausible serving win via Amdahl. Reject tiny wins before editing.
2. **Correctness first:** run the existing v10 attention, decode and mixed-paging tests plus appropriate changed-operator BF16/fallback GPU tests, paging boundaries, mixed batches, masks, shape tails, NaNs and numeric tolerances.
3. **Paired kernel A/B:** warmup and repeat identical tensors/layout/dtype across control and candidate; report medians, variability, register/SMEM/spills/occupancy, bandwidth/bank/conflicts if available. Confirm changed operator executes in production.
4. **Short paired serving A/B:** use the directly documented *diagnostic* probes for both 4K and 16K, matched restarts and ordering; check errors, TTFT, TPOT, output rate and cache effects. Short probes are NEVER scored half-runs.
5. **Written go/no-go:** approximately 20–30 minutes initial screen per hypothesis; only launch costly both-workload half-run if causal kernel/serving evidence predicts an actual win, with at least **90 minutes** remaining. This is merely a start floor; 90–119 minutes may not cover a 120-minute run. Prefer **~300+ minutes** when a fresh baseline and independent repeats are needed. If deadline is unknown, never invent remaining time or claim guaranteed completion.
6. **Repeated matched half-runs:** require both cases to have two complete SUCCESS repetitions with all requests; independently repeat/interleave baseline and candidate if noise warrants, especially 4K's historic warm/cache-driven variability. Use `autoresearch/evaluate.py` only as a read-only CSV validator: its exit 0 means syntactically valid inputs, not a demonstrated gain.
7. **Promotion:** if repeatable >=5% versus last accepted best, commit only v10 operator/test/doc changes, push and open an evidence-backed PR, then continue. If repeatably >=6% relative to the FIXED v9 anchor in at least one case, with positive matched-v10 uplift and no unapproved material net-negative tradeoff, open a final PR and stop. Do not require the other case to beat its target, but disclose its result and any regression. If a candidate fails correctness, production invocation, or plausibility, reject and restore only experimental edits. Never auto-merge.

Log every hypothesis and rejected experiment in `autoresearch/EXPERIMENTS.md` with exact run provenance, raw CSV references, trace paths, negative evidence, resource metrics, and PR URL. Keep large logs under ignored `.autoresearch-work`. On deadline or blocker report the best verified result and unresolved questions. Codex works interactively in the connected Remote-SSH IDE; creating files does NOT launch unattended research.
