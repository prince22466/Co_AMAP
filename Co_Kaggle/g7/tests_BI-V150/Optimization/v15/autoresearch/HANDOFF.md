# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T17:29:22+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: b6fa34e84ab9d7312b716194f4227c4103879a51

Remaining GPU time: unknown

Finding: Reject grouped MM candidate on 4K pressure screen

Evidence/provenance: Candidate source commit 342e8d35; low-concurrency matched 8/8 each control/candidate 4K 1822.06/1804.21 tok/s, TTFT 10257.89/10375.08 ms, TPOT 64.11/64.62 ms; 16K 3557.37/3597.35 tok/s, TTFT 16756.01/16467.06 ms, TPOT 155.62/154.65 ms. Candidate-first 4K pressure 64/64 each control/candidate 1934.69/1797.72 tok/s (-7.08%), TTFT 24690.07/24288.74 ms, TPOT 709.55/807.14 ms (+13.75%); no failures. One pressure pair; no replication, causality or official score claim. Reject for guardrail; no half-run. Source hashes and logs available; evidence archive SHA256 d8f8342b8f5b636cf6f0b02f0fd7bd87108fec6fef4b7f2f61fa77a9c75fc590 backed up locally. Actual worker candidate-kernel launch not independently proven; source path alone is not attribution.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Keep candidate PR as rejected/provisional kernel evidence; switch to clean baseline and investigate a genuinely different GEMM or memory-lifetime hypothesis.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
