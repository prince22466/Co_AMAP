# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T19:41:08+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: d6f04770d6766be5fc96a6b9239cae93bb6289a7

Remaining GPU time: unknown

Finding: Sampler reuse fails 16K pressure screen despite extra KV blocks

Evidence/provenance: Candidate source 1fd9d417, candidate-first matched restart. Candidate/control KV tokens 525056/521936 (+3120, 195 blocks, +0.60%). 4K low concurrency candidate/control 1805.59/1814.49 tok/s, mean TTFT 10365.94/10285.81 ms, TPOT 64.57/64.49 ms, 8/8 each. 16K low concurrency 3556.83/3544.86 tok/s, TTFT 16746.18/16746.63 ms, TPOT 155.75/156.73 ms, 8/8 each. 16K pressure concurrency 64 candidate/control 3282.76/3490.77 tok/s (-5.96%), mean TTFT 139102.68/122267.96 ms (+13.77%), TPOT 581.95/568.36 ms (+2.39%), 64/64 each, no failures. One pressure pair, not causal proof or independent replication; fails guardrails. Prior direct seeded and graph checks limited; no complete correctness or quality coverage. No half-run. Raw evidence archive SHA256 8f556eb192c7dbcdd4cec3868bf90416ddb11f8f6a886989d02819319402f3ec backed up locally.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Keep PR as experimental evidence, reject promotion on current screen and revert to original v15 before new research.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
