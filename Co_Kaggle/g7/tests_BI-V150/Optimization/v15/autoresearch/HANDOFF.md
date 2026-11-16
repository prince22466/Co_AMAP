# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T20:42:29+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: fc31cfa8296efcaf2d162ac1980e7d1104e172be

Remaining GPU time: unknown

Finding: Longer 16K cache-growth diagnostic does not show sampler gain

Evidence/provenance: Control-first matched diagnostic 16384 input, 1024 output, 64 prompts/concurrency 64, seed42, original flags. Candidate/control: 1768.94/1826.72 total tok/s (-3.16%), TTFT 225604.12/218122.45 ms (+3.43%), TPOT 239.72/229.25 ms (+4.57%); 64/64 successes each, zero failures, 2 preemptions each and zero prefix hits. One pair, no independent replication or official score. Both reached ~100 percent KV occupancy; extra candidate blocks did not eliminate preemption or produce gain. Raw archive SHA256 c81cc7da2a49e2346fec79ea6de5ab656daf435a6d4a23f873f430b0b7f3a3a9 backed up locally. Prior short pressure pairs -5.96% and -2.69%; candidate remains rejected and research-only.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Restore original production source and continue only genuinely new small experiments while preserving confirmed allocation reserve.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
