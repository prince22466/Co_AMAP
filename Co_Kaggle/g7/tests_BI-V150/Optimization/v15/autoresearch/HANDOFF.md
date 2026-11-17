# v15 latest research handoff

Status: GPU research completed for the November 16 allocation; no verified end-to-end throughput improvement or final accuracy result. The sampler candidate remains research-only and failed the current serving guardrails.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Sampler candidate source commit: 1fd9d417. Startup probes observed a 128 MiB lower activation peak and approximately 3,000 more KV tokens, but the short and long 16K pressure comparisons did not show a serving gain. See EXPERIMENTS.md and draft PR #256 for measurements, correctness limits and evidence locations. The rejected log-domain fusion changes token selections and must not be integrated.

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
