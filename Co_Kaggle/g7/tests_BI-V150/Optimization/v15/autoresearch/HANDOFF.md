# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T17:06:25+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 342e8d359730c4c06d325e821a9f0f7464bd1b88

Remaining GPU time: unknown

Finding: Grouped MM kernel gain with direct correctness and graph replay

Evidence/provenance: Wide MM M=2048 K=2048 N=12288 median control1/candidate1/control2/candidate2 1.40987/1.17073/1.3928/1.1703 ms, 15 warm event timings each; other production shapes and irregular fallback were nearly unchanged. Direct FP32-reference BF16 checks passed on four shapes plus fallback (atol .25 rtol .02); grouped odd M/N tail and three graph replays passed for FP16/BF16. Native original-wide capture: 76.41 percent TCU efficiency, 25.14 percent occupancy, 97 vector registers per thread, 65536 bytes shared memory, launch 384 blocks x 1024 threads; selected 256x256x32, 16 warps, 2 stages. Only kernel evidence, sequential not alternating per sample, no serving result or quality claim. Prior v11 serving was inconclusive; no throughput-goal claim.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Back up raw logs; run matched short 4K and 16K serving screens, confirm production candidate execution and keep latency guardrails.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
