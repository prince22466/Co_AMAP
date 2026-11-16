# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T20:02:21+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: ff0805f2d9a7b496d12d59b4b6c113443119dbe3

Remaining GPU time: unknown

Finding: Reverse-order sampler serving screen also below control

Evidence/provenance: Control-first matched restart, 8/8 short cases and 64/64 pressure cases with zero failures. 4K candidate/control 1806.90/1807.88 tok/s, TTFT 10333.14/10317.14 ms, TPOT 64.73/64.78 ms. 16K candidate/control 3557.90/3555.52 tok/s, TTFT 16743.76/16734.46 ms, TPOT 155.69/155.94 ms. 16K pressure candidate/control 3565.89/3664.46 tok/s (-2.69%), TTFT 127515.54/122818.35 ms (+3.82%), TPOT 559.51/542.79 ms (+3.08%). Prior candidate-first pair was -5.96%; differences are noisy and not causal proof, but neither supports a serving gain and both throughput screens exceed the 2% guardrail. No half-run; candidate remains research-only. Raw archive SHA256 e1e3bc1c1e6e2d668e943dfc0aa4f72a96b918c0b2f30983ec6a5139e7b1936c backed up locally.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Return to original v15; seek genuinely different GEMM or allocation mechanism, subject to remaining budget and rules.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
