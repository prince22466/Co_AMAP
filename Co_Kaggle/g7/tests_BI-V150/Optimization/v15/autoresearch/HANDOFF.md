# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T20:57:32+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 42e81ce380aa3aeee5d27a1ce190abc88afd3e19

Remaining GPU time: unknown

Finding: Grouped MM reverse-order pressure screen is inconclusive

Evidence/provenance: Control-first 4K pressure 4096 input, 128 output, 64 prompts/concurrency, seed42, fixed flags. Candidate/control total 2031.52/2011.02 tok/s (+1.02%), TTFT 23978.47/24527.79 ms (-2.24%), TPOT 675.56/681.84 ms (-0.92%); 64/64 successes each, no failures. Earlier candidate-first pair 1797.72/1934.69 tok/s (-7.08%) and +13.75% TPOT. The reversed result and earlier regression indicate order or run variability; no repeatable serving gain or causal attribution is established, and a single positive diagnostic is not target achievement. Candidate remains provisional/research-only; no unchanged half-run. Raw archive SHA256 aa521f9e2cfb450af383f7f5bffae61dc80500cecd9a4aef96f1f258f5da4449 backed up locally.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Do not promote without balanced independent serving replication, production kernel attribution and matched 4K/16K half-runs; restore original source.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
