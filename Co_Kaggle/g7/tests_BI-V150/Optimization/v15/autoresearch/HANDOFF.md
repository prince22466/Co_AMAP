# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T19:12:18+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: c8da3be15512f0cb3d2e722a832da3bf33d13de8

Remaining GPU time: unknown

Finding: Reject sampler log-domain fusion on exact token semantics

Evidence/provenance: Standalone prototype uses identical precomputed exponential noise. Normal random rows 8x10003 and 256x130560 yielded zero token mismatches; large baseline/candidate median 2.153/0.408 ms, temporary peak 134221824/67584 bytes, excluding original logits and noise buffers. It mismatched existing sampler for +inf and NaN logits and for 14956 of 65536 constructed finite close-noise ties; therefore exact generation behavior is not preserved. No production integration, no KV block or serving measurement; timings do not justify promotion. Scripts and raw logs preserved. Original sampler and research branch production source unchanged.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Investigate exact softmax-and-noise workspace fusion or other real memory reductions only if numerical equivalence can be maintained; preserve current KV accounting.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
