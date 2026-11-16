# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T19:10:08+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 2aa3cf10028a7d642f2bf2a968e46538cb9c98be

Remaining GPU time: unknown

Finding: Attribute activation peak to sampler

Evidence/provenance: Original v15 source, diagnostic logging only, no resets to peak statistics: after dummy model forward allocated 5161098752, reserved 5395972096, cumulative peak 5252322816 bytes; after logits allocated 5229256192, reserved 5463080960, peak 5252322816; after dummy sampler allocated 5371865600, reserved 5999951872, cumulative peak 5774530048. The cumulative peak rises 522207232 bytes during sampling, but does not identify the exact live buffers. Dummy sampler tests top-p/top-k and per-request RNG paths; actual benchmark sampling semantics are not established. Raw log SHA256 48d04a812a537c6b42d742d45347e2c94e8a1a2428e51e11845a1dd3b7c584db backed up locally. Logger removed and owned server stopped.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Identify sampler live allocations and determine if exact generation semantics can be retained while reducing real workspace; do not merely reduce profile workload.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
