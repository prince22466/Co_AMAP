# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T16:53:19+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: f896b5e2a72aa702b9929005115a31b9b70703fc

Remaining GPU time: unknown

Finding: Measure v15 startup non-KV memory breakdown

Evidence/provenance: Baseline source 1b680a7781ff0ba07c12d681708695fd5ccda55e; temporary diagnostic worker logging only, removed after owned server shutdown; fixed flags per program.md; requested 29205777612 bytes, weights 5101233152, activation peak 656424960, non-torch 1000341504, graph estimate and applied 0, available KV 22447777996, 521936 KV tokens; full decode graph capture completed. Raw startup log SHA256 36a195657709a0c5a82a8eaab23a39efd04d9a658f781ba81f8c0f97603d45f5, backed up locally. No reclaimability or throughput gain established; actual graph pool bytes not isolated.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Find specific reclaimable lifetime or obtain targeted GEMM resource evidence, then test genuinely new candidate.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
