# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T17:34:11+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 1b680a7781ff0ba07c12d681708695fd5ccda55e

Remaining GPU time: unknown

Finding: Attribute startup non-KV and graph allocations

Evidence/provenance: Original v15 1b680a7781ff0ba07c12d681708695fd5ccda55e, unchanged flags; initial non-torch 153092096 bytes; before profiling non-torch 1092616192, after profiling 1153433600; torch reserved before/after 5226102784/5500829696; weights 5101233152; peak activation 656424960; available KV 22447777996 bytes; actual graph capture return 228589568 bytes with reported estimate 0 on this platform. Most non-torch growth predates profiling, 60817408 bytes arose between pre/post snapshots. No part proven reclaimable; do not relax memory accounting. Temporary logger removed, owned server stopped. Raw log SHA256 7ca50516c0f1175e1554e389c85963581741b6dff5d9b9293ba032946a0a2771, backed up locally.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Identify the peak allocation phase inside profile_run, then pursue only a real live-workspace reduction.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
