# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T16:47:52+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 1b680a7781ff0ba07c12d681708695fd5ccda55e

Remaining GPU time: unknown

Finding: Start v15 allocation; pin clean baseline and budget

Evidence/provenance: Baseline 1b680a7781ff0ba07c12d681708695fd5ccda55e; FlagGems tree 7cdae638186b8dba929cbfb1a4ccbf5eea18787d; plugin tree b481b8fa8b41712267262ce678adba8132a3043f; benchmark SHA256 873f17f932d4643222be9903ba3b569c1863c4e2d246ceafbe6f75ebe7163ffc; BI-V150 32 GiB; UTC deadline 2026-11-16T22:00:00Z; 27 helper tests pass; shell imports point to checkout; no GPU experiment yet.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Target unexplained non-KV startup allocations and inspect production GEMM resource evidence; apply novelty gate before GPU tests.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
