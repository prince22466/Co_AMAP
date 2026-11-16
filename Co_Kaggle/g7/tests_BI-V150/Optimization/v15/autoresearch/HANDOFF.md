# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T19:19:17+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 092dc0aea560caa0c098619bef985665ac1c623e

Remaining GPU time: unknown

Finding: Sampler in-place softmax adds KV blocks in startup probe

Evidence/provenance: Candidate source 1fd9d417. Direct seeded tests matched native token IDs for unfiltered, top-k, top-p and per-row generator cases, and nonfinite behavior; processed-logits fallback and three graph replays passed. Actual diagnostic worker invoked candidate twice on 256x130560. Measured original/candidate activation peak 656424960/522207232 bytes; non-torch 1000341504/1004535808; KV available 22447777996/22577801420 bytes; 521936/524960 KV tokens, +3024 tokens (+189 blocks, +0.58%). Same fixed server flags, one startup each, temporary provenance/accounting loggers removed before use. Candidate actual device memory after capture 28476 MiB; source and tests committed separately. No matched serving or independent block replication; exact logits alias side effects, processed modes, mixed RNG paths and competition eligibility still need review. Baseline graph pool actual 228589568 bytes. Candidate startup log SHA256 a6b5587dcd19e0eb0e7a0d06bba9d8050b1caa0cbe3cb0b4140b1622f3b894be.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Back up raw evidence; run clean-source 4K/16K serving checks and a 16K concurrency-64 pressure screen; replicate startup block allocation and expand semantics tests.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
