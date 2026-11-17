# v15 latest research handoff

Status: GPU research was completed for the November 16 allocation. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. The grouped worker-attribution probe was stopped before completing a request. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-17T11:50:03+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 063f1fb7d079cb276106af81fcc4c3b64dbc3793

Remaining GPU time: unknown

Finding: Start continuation allocation with original v15 production control

Evidence/provenance: User confirmed 2026-11-17 16:00 UTC deadline on new host 2309343dd3b8; BI-V150 32GiB idle. Production control commit 063f1fb7d079cb276106af81fcc4c3b64dbc3793, identical source trees to original 1b680a7781ff0ba07c12d681708695fd5ccda55e: FlagGems 7cdae638186b8dba929cbfb1a4ccbf5eea18787d and plugin b481b8fa8b41712267262ce678adba8132a3043f. Parent includes merged 255/256 evidence; candidates isolated out of control. Prior session budget archived; benchmark SHA256 873f17f932d4643222be9903ba3b569c1863c4e2d246ceafbe6f75ebe7163ffc, RUNS=2 SKIP_FIRST=0 and helper supplies exact cases. Fresh matched half-run pair not admitted at 250 minutes; reserve 25 minutes.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Inspect production MLP projection/activation execution and choose a novel fused epilogue probe.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
