# v15 latest research handoff

Status: GPU research was completed for the November 16 allocation. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. The grouped worker-attribution probe was stopped before completing a request. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-17T13:27:40+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 586252064ff732956b43b29c02265e194e4ac597

Remaining GPU time: unknown

Finding: Localize model non-Torch growth to allocation granularity

Evidence/provenance: Baseline unmodified v15: non-Torch 264241152 bytes before initialize_model and 1096810496 immediately after (+832569344); weight loading and postprocessing did not change it. Fresh synthetic processes allocating 5GB: one 5GB allocation non-Torch unchanged; 50 x 100MB unchanged; 500 x 10MB raised non-Torch 394264576 bytes and released it when all tensors/cache freed. With PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, 500 x 10MB raised non-Torch by 0 and torch reserved 5012193280 vs default 5242880000. This is a synthetic feasibility result, not evidence of actual model KV blocks or throughput. Changing allocator policy may have unclear competition eligibility; diagnostic only until organizer approval. No flags, model data, cache accounting or official benchmark modified.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Measure actual model allocation, KV blocks, graph capture and reliability with the same server flags under allocator A/B; assess eligibility independently.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
