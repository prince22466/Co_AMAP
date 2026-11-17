# v15 latest research handoff

Status: GPU research was completed for the November 16 allocation. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. The grouped worker-attribution probe was stopped before completing a request. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-17T19:30:54+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: 5a59283f98443a08fc7c49b9b1431f050bbb4790

Remaining GPU time: unknown

Finding: Grouped fused SwiGLU is slower than original chain

Evidence/provenance: User granted two additional hours; same host, budget extended to 2026-11-17T21:29:00Z preserving old deadline history. Original v15 production trees; grouped fused 256x128 two-accumulator BF16 kernel with 16 warps, 2 stages vs original MM plus explicit single-launch activation proxy. Alternating 12-event median production shape M2048 K2048 N6144: grouped fused 2.04581 ms, control 1.54150 ms (+32.7% latency); M33 K96 N257 0.01316/0.00986 ms. Outputs exactly match for both sampled BF16 cases; FP16, graph, full stride coverage and production serving not tested. The activation proxy is not the exact compiled production kernel. Reject candidate, no production change.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Investigate a distinct source-owned projection, memory-lifetime or small-M dataflow mechanism; preserve 25-minute reserve and do not attempt unadmitted half-runs.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
