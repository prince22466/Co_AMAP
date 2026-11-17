# v15 latest research handoff

Status: GPU research was completed for the November 16 allocation. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. The grouped worker-attribution probe was stopped before completing a request. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-17T19:39:21+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: f103820d6efd88521527a5993a49a9979163208a

Remaining GPU time: unknown

Finding: Reject interleaved single-dot SwiGLU fusion

Evidence/provenance: Original v15 source; interleaved gate/up columns, single 256-wide dot and split BF16-rounded accumulator vs original MM plus one-launch pointwise proxy. For M2048 K2048 N6144, alternating 12-event median fused 3.01379 ms vs baseline 1.53813 ms (+95.9%); M33 K96 N257 0.01843 vs 0.01199 ms. Sampled BF16 outputs exactly matched, but FP16/strides/graph/production integration were not tested. Reject this kernel; no serving or accuracy result.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Find a genuinely different mechanism; do not retest this interleaved fusion without new causal evidence.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
