# v15 latest research handoff

Status: GPU research was completed for the November 16 allocation. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. The grouped worker-attribution probe was stopped before completing a request. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-17T11:54:44+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: bd666241cce6f861c285643a6ea21fa86a32b177

Remaining GPU time: unknown

Finding: Reject fused SwiGLU after production-like baseline check

Evidence/provenance: Original production control 063f1fb7 with original source trees. Fused BF16 output exactly matched baseline on M33 K96 N257 and M2048 K2048 N6144. First 128x128 8-warp fused tile was 3.38967 ms vs MM plus FlagGems wrapper 1.94357 ms. Resource-matched 256x128 16-warp fused tile measured 1.89325 and 1.88678 ms vs wrapper baseline 1.93786 and 1.93790 ms in two alternating screens, but wrapper exaggerates production activation cost: original MM plus explicit single-launch Triton activation proxy measured 1.54745 ms vs fused 1.89528 ms (+22.48% latency), BF16 outputs exact. v14 real compiled activation averaged 241.05 ms /1764 calls =0.137 ms/call; proxy is not proof of identical production kernel, but rules out initial gain claim. No integration, no serving, FP16/strides/graph/accuracy not tested. Exact scripts and raw logs preserved.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Investigate a different resource mechanism or small-M/vocabulary bottleneck; do not promote this fused tile or count the wrapper result as a production gain.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
