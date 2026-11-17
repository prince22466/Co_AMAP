# v15 latest research handoff

Status: GPU research was completed for the November 16 allocation. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. The grouped worker-attribution probe was stopped before completing a request. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-17T13:23:35+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 8461eb2a6dbe6251cad9134a4dba84a728bf5ed6

Remaining GPU time: unknown

Finding: Attribute vocabulary Linear and non-Torch growth

Evidence/provenance: Original v15 production trees. Native vocabulary Linear M1 K2048 N130560 selected 1x256x32 8 warps: 0.818 ms profiler duration, DRAM read efficiency 88.04%, TCU 10.77%, occupancy 24.83%; M256 selected 256x256x32 16 warps: 1.826 ms, DRAM 40.20%, TCU 76.27%, occupancy 24.56%. Profiler captures are diagnostic, warmup and serving impact unknown; no speedup. Initial worker snapshots: runner created non-Torch 264241152 bytes, model loaded 1096810496, profiling done 1157627904. Thus 832569344 bytes (~794 MiB) accumulated during model loading, with 60817408 bytes (~58 MiB) during profiling; it is not yet proven reclaimable. Instrumented server stopped and worker restored. Raw logs preserved off-instance after checkpoint.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Split model loading into initialization, weight loading and post-processing snapshots; design a genuine memory-lifetime candidate only if evidence identifies one.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
