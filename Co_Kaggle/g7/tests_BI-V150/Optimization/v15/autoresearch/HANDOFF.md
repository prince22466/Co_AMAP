# v15 latest research handoff

Status: GPU research was completed for the November 16 allocation. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. The grouped worker-attribution probe was stopped before completing a request. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-17T19:42:50+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: 369af371168c7acf69b128ba0ae2679c13231639

Remaining GPU time: unknown

Finding: Reject smaller-N fusion and record resource attribution

Evidence/provenance: Original v15 source; native profile fused dual-dot 256x128 LSU 78.31%, TCU 55.77%, 102 vector registers/thread, 65536 shared bytes/block, no local memory; archived original MM 54.59%, 76.41%, 97 and 65536 respectively. Independent profiler captures, no causality established. Profiler forced termination after capture triggered pybind shutdown error; report completed. Smaller-N fused 256x64 eight warps 2 stages exact sampled BF16 on production and ragged shapes; production M2048 K2048 N6144 median 2.59466 ms vs original MM plus activation proxy 1.53635 ms (+68.9%); ragged 0.01366 vs 0.01196 ms. Reject; proxy differs from actual compiled production activation; no production serving claim.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Leave fusion absent a materially distinct mechanism; investigate other priorities within session budget.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
