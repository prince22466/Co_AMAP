# v15 latest research handoff

Status: GPU research was completed for the November 16 allocation. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. The grouped worker-attribution probe was stopped before completing a request. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-17T19:37:09+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 1e8cc4a87dc83e3225c8bbb6f2b6bfc0130ccdfb

Remaining GPU time: unknown

Finding: Verify grouped MM GPU execution in 4K worker

Evidence/provenance: Diagnostic original v15 control source plus isolated grouped candidate 342e8d35; duplicate identical grouped kernel renamed mm_kernel_grouped_probe and selected only for M>=1024 N>=8192 stride_bk=1, profiler enabled, no performance claim. A 4K random diagnostic with 8/8 successful requests generated 672 rank0 GPU kernel events matching mm_kernel_grouped_probe, summed duration 784050.465 microseconds in profiler window; this establishes grouped-specialized worker execution in that diagnostic. Async frontend trace has zero GPU events; correct worker rank0 trace SHA256 8f46da15738f8063bc94aef4c12bb00738af80001a3e0e1f28175cacc2ef4a57; 4k client log SHA256 1241260ebaeeac420a9d3e1577f4bc94d3ef4efbbd58514740a3f9aba21e9313. Trace perturbed timings and excludes unseen graph replay; no claim of throughput or accuracy gain. Temporary source restored, owned server stopped.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Analyze new independent GEMM/epilogue mechanism and preserve baseline; prior balanced serving remains inconclusive and half-runs are not admitted in remaining time.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
