# v15 latest research handoff

Status: The confirmed GPU allocation and its November 17 extension have ended; a newly confirmed allocation is required before further GPU experiments. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. An initial grouped worker-attribution probe stopped before a request; a later diagnostic completed 8/8 requests and captured 672 grouped-specialized GPU kernel events. This confirms execution, not a serving gain. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-18T10:18:41+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: b5e80789bea59511040ab2433ece3cdbbe66e6de

Remaining GPU time: unknown

Finding: Integrated sampler passes initial correctness and full-path GPU screen

Evidence/provenance: Candidate source 9b870223; nine full native sampler cases passed exact tokens, default and per-row generator state for no/top-k/top-p filtering, mixed/all/no row RNG, FP64 fallback, truly noncontiguous input fallback, processed outputs and sampled NaN/inf. Six graph replays with changing input data matched in-place native division and argmax. Alternating 14 CUDA-event samples, vocab 130560, full softmax+noise+selection baseline/candidate medians: batch8 0.41853/0.13910 ms; batch64 1.10934/0.45963 ms; batch256 2.45832/1.59297 ms. Baseline is a hand-constructed native path with no filtering, not a measured production worker or end-to-end result; correctness coverage is limited. Candidate does not alter shared helper mutation behavior, and FP64/processed modes fall back. Raw logs and tests saved; actual worker dispatch, serving, MATH-500 and competition eligibility remain unverified.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Verify actual worker candidate dispatch and run matched short 4K/16K pressure screens before considering an admitted half-run. Retain original source as control.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
