# v15 latest research handoff

Status: The confirmed GPU allocation and its November 17 extension have ended; a newly confirmed allocation is required before further GPU experiments. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. An initial grouped worker-attribution probe stopped before a request; a later diagnostic completed 8/8 requests and captured 672 grouped-specialized GPU kernel events. This confirms execution, not a serving gain. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-17T19:48:40+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 300c3fa7df988894c197001e99b198927e42e6f8

Remaining GPU time: unknown

Finding: Promising isolated division-argmax fusion requires serving validation

Evidence/provenance: Same FP32 probs and exponential noise, original softmax and RNG outside prototype, two-stage divide with round-to-nearest and first-index NaN/tie reduction. 256x130560 median preallocated torch div plus argmax 1.27503 ms vs fused 0.41023 ms (16 alternating event samples; approximately 68% lower operation latency). Repeat 1.27517 vs 0.40996 ms. Compared against native in-place div_ plus argmax: zero token mismatches in a 65536x16 batch comprising 65532 constructed finite near-tie rows and four overwritten edge-case rows, 2048x2047/2048/2049 and 256x130560 with zeros/zero, NaNs, infinities and ties; three CUDA graph replays with unchanged inputs passed for each shape. Timing reference uses preallocated out= division rather than in-place div_; no RNG, integration, fallback, FP64, noncontiguous, seeded generator or production serving validation; no end-to-end gain claimed. Temporary workspace is small; no KV-block claim. Code in autoresearch/probes/sampler_divargmax_*.py, raw logs archived separately.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Audit exact division and mutation contract; implement isolated backend integration with fallback only if legal, then test seeded sampling, multiple layouts, graph, and short balanced serving screens. Do not use half-runs without budget admission.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
