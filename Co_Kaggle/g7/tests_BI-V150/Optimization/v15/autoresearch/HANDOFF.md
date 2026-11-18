# v15 latest research handoff

Status: The confirmed GPU allocation and its November 17 extension have ended; a newly confirmed allocation is required before further GPU experiments. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. An initial grouped worker-attribution probe stopped before a request; a later diagnostic completed 8/8 requests and captured 672 grouped-specialized GPU kernel events. This confirms execution, not a serving gain. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-18T10:33:23+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 710a53911af4827830d830a0ae94c38d7500a9f6

Remaining GPU time: unknown

Finding: Candidate serving diagnostics completed; matched control pending

Evidence/provenance: Candidate source 9b870223 plus temporary first-shape logging in divargmax.sample; worker log observed 1..8x130560 during live 4K requests and 256x130560 during startup. Candidate 4K/16K low and 4K/16K concurrency64 screens completed 8/8,8/8,64/64,64/64. Candidate 4K pressure 2300.91 tok/s TTFT13211.58 TPOT625.04 ms; 16K pressure 3476.71 tok/s TTFT122484.05 TPOT612.82 ms. Logging and warmed cache make these diagnostic; no control, no comparison and no causal gain yet. Live monitor began after 4K pressure started. Candidate server stopped and temporary logging restored.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Run matched original source control with identical order, flags, seeds and metrics; assess screening guardrails before any half-run.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
