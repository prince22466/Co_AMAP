# v15 latest research handoff

Status: GPU research was completed for the November 16 allocation. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. The grouped worker-attribution probe was stopped before completing a request. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-17T19:27:29+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: ecc9fee944c2d107d7ba7962b9cc69429b062f2d

Remaining GPU time: unknown

Finding: Expandable allocator long 16K diagnostic regresses

Evidence/provenance: Original v15 production control 063f1fb7, unchanged server flags. Fresh control-first 16K diagnostic: 16384 input, 1024 output, 64 prompts/concurrency, seed42. Expandable/control 1678.40/1878.24 total tok/s (-10.64%), TTFT 228617.09/218196.08 ms (+4.78%), TPOT 264.75/222.17 ms (+19.17%). Both completed 64 requests, zero failures, zero prefix hits, and two preemptions each. Single pair is not causal proof or official half-run, but fails throughput and latency guardrails; no verified gain. Standalone allocation smoke: 80 reuse iterations, ten graph replays and matching output SHA256 9fb6b2c687b0d115530224460470c3cd950842303acb0c6f636ae44b403cc227 for both modes. This does not prove full KV lifecycle correctness. Allocator-policy eligibility remains unresolved; diagnostic only. Confirmed allocation expired 2026-11-17 16:00 UTC. Raw evidence backup required.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Do not promote allocator mode on these screens. Restore original source, back up logs and continue future work only on a new confirmed allocation.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
