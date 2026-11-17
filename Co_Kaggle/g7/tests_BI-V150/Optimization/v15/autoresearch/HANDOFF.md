# v15 latest research handoff

Status: GPU research was completed for the November 16 allocation. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. The grouped worker-attribution probe was stopped before completing a request. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-17T13:49:07+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: a06e86030e0fb0ef25f008f6e5a6c9928e466fce

Remaining GPU time: unknown

Finding: Expandable allocation adds KV capacity; short serving near neutral

Evidence/provenance: Diagnostic allocator policy only; original source 063f1fb7 production trees, unchanged model/server/benchmark flags. Candidate-first matched startup default/expandable: non-Torch increase 1000341504/67108864 bytes, activation peak 656424960/654032896, KV bytes 22447777996/23383402700, token capacity 521936/543696 (+21760, +1360 blocks, +4.17%). Both full decode graph captures completed. 4K 8/8 candidate/control 1785.25/1786.35 tok/s, TTFT 10508.31/10466.80 ms, TPOT 65.11/65.38 ms; 16K 8/8 3545.67/3529.18, TTFT 16755.13/16798.55, TPOT 156.60/157.63. 16K pressure 64/64 3662.88/3646.08 (+0.46%), TTFT 122457.78/122507.27, TPOT 541.53/544.34, zero failed requests, zero new preemptions and 130944 prefix hits each. One pair; no independent throughput replication, correctness or MATH-500; pressure uses warmed prefix cache and 128 output tokens. Allocator-policy eligibility requires organizer approval; never count as competition win merely from diagnostic. Server instrumentation removed and processes stopped.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Preserve raw logs; independently verify allocator graph/lifetime correctness and evaluate cache-growth pressure only if budget allows; organizer approval required for any scored allocator-policy candidate.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
