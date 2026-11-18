# v15 latest research handoff

Status: The confirmed GPU allocation and its November 17 extension have ended; a newly confirmed allocation is required before further GPU experiments. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. An initial grouped worker-attribution probe stopped before a request; a later diagnostic completed 8/8 requests and captured 672 grouped-specialized GPU kernel events. This confirms execution, not a serving gain. See EXPERIMENTS.md and draft PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-18T10:51:05+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 0c6598327d4620696520ae73b55dd4e1e2101eeb

Remaining GPU time: unknown

Finding: First sampler serving pair promising; clean reverse order in progress

Evidence/provenance: Instrumented candidate-first, original control second, same model/server/benchmark flags, seed and 4K8/16K8/4K64/16K64 sequence. All cases 8/8,8/8,64/64 succeeded. Candidate/control 4K64 2300.91/2169.98 tok/s (+6.03%), TTFT13211.58/13286.78 ms, TPOT625.04/691.31 ms, prefix hits32640/32640, zero new preemptions. 16K64 3476.71/3382.25 (+2.79%), TTFT122484.05/129160.66, TPOT612.82/603.53, prefix hits196448/184160, zero new preemptions. Candidate attribution logger was enabled, control was not; 16K prefix-hit counts differ. Diagnostic only, no half-run or verified improvement; raw archive SHA256 fcf90fbf92bb29c4b304bd95be6b5e84f489d56d419f2fbcda1dbd7e470d02f5 backed up locally. Uninstrumented candidate repeat running after clean control.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Complete clean repeat, compare both workloads and cache state, then check two-run budget and all correctness/rules gates before half-run.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
