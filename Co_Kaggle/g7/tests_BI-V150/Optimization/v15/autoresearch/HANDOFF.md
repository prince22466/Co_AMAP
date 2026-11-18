# v15 latest research handoff

Status: The November 17 allocation ended. The user later confirmed a November 18 allocation through 17:00 Warsaw time; consult the current budget and allocation status before further GPU experiments. Neither the grouped MM candidate nor the sampler candidate has a verified end-to-end throughput improvement or final accuracy result; both remain research-only.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35; sampler buffer-reuse candidate source commit: 1fd9d417. Grouped MM improved a wide projection in direct kernel tests, while its serving screens were order-sensitive. Sampler buffer reuse increased measured KV capacity, but its 16K serving screens did not show a gain. An initial grouped worker-attribution probe stopped before a request; a later diagnostic completed 8/8 requests and captured 672 grouped-specialized GPU kernel events. This confirms execution, not a serving gain. See EXPERIMENTS.md and PRs #255 and #256 for evidence, test gaps and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource. The rejected log-domain sampler fusion changes token selections and must not be integrated.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-18T10:51:05+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 0c6598327d4620696520ae73b55dd4e1e2101eeb

Remaining GPU time: unknown

Finding: Clean sampler serving screen completed; half-runs and accuracy remain outstanding

Evidence/provenance: Instrumented candidate-first, original control second, same model/server/benchmark flags, seed and 4K8/16K8/4K64/16K64 sequence. All cases 8/8,8/8,64/64 succeeded. Candidate/control 4K64 2300.91/2169.98 tok/s (+6.03%), TTFT13211.58/13286.78 ms, TPOT625.04/691.31 ms, prefix hits32640/32640, zero new preemptions. 16K64 3476.71/3382.25 (+2.79%), TTFT122484.05/129160.66, TPOT612.82/603.53, prefix hits196448/184160, zero new preemptions. Candidate attribution logger was enabled, control was not; 16K prefix-hit counts differ. Diagnostic only, no half-run or verified improvement; raw archive SHA256 fcf90fbf92bb29c4b304bd95be6b5e84f489d56d419f2fbcda1dbd7e470d02f5 backed up locally. Uninstrumented candidate after clean control: 4K 2236.28/2169.98 tok/s (+3.06%); 16K 3468.66/3382.25 (+2.55%). All 64 requests succeeded in each pressure case; prefix hits matched within each pair, no new preemptions. 16K TPOT 613.15/603.53 ms (+1.59%). One clean candidate and one control, short 128-output screens; no verified half-run gain.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Run matched unchanged control and candidate half-runs, independently replicate any gain, complete sampler correctness/eligibility review and require MATH-500 before final acceptance. The user intends to run the half-runs manually.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
