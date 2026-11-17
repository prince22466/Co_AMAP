# v15 latest research handoff

Status: GPU research completed for the November 16 allocation; no verified end-to-end throughput improvement or final accuracy result. Grouped MM remains a research-only candidate with inconclusive serving results.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original source commit: 1b680a7781ff0ba07c12d681708695fd5ccda55e. Grouped MM candidate source commit: 342e8d35. Direct wide-projection kernel timing improved, but candidate-first and control-first serving screens disagreed. The later worker-attribution probe was stopped before completing a request. See EXPERIMENTS.md and draft PR #255 for measurements and limitations. Separate four-warp and three-stage GEMM probes regressed; their evidence is on research/v15-gemm-resource.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T21:02:37+00:00

Status: blocked (researcher-reported)

Source HEAD before checkpoint: 17a49650db5cdfb3bad10ea189657139090d5629

Remaining GPU time: unknown

Finding: Stop grouped worker attribution to conserve credits

Evidence/provenance: User requested findings be recorded as credits run out. Instrumented grouped worker attribution was stopped during graph capture before a 4K request; no production request or kernel attribution result, so do not count it as completed. GPU idle and temporary MM logger restored. Existing grouped direct kernel, balanced serving, sampler capacity and pressure findings are preserved in draft PRs 255 and 256 and separate evidence archives. Four-warp and stage3 negative experiments preserved on research/v15-gemm-resource.

Small CSV evidence committed when --push works: none (external backup required)

Next action: On a fresh allocation with confirmed deadline, verify actual candidate worker kernel execution, then balanced matched serving and guarded half-runs; investigate larger real non-KV workspace reductions and preserve exact sampling semantics.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
