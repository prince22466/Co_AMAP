# v12 autoresearch handoff

**Status: provisional research candidate; no verified official throughput improvement.** User shortened the remaining work window to 40 minutes at approximately 15:07 UTC on 2026-11-11. GPU allocation ends at 16:00 UTC. The original budget guard rejects a half-run because it requires at least 150 minutes.

## Current state

- The remote server used for diagnostics was stopped and the original operator and mixed-paging test restored. Original remote operator SHA-256: `6e3e8f7cbc24140a9cfd3ef3d2ed8d99cf7ae0f4e0147ac3d9d9b0775925931a`. The machine is an extracted source tree, not a Git checkout.
- The draft PR branch retains the tested one-stage BF16 GQA prefill pipeline candidate and its test expectation change. Candidate remote raw SHA-256: `1c65864df8333a6ff91f005b29547441ef128a33c847466c09accca0cf6a0d28`. Source commit `26b4cbc320f6f8648c35f6ab5ab9f24582c9dcf0`; test commit `d60c295317b6459bea545f6aa8ab100d2edde34b`. Neither has met the half-run, quality, or reproducibility gate; do not merge.
- GPU allocation has one BI-V150. Tensor and FlagGems modules imported from v12. The unmodified benchmark SHA-256 is `873f17f932d4643222be9903ba3b569c1863c4e2d246ceafbe6f75ebe7163ffc`. Original helper tests: 21 passed; original and candidate GPU suites: 21 passed each. Sampled FP16 fallback outputs matched the original. Human-owned MATH-500 and broader quality tests were not run.

## Evidence and limitations

The GPU trace records `kernel_unified_attention` at 71.672% of summed kernel time across its captured interval. This is not a wall-clock or phase-specific share. Separate kernel profiling on one 16K-shaped prefill reports 114 -> 97 vector registers and mean 17.379 -> 15.122 ms; no occupancy or reliable spill accounting. An instrumented diagnostic server showed the v12 stage-one path invoked for a 2048-token prefill; this probe's timing is not serving evidence.

| Short diagnostic | 4K total tok/s; TTFT / TPOT ms | 16K total tok/s; TTFT / TPOT ms |
|---|---|---|
| Control A1 | 2459.48; 6038.05 / 60.03 | 3196.27; 19050.60 / 170.62 |
| Candidate B1 | 2263.25; 7245.44 / 59.91 | 3567.85; 16710.78 / 155.12 |
| Control A2 | 2460.35; 6022.39 / 60.10 | 3195.17; 19069.86 / 170.55 |
| Candidate B2 | 2500.87; 5926.09 / 59.13 | 3546.01; 16764.00 / 156.52 |
| Candidate B3 | 2512.00; 5883.39 / 58.99 | Not run |

All completed requests in these short diagnostics. They use concurrency 8 and output length 128, not the official benchmark. The 16K short gain repeated, but 4K varied, including one material regression. No unchanged official half-run, fresh full control, TTFT-limit validation, score calculation, independently replicated half-run, or accuracy verification has been performed.

Four PR #237 sparse-reference tests failed. Original dense and a full-window sparse variant were identical for one case but disagreed with an FP32 CPU oracle; GPU and CPU references agreed and shuffled page reconstruction was checked. The cause is unresolved. Sparse output, middle-context retrieval, MATH-500 and organizer eligibility remain unverified.

Rejected kernel-only investigations: prefill page-table reuse, 128/32 query tiles and pipeline stage 3; explicit four-warps gave no meaningful gain. A 64-token KV tile improved one short-query kernel but substantially regressed longer prefill shapes, so it is only a narrow future research lead.

## Evidence and next action

The append-only `EXPERIMENTS.md` records measurements and limitations. Remote raw traces, SQLite, test logs, candidate sources and client logs are under `/root/Optimization/v12/.autoresearch-work/`; the off-instance backup is under the local workspace `Optimization/v12/.autoresearch-work/remote_backup/`. The production trace and engine SQLite backup SHA-256 hashes are recorded in the ledger. A final refresh of the backup is in progress.

Next allocation: investigate dense-vs-FP32 discrepancy, confirm production invocation without changing benchmark behavior, extend mixed/paged/decode and retrieval validation, and reproduce both scenarios with matched restarts. Only if the short-serving and quality gates pass, obtain a confirmed allocation deadline and use the unchanged guarded half-run for fresh control and candidate, with independent replication and a 25-minute preservation reserve. Do not auto-merge.
