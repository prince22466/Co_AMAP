# v15 experiment ledger (append only)

## Setup — no v15 GPU experiments run

Original v15 production source equals v14 (both tree hashes recorded in AGENTS.md). Historical evidence and ranked hypotheses are in RESEARCH_QUEUE.md; no historical measurement is a fresh v15 control.

For each entry record: UTC; original/control/candidate source SHA and clean diff; hypothesis; exact command and immutable config/script hash; actual imported worker and kernel; correctness and skipped cases; shape-weighted A/B or memory accounting; raw artifact paths/checksums; 4K/16K throughput, completed requests, TTFT/TPOT, KV blocks, preemptions and recompute as applicable; noise/replication and quality/rule status; decision; remaining allocation; next action. Mark failed, blocked or incomplete attempts explicitly.

## Prior-work boundary (planning update; not a new GPU result)

v7 grouped MM/edge masking/Linear accumulation, shape inventory and operator A/B are completed prior work; v11 selective grouped-MM experiments did not establish a reproducible serving win and the second projection extension regressed. See RESEARCH_QUEUE.md for links and exact scope. Do not repeat these experiments or automatically stack their code; spend the allocation on new GEMM dataflow/resource/epilogue hypotheses and genuine KV capacity savings.

## Prior-work audit v0–v14 (planning update; no new GPU result)

The research queue now registers all previous versions, v8 rejected stacks and existing v14 startup/queue/prefix/cache evidence. Before each new GPU experiment record closest prior result, material novelty, specific missing evidence, smallest decisive test and decision. Do not spend GPU time on old measurements, old patches or generic profiling. New candidate correctness and matched performance validation remain required.

### Research checkpoint 2026-11-16T16:47:52+00:00: provisional

- HEAD before checkpoint: 1b680a7781ff0ba07c12d681708695fd5ccda55e
- Budget remaining: unknown
- Finding: Start v15 allocation; pin clean baseline and budget
- Evidence and artifact paths: Baseline 1b680a7781ff0ba07c12d681708695fd5ccda55e; FlagGems tree 7cdae638186b8dba929cbfb1a4ccbf5eea18787d; plugin tree b481b8fa8b41712267262ce678adba8132a3043f; benchmark SHA256 873f17f932d4643222be9903ba3b569c1863c4e2d246ceafbe6f75ebe7163ffc; BI-V150 32 GiB; UTC deadline 2026-11-16T22:00:00Z; 27 helper tests pass; shell imports point to checkout; no GPU experiment yet.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Target unexplained non-KV startup allocations and inspect production GEMM resource evidence; apply novelty gate before GPU tests.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-16 UTC: targeted startup allocation attribution (planned)
Closest prior: v14 startup and telemetry established 521840 KV tokens, 4.69 GiB checkpoint, and high 16K occupancy, but did not report a complete non-KV breakdown. Distinct unknown: weights, peak activations, non-torch allocation, and graph-pool estimate/actual with unchanged serving configuration. Smallest test: temporary diagnostic logging at the existing accounting boundary and one owned startup, no workload or allocation changes. Decision: inspect largest reclaimable lifetime before proposing any memory optimization. This does not repeat queue or capacity telemetry, establish a gain, or alter accounting.

### Research checkpoint 2026-11-16T16:53:19+00:00: provisional

- HEAD before checkpoint: f896b5e2a72aa702b9929005115a31b9b70703fc
- Budget remaining: unknown
- Finding: Measure v15 startup non-KV memory breakdown
- Evidence and artifact paths: Baseline source 1b680a7781ff0ba07c12d681708695fd5ccda55e; temporary diagnostic worker logging only, removed after owned server shutdown; fixed flags per program.md; requested 29205777612 bytes, weights 5101233152, activation peak 656424960, non-torch 1000341504, graph estimate and applied 0, available KV 22447777996, 521936 KV tokens; full decode graph capture completed. Raw startup log SHA256 36a195657709a0c5a82a8eaab23a39efd04d9a658f781ba81f8c0f97603d45f5, backed up locally. No reclaimability or throughput gain established; actual graph pool bytes not isolated.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Find specific reclaimable lifetime or obtain targeted GEMM resource evidence, then test genuinely new candidate.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.
