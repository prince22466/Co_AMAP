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
