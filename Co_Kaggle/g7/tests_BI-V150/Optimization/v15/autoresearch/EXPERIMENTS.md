# v15 experiment ledger (append only)

## Setup — no v15 GPU experiments run

Original v15 production source equals v14 (both tree hashes recorded in AGENTS.md). Historical evidence and ranked hypotheses are in RESEARCH_QUEUE.md; no historical measurement is a fresh v15 control.

For each entry record: UTC; original/control/candidate source SHA and clean diff; hypothesis; exact command and immutable config/script hash; actual imported worker and kernel; correctness and skipped cases; shape-weighted A/B or memory accounting; raw artifact paths/checksums; 4K/16K throughput, completed requests, TTFT/TPOT, KV blocks, preemptions and recompute as applicable; noise/replication and quality/rule status; decision; remaining allocation; next action. Mark failed, blocked or incomplete attempts explicitly.

## Prior-work boundary (planning update; not a new GPU result)

v7 grouped MM/edge masking/Linear accumulation, shape inventory and operator A/B are completed prior work; v11 selective grouped-MM experiments did not establish a reproducible serving win and the second projection extension regressed. See RESEARCH_QUEUE.md for links and exact scope. Do not repeat these experiments or automatically stack their code; spend the allocation on new GEMM dataflow/resource/epilogue hypotheses and genuine KV capacity savings.

## Prior-work audit v0–v14 (planning update; no new GPU result)

The research queue now registers all previous versions, v8 rejected stacks and existing v14 startup/queue/prefix/cache evidence. Before each new GPU experiment record closest prior result, material novelty, specific missing evidence, smallest decisive test and decision. Do not spend GPU time on old measurements, old patches or generic profiling. New candidate correctness and matched performance validation remain required.

## 2026-11-16 UTC: non-KV allocation lifetime and graph pool attribution (planned)
Prior: v14 521840 KV tokens, v15 startup probe weights 5101233152, activation peak 656424960, non-torch increase 1000341504, zero applied graph estimate; these totals do not identify lifetimes. New evidence required: initial versus pre/post-profile non-Torch/reserved and actual captured graph bytes. Probe the existing snapshots and graph capture return without changing accounting or flags. Decision: pursue only an identifiable duplicate or temporary allocation whose removal increases actual KV blocks safely.

### Research checkpoint 2026-11-16T17:34:11+00:00: provisional

- HEAD before checkpoint: 1b680a7781ff0ba07c12d681708695fd5ccda55e
- Budget remaining: unknown
- Finding: Attribute startup non-KV and graph allocations
- Evidence and artifact paths: Original v15 1b680a7781ff0ba07c12d681708695fd5ccda55e, unchanged flags; initial non-torch 153092096 bytes; before profiling non-torch 1092616192, after profiling 1153433600; torch reserved before/after 5226102784/5500829696; weights 5101233152; peak activation 656424960; available KV 22447777996 bytes; actual graph capture return 228589568 bytes with reported estimate 0 on this platform. Most non-torch growth predates profiling, 60817408 bytes arose between pre/post snapshots. No part proven reclaimable; do not relax memory accounting. Temporary logger removed, owned server stopped. Raw log SHA256 7ca50516c0f1175e1554e389c85963581741b6dff5d9b9293ba032946a0a2771, backed up locally.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Identify the peak allocation phase inside profile_run, then pursue only a real live-workspace reduction.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-16 UTC: allocation peak phase attribution (planned)
Closest prior: startup profiling shows 656424960-byte peak activation but no attribution to model forward versus logits/sampler. New evidence: log cumulative peak and live/reserved bytes at the existing dummy forward and sampler boundaries, including logits allocation; do not reset stats or change allocation decisions. Decision: identify an operator-owned transient that can be reduced in real serving as well as profiling; reject any change that merely shrinks the diagnostic workload.

### Research checkpoint 2026-11-16T19:10:08+00:00: provisional

- HEAD before checkpoint: 2aa3cf10028a7d642f2bf2a968e46538cb9c98be
- Budget remaining: unknown
- Finding: Attribute activation peak to sampler
- Evidence and artifact paths: Original v15 source, diagnostic logging only, no resets to peak statistics: after dummy model forward allocated 5161098752, reserved 5395972096, cumulative peak 5252322816 bytes; after logits allocated 5229256192, reserved 5463080960, peak 5252322816; after dummy sampler allocated 5371865600, reserved 5999951872, cumulative peak 5774530048. The cumulative peak rises 522207232 bytes during sampling, but does not identify the exact live buffers. Dummy sampler tests top-p/top-k and per-request RNG paths; actual benchmark sampling semantics are not established. Raw log SHA256 48d04a812a537c6b42d742d45347e2c94e8a1a2428e51e11845a1dd3b7c584db backed up locally. Logger removed and owned server stopped.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Identify sampler live allocations and determine if exact generation semantics can be retained while reducing real workspace; do not merely reduce profile workload.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.
