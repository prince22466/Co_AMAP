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

## 2026-11-16 UTC: sampler workspace fusion feasibility (planned)
Closest prior: v15 sampler phase adds 522207232 bytes to cumulative peak, though individual buffers are not attributed. The native sampler computes softmax, exponential noise and argmax. Test a two-stage logit-minus-log-noise argmax using identical noise, comparing outputs, memory and speed. This is an isolated feasibility test; bitwise and RNG semantics must be verified before any integration.

### Research checkpoint 2026-11-16T19:12:18+00:00: rejected

- HEAD before checkpoint: c8da3be15512f0cb3d2e722a832da3bf33d13de8
- Budget remaining: unknown
- Finding: Reject sampler log-domain fusion on exact token semantics
- Evidence and artifact paths: Standalone prototype uses identical precomputed exponential noise. Normal random rows 8x10003 and 256x130560 yielded zero token mismatches; large baseline/candidate median 2.153/0.408 ms, temporary peak 134221824/67584 bytes, excluding original logits and noise buffers. It mismatched existing sampler for +inf and NaN logits and for 14956 of 65536 constructed finite close-noise ties; therefore exact generation behavior is not preserved. No production integration, no KV block or serving measurement; timings do not justify promotion. Scripts and raw logs preserved. Original sampler and research branch production source unchanged.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Investigate exact softmax-and-noise workspace fusion or other real memory reductions only if numerical equivalence can be maintained; preserve current KV accounting.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-16 UTC: sampler in-place softmax feasibility (planned)
Closest prior: log-domain fusion reduced memory but mismatched reference tokens. New mechanism: native torch.softmax with its output aliased to the mutable FP32 logits buffer, preserving same exponential noise and division/argmax; native top-k/top-p API permits in-place logits updates. Missing evidence: vendor backend supports aliased softmax correctly; exact output, unusual values, graph replay, logprob modes and caller ownership; peak allocation impact. Test standalone against out-of-place reference first. Do not integrate unless runtime and all invariants hold.

### Research checkpoint 2026-11-16T19:19:17+00:00: provisional

- HEAD before checkpoint: 092dc0aea560caa0c098619bef985665ac1c623e
- Budget remaining: unknown
- Finding: Sampler in-place softmax adds KV blocks in startup probe
- Evidence and artifact paths: Candidate source 1fd9d417. Direct seeded tests matched native token IDs for unfiltered, top-k, top-p and per-row generator cases, and nonfinite behavior; processed-logits fallback and three graph replays passed. Actual diagnostic worker invoked candidate twice on 256x130560. Measured original/candidate activation peak 656424960/522207232 bytes; non-torch 1000341504/1004535808; KV available 22447777996/22577801420 bytes; 521936/524960 KV tokens, +3024 tokens (+189 blocks, +0.58%). Same fixed server flags, one startup each, temporary provenance/accounting loggers removed before use. Candidate actual device memory after capture 28476 MiB; source and tests committed separately. No matched serving or independent block replication; exact logits alias side effects, processed modes, mixed RNG paths and competition eligibility still need review. Baseline graph pool actual 228589568 bytes. Candidate startup log SHA256 a6b5587dcd19e0eb0e7a0d06bba9d8050b1caa0cbe3cb0b4140b1622f3b894be.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Back up raw evidence; run clean-source 4K/16K serving checks and a 16K concurrency-64 pressure screen; replicate startup block allocation and expand semantics tests.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.
