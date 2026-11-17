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

## 2026-11-16 UTC: targeted production GEMM frequency probe (planned)
Closest prior: v7 distinct shape inventory and v14 47% 4K early GEMM share. Missing: phase-specific MM shape call weighting. New measurement: temporary MM dispatch logging during a short 4K serving diagnostic; Python dispatch counts may miss graph replay and are not timings. Decision: guide a genuinely new K-loop/resource probe.

### Research checkpoint 2026-11-16T17:00:07+00:00: provisional

- HEAD before checkpoint: d356e5b158e3006972d44313f769072823b95c0e
- Budget remaining: unknown
- Finding: 4K MM Python dispatch shape frequency diagnostic
- Evidence and artifact paths: Baseline source 1b680a7781ff0ba07c12d681708695fd5ccbf5eea18787d was not edited except temporary logger; correct pinned commit is 1b680a7781ff0ba07c12d681708695fd5ccda55e. Instrumented 4K quick screen: 8/8 requests, 1786.12 total tok/s, mean TTFT 10553.19 ms, mean TPOT 64.71 ms; NOT an uninstrumented baseline. After server startup, Python logger saw 672 each M=2048 K=2048 N=2560,2048,12288 and M=2048 K=6144 N=2048, plus 42 each of these at M=57. Logger does not observe graph replay; phase markers overwritten so interval is after application startup, including warmup; no representative full-run frequency claim. Raw startup log SHA256 8f2247c3eebb71ce4794211b33bf1670014622d14487c775ffd8aaec67090c99 and bench log 9e72c34de58a826d3132317dcbeabdb290ad694895cb597b21f265822f974c9a backed up locally; logger removed and server stopped.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Profile resource bottleneck of observed 2048x2048x12288 MM; use native counters and test only new K-loop hypothesis.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-16 UTC: native resource attribution for wide prefill MM (planned)
Prior: v7 and v11 grouped ordering improved this shape in isolation but did not yield a replicated serving gain; stages 1/2 are existing autotune candidates, and no K-loop resource bottleneck has been established. New question: selected launch configuration, generated resources and native memory/occupancy counters for M=2048 K=2048 N=12288 with production B stride. Smallest decisive probe: direct original-v15 kernel launch with fixed production layout, check numerical correctness and inspect selected kernel and profiler capabilities. Decision: only pursue a distinct dataflow or staging mechanism if the counters identify a plausible bottleneck; no repeat grouping sweep.

## 2026-11-16 UTC: grouped wide MM A/B (requested follow-up; planned)
Prior v7 and v11 grouped wide MM reduced kernel latency, but v11 short-serving A/B was inconclusive and a second projection regressed. New evidence: this allocation's 4K short request window shows 672 Python dispatches per M=2048 projection family, including the wide projection; graph replay not counted. User explicitly requested grouped scheduling experiments. Candidate uses a general wide-output/large-M/transpose-layout resource predicate with fallback, not benchmark-length dispatch. Test direct all four production shapes plus irregular fallback against FP32, alternating warmed events, then only consider serving A/B if correct and total projected gain is material. This reopens the old mapping at the user's request; it does not assume v11 benefit transfers or claim novelty of the mechanism.

### Research checkpoint 2026-11-16T17:06:25+00:00: provisional

- HEAD before checkpoint: 342e8d359730c4c06d325e821a9f0f7464bd1b88
- Budget remaining: unknown
- Finding: Grouped MM kernel gain with direct correctness and graph replay
- Evidence and artifact paths: Wide MM M=2048 K=2048 N=12288 median control1/candidate1/control2/candidate2 1.40987/1.17073/1.3928/1.1703 ms, 15 warm event timings each; other production shapes and irregular fallback were nearly unchanged. Direct FP32-reference BF16 checks passed on four shapes plus fallback (atol .25 rtol .02); grouped odd M/N tail and three graph replays passed for FP16/BF16. Native original-wide capture: 76.41 percent TCU efficiency, 25.14 percent occupancy, 97 vector registers per thread, 65536 bytes shared memory, launch 384 blocks x 1024 threads; selected 256x256x32, 16 warps, 2 stages. Only kernel evidence, sequential not alternating per sample, no serving result or quality claim. Prior v11 serving was inconclusive; no throughput-goal claim.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Back up raw logs; run matched short 4K and 16K serving screens, confirm production candidate execution and keep latency guardrails.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-16T17:29:22+00:00: rejected

- HEAD before checkpoint: b6fa34e84ab9d7312b716194f4227c4103879a51
- Budget remaining: unknown
- Finding: Reject grouped MM candidate on 4K pressure screen
- Evidence and artifact paths: Candidate source commit 342e8d35; low-concurrency matched 8/8 each control/candidate 4K 1822.06/1804.21 tok/s, TTFT 10257.89/10375.08 ms, TPOT 64.11/64.62 ms; 16K 3557.37/3597.35 tok/s, TTFT 16756.01/16467.06 ms, TPOT 155.62/154.65 ms. Candidate-first 4K pressure 64/64 each control/candidate 1934.69/1797.72 tok/s (-7.08%), TTFT 24690.07/24288.74 ms, TPOT 709.55/807.14 ms (+13.75%); no failures. One pressure pair; no replication, causality or official score claim. Reject for guardrail; no half-run. Source hashes and logs available; evidence archive SHA256 d8f8342b8f5b636cf6f0b02f0fd7bd87108fec6fef4b7f2f61fa77a9c75fc590 backed up locally. Actual worker candidate-kernel launch not independently proven; source path alone is not attribution.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Keep candidate PR as rejected/provisional kernel evidence; switch to clean baseline and investigate a genuinely different GEMM or memory-lifetime hypothesis.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-16 UTC: reverse-order grouped 4K pressure replication (planned)
Prior candidate-first matched pressure pair was -7.08% throughput with +13.75% mean TPOT. Subsequent sampler pressure experiment found material order/noise effects and motivates one control-first replication. Same 4096 input, 128 output, concurrency/prompts 64, seed42, flags and cache protocol. Decision: determine whether the grouped result remains a regression or is inconclusive; this is not a half-run or a new candidate.

### Research checkpoint 2026-11-16T20:57:32+00:00: provisional

- HEAD before checkpoint: 42e81ce380aa3aeee5d27a1ce190abc88afd3e19
- Budget remaining: unknown
- Finding: Grouped MM reverse-order pressure screen is inconclusive
- Evidence and artifact paths: Control-first 4K pressure 4096 input, 128 output, 64 prompts/concurrency, seed42, fixed flags. Candidate/control total 2031.52/2011.02 tok/s (+1.02%), TTFT 23978.47/24527.79 ms (-2.24%), TPOT 675.56/681.84 ms (-0.92%); 64/64 successes each, no failures. Earlier candidate-first pair 1797.72/1934.69 tok/s (-7.08%) and +13.75% TPOT. The reversed result and earlier regression indicate order or run variability; no repeatable serving gain or causal attribution is established, and a single positive diagnostic is not target achievement. Candidate remains provisional/research-only; no unchanged half-run. Raw archive SHA256 aa521f9e2cfb450af383f7f5bffae61dc80500cecd9a4aef96f1f258f5da4449 backed up locally.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Do not promote without balanced independent serving replication, production kernel attribution and matched 4K/16K half-runs; restore original source.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-16 UTC: grouped worker dispatch attribution (planned)
Prior: grouped kernel isolated gain, serving pressure -7.08% candidate-first and +1.02% control-first. Missing: direct evidence that the production worker reaches grouped-eligible MM calls during a 4K request. Temporary log at the actual guarded launch, one short 4K diagnostic and post-start log count; instrumentation distorts performance and does not itself prove emitted kernel identity or graph replay attribution. Decision: determine whether future independent kernel tracing is warranted; no throughput claim.

### Research checkpoint 2026-11-16T21:02:37+00:00: blocked

- HEAD before checkpoint: 17a49650db5cdfb3bad10ea189657139090d5629
- Budget remaining: unknown
- Finding: Stop grouped worker attribution to conserve credits
- Evidence and artifact paths: User requested findings be recorded as credits run out. Instrumented grouped worker attribution was stopped during graph capture before a 4K request; no production request or kernel attribution result, so do not count it as completed. GPU idle and temporary MM logger restored. Existing grouped direct kernel, balanced serving, sampler capacity and pressure findings are preserved in draft PRs 255 and 256 and separate evidence archives. Four-warp and stage3 negative experiments preserved on research/v15-gemm-resource.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: On a fresh allocation with confirmed deadline, verify actual candidate worker kernel execution, then balanced matched serving and guarded half-runs; investigate larger real non-KV workspace reductions and preserve exact sampling semantics.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## Review correction: baseline commit identifier

The earlier 4K shape-attribution checkpoint contains a mistyped baseline identifier beginning `1b680a77` and ending `5ccbf5eea18787d`. The correct original v15 commit is `1b680a7781ff0ba07c12d681708695fd5ccda55e`; the original FlagGems tree is `7cdae638186b8dba929cbfb1a4ccbf5eea18787d`. The affected checkpoint itself also states the correct commit. This correction does not change the recorded measurements.
