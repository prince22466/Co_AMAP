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

### Research checkpoint 2026-11-16T19:41:08+00:00: rejected

- HEAD before checkpoint: d6f04770d6766be5fc96a6b9239cae93bb6289a7
- Budget remaining: unknown
- Finding: Sampler reuse fails 16K pressure screen despite extra KV blocks
- Evidence and artifact paths: Candidate source 1fd9d417, candidate-first matched restart. Candidate/control KV tokens 525056/521936 (+3120, 195 blocks, +0.60%). 4K low concurrency candidate/control 1805.59/1814.49 tok/s, mean TTFT 10365.94/10285.81 ms, TPOT 64.57/64.49 ms, 8/8 each. 16K low concurrency 3556.83/3544.86 tok/s, TTFT 16746.18/16746.63 ms, TPOT 155.75/156.73 ms, 8/8 each. 16K pressure concurrency 64 candidate/control 3282.76/3490.77 tok/s (-5.96%), mean TTFT 139102.68/122267.96 ms (+13.77%), TPOT 581.95/568.36 ms (+2.39%), 64/64 each, no failures. One pressure pair, not causal proof or independent replication; fails guardrails. Prior direct seeded and graph checks limited; no complete correctness or quality coverage. No half-run. Raw evidence archive SHA256 8f556eb192c7dbcdd4cec3868bf90416ddb11f8f6a886989d02819319402f3ec backed up locally.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Keep PR as experimental evidence, reject promotion on current screen and revert to original v15 before new research.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-16T20:02:21+00:00: rejected

- HEAD before checkpoint: ff0805f2d9a7b496d12d59b4b6c113443119dbe3
- Budget remaining: unknown
- Finding: Reverse-order sampler serving screen also below control
- Evidence and artifact paths: Control-first matched restart, 8/8 short cases and 64/64 pressure cases with zero failures. 4K candidate/control 1806.90/1807.88 tok/s, TTFT 10333.14/10317.14 ms, TPOT 64.73/64.78 ms. 16K candidate/control 3557.90/3555.52 tok/s, TTFT 16743.76/16734.46 ms, TPOT 155.69/155.94 ms. 16K pressure candidate/control 3565.89/3664.46 tok/s (-2.69%), TTFT 127515.54/122818.35 ms (+3.82%), TPOT 559.51/542.79 ms (+3.08%). Prior candidate-first pair was -5.96%; differences are noisy and not causal proof, but neither supports a serving gain and both throughput screens exceed the 2% guardrail. No half-run; candidate remains research-only. Raw archive SHA256 e1e3bc1c1e6e2d668e943dfc0aa4f72a96b918c0b2f30983ec6a5139e7b1936c backed up locally.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Return to original v15; seek genuinely different GEMM or allocation mechanism, subject to remaining budget and rules.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-16T20:42:29+00:00: rejected

- HEAD before checkpoint: fc31cfa8296efcaf2d162ac1980e7d1104e172be
- Budget remaining: unknown
- Finding: Longer 16K cache-growth diagnostic does not show sampler gain
- Evidence and artifact paths: Control-first matched diagnostic 16384 input, 1024 output, 64 prompts/concurrency 64, seed42, original flags. Candidate/control: 1768.94/1826.72 total tok/s (-3.16%), TTFT 225604.12/218122.45 ms (+3.43%), TPOT 239.72/229.25 ms (+4.57%); 64/64 successes each, zero failures, 2 preemptions each and zero prefix hits. One pair, no independent replication or official score. Both reached ~100 percent KV occupancy; extra candidate blocks did not eliminate preemption or produce gain. Raw archive SHA256 c81cc7da2a49e2346fec79ea6de5ab656daf435a6d4a23f873f430b0b7f3a3a9 backed up locally. Prior short pressure pairs -5.96% and -2.69%; candidate remains rejected and research-only.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Restore original production source and continue only genuinely new small experiments while preserving confirmed allocation reserve.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

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

### Research checkpoint 2026-11-17T11:50:03+00:00: provisional

- HEAD before checkpoint: 063f1fb7d079cb276106af81fcc4c3b64dbc3793
- Budget remaining: unknown
- Finding: Start continuation allocation with original v15 production control
- Evidence and artifact paths: User confirmed 2026-11-17 16:00 UTC deadline on new host 2309343dd3b8; BI-V150 32GiB idle. Production control commit 063f1fb7d079cb276106af81fcc4c3b64dbc3793, identical source trees to original 1b680a7781ff0ba07c12d681708695fd5ccda55e: FlagGems 7cdae638186b8dba929cbfb1a4ccbf5eea18787d and plugin b481b8fa8b41712267262ce678adba8132a3043f. Parent includes merged 255/256 evidence; candidates isolated out of control. Prior session budget archived; benchmark SHA256 873f17f932d4643222be9903ba3b569c1863c4e2d246ceafbe6f75ebe7163ffc, RUNS=2 SKIP_FIRST=0 and helper supplies exact cases. Fresh matched half-run pair not admitted at 250 minutes; reserve 25 minutes.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Inspect production MLP projection/activation execution and choose a novel fused epilogue probe.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-17 UTC: fused gate/up projection plus SwiGLU feasibility (planned)
Closest prior: v7 grouped MM/edge-mask/direct-accumulation, v11 selective grouping, and v15 grouped A/B did not establish a serving win; none fused gate/up GEMM with SwiGLU. Archived 4K trace shows mm_kernel plus a separate silu/multiply pointwise kernel; installed LlamaMLP executes merged gate_up_proj followed by SiluAndMul. Missing: whether two dot accumulators with a BF16-rounding-preserving activation epilogue save enough time and intermediate storage to offset resource pressure. Smallest decisive probe: compare original v15 MM plus fused activation against one two-accumulator Triton GEMM epilogue, identical weights/layout, warmed alternating events, M=2048 K=2048 output=2x6144 and a small ragged correctness case. Preserve BF16 rounding before activation. Reject if reference fails or chain time regresses. This is an isolated new operator/dataflow, not a change to dispatch, official flags, or benchmark.

## 2026-11-17 UTC: resource-matched fused SwiGLU tile (planned)
Prior immediate probe used 128x128, 8-warps, two accumulators and was exact on two BF16 cases but production-shaped chain 3.38967 vs baseline 1.94357 ms (+74.4%). This doubles the output program count relative to selected original wide 256x256 tile. New specific test: 256x128 with 16 warps and two 256x128 accumulators, preserving the original 256x256 output footprint, 384 programs, and equivalent A/B tile volume while fusing the activation. Same original MM+activation control and BF16 checks, alternating timings; reject if it does not outperform the full original chain. No generic tile sweep.

## 2026-11-17 UTC: production-like activation baseline check (planned)
The fused 256x128 prototype measured 1.893/1.887 ms vs MM plus FlagGems pointwise wrapper 1.938/1.938 ms in two screens, but v14 production uses a distinct compiler-generated triton_poi_fused_mul_silu_slice_1 (1764 calls, 241.05 ms, about 0.137 ms/call across observed early trace). Missing: whether wrapper overhead exaggerates the activation cost. Compare against original MM plus an explicit single-launch Triton slice/silu/mul kernel on the same inputs. This is a proxy, not proof of exact production generated kernel behavior.

### Research checkpoint 2026-11-17T11:54:44+00:00: rejected

- HEAD before checkpoint: bd666241cce6f861c285643a6ea21fa86a32b177
- Budget remaining: unknown
- Finding: Reject fused SwiGLU after production-like baseline check
- Evidence and artifact paths: Original production control 063f1fb7 with original source trees. Fused BF16 output exactly matched baseline on M33 K96 N257 and M2048 K2048 N6144. First 128x128 8-warp fused tile was 3.38967 ms vs MM plus FlagGems wrapper 1.94357 ms. Resource-matched 256x128 16-warp fused tile measured 1.89325 and 1.88678 ms vs wrapper baseline 1.93786 and 1.93790 ms in two alternating screens, but wrapper exaggerates production activation cost: original MM plus explicit single-launch Triton activation proxy measured 1.54745 ms vs fused 1.89528 ms (+22.48% latency), BF16 outputs exact. v14 real compiled activation averaged 241.05 ms /1764 calls =0.137 ms/call; proxy is not proof of identical production kernel, but rules out initial gain claim. No integration, no serving, FP16/strides/graph/accuracy not tested. Exact scripts and raw logs preserved.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Investigate a different resource mechanism or small-M/vocabulary bottleneck; do not promote this fused tile or count the wrapper result as a production gain.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-17 UTC: vocabulary Linear resource attribution (planned)
Prior v7 covered small-M grouping and direct accumulation and observed M=1/256 K2048 N130560 but did not establish a serving improvement. v14 4K middle rank0 trace includes 37 linear_kernel events totaling 42.05 ms; late graph window has visibility gaps and is not representative. New missing evidence: native cache/bandwidth, registers, shared memory and occupancy for the actual original vocabulary Linear at M1 and M256 with contiguous N-by-K weights. Smallest probe: standalone unmodified operator native counters for one representative launch each; no grouping, edge-mask, direct-dot experiment or shape sweep. Decision: only design a different output/epilogue or work-partition mechanism if counters show a concrete bottleneck.

## 2026-11-17 UTC: non-Torch growth by worker initialization phase (planned)
Prior v15 snapshots showed initial non-Torch 153092096 bytes, before-profile 1092616192, after-profile 1153433600; most growth (939524096 bytes) occurred before profiling. The responsible operation and reclaimability remain unknown. Novel targeted evidence: read-only MemorySnapshot calls immediately after runner construction and model loading, compared with existing initial and before-profile points, under unchanged server configuration. Stop after allocation profiling; do not change accounting. Decision: investigate only an identified reclaimable resource, never reduce safety reserves based on unproven free memory.

### Research checkpoint 2026-11-17T13:23:35+00:00: provisional

- HEAD before checkpoint: 8461eb2a6dbe6251cad9134a4dba84a728bf5ed6
- Budget remaining: unknown
- Finding: Attribute vocabulary Linear and non-Torch growth
- Evidence and artifact paths: Original v15 production trees. Native vocabulary Linear M1 K2048 N130560 selected 1x256x32 8 warps: 0.818 ms profiler duration, DRAM read efficiency 88.04%, TCU 10.77%, occupancy 24.83%; M256 selected 256x256x32 16 warps: 1.826 ms, DRAM 40.20%, TCU 76.27%, occupancy 24.56%. Profiler captures are diagnostic, warmup and serving impact unknown; no speedup. Initial worker snapshots: runner created non-Torch 264241152 bytes, model loaded 1096810496, profiling done 1157627904. Thus 832569344 bytes (~794 MiB) accumulated during model loading, with 60817408 bytes (~58 MiB) during profiling; it is not yet proven reclaimable. Instrumented server stopped and worker restored. Raw logs preserved off-instance after checkpoint.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Split model loading into initialization, weight loading and post-processing snapshots; design a genuine memory-lifetime candidate only if evidence identifies one.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-17 UTC: model-loader non-Torch attribution (planned)
Prior targeted snapshots show non-Torch bytes 264241152 after runner creation and 1096810496 after model loading, a growth of 832569344 bytes. Missing attribution: model object initialization, actual weight loading, and post-processing. Temporarily wrap those already existing loader calls inside v15-owned runner code to log CUDA free/total and torch allocated/reserved before and after each; no installed package edits, no allocation-policy change. Decision: identify the exact stage responsible before considering any real resource/lifetime optimization.

## 2026-11-17 UTC: device allocator overhead isolation (planned)
Prior targeted loader snapshots placed 832569344 bytes of non-Torch growth entirely inside initialize_model, before any weight loading or processing, but cannot distinguish driver allocator overhead from lazy library initialization or hidden model allocations. Novel minimal synthetic probe: compare free-device memory versus PyTorch reserved/allocated during one large and many smaller equally sized allocations in fresh standalone processes, synchronized with unchanged allocator configuration. Decision: if extra non-Torch growth tracks model-like allocation count/size, inspect safe buffer coalescing; if not, investigate initialization side effects. This does not alter production accounting or weights.

## 2026-11-17 UTC: expandable allocator feasibility (planned)
Prior synthetic standalone 5GB split across 500 x 10MB allocations grew non-Torch device use by 394264576 bytes; one 5GB allocation and 50 x 100MB did not. Model initialization alone increased the non-Torch estimate by 832569344 bytes. Missing: whether allocator segment expansion genuinely reduces this overhead without changing model computation, and whether the vendor runtime supports it. Smallest test: identical 500-allocation probe in a fresh process with PyTorch expandable_segments enabled, compare actual free/reserved bytes and allocation failures against existing default; no admission, cache, workload or server flag changes. If unsupported or no real savings, reject; before any integration establish runtime safety and competition eligibility.

### Research checkpoint 2026-11-17T13:27:40+00:00: provisional

- HEAD before checkpoint: 586252064ff732956b43b29c02265e194e4ac597
- Budget remaining: unknown
- Finding: Localize model non-Torch growth to allocation granularity
- Evidence and artifact paths: Baseline unmodified v15: non-Torch 264241152 bytes before initialize_model and 1096810496 immediately after (+832569344); weight loading and postprocessing did not change it. Fresh synthetic processes allocating 5GB: one 5GB allocation non-Torch unchanged; 50 x 100MB unchanged; 500 x 10MB raised non-Torch 394264576 bytes and released it when all tensors/cache freed. With PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, 500 x 10MB raised non-Torch by 0 and torch reserved 5012193280 vs default 5242880000. This is a synthetic feasibility result, not evidence of actual model KV blocks or throughput. Changing allocator policy may have unclear competition eligibility; diagnostic only until organizer approval. No flags, model data, cache accounting or official benchmark modified.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Measure actual model allocation, KV blocks, graph capture and reliability with the same server flags under allocator A/B; assess eligibility independently.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-17T13:49:07+00:00: provisional

- HEAD before checkpoint: a06e86030e0fb0ef25f008f6e5a6c9928e466fce
- Budget remaining: unknown
- Finding: Expandable allocation adds KV capacity; short serving near neutral
- Evidence and artifact paths: Diagnostic allocator policy only; original source 063f1fb7 production trees, unchanged model/server/benchmark flags. Candidate-first matched startup default/expandable: non-Torch increase 1000341504/67108864 bytes, activation peak 656424960/654032896, KV bytes 22447777996/23383402700, token capacity 521936/543696 (+21760, +1360 blocks, +4.17%). Both full decode graph captures completed. 4K 8/8 candidate/control 1785.25/1786.35 tok/s, TTFT 10508.31/10466.80 ms, TPOT 65.11/65.38 ms; 16K 8/8 3545.67/3529.18, TTFT 16755.13/16798.55, TPOT 156.60/157.63. 16K pressure 64/64 3662.88/3646.08 (+0.46%), TTFT 122457.78/122507.27, TPOT 541.53/544.34, zero failed requests, zero new preemptions and 130944 prefix hits each. One pair; no independent throughput replication, correctness or MATH-500; pressure uses warmed prefix cache and 128 output tokens. Allocator-policy eligibility requires organizer approval; never count as competition win merely from diagnostic. Server instrumentation removed and processes stopped.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Preserve raw logs; independently verify allocator graph/lifetime correctness and evaluate cache-growth pressure only if budget allows; organizer approval required for any scored allocator-policy candidate.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-17 UTC: expandable allocator correctness stress (planned)
Prior: allocator mode added 21760 KV tokens and completed graph startup; 16K pressure showed +0.46% in one candidate-first 128-output diagnostic with no preemptions. Missing: baseline/candidate behavior for allocation reuse, streams, graph replay and data isolation; production KV cancellation/preemption integrity remains untested. Test fixed-seed standalone Torch allocation/free/reuse and graph replay in fresh default and expandable processes, identical operations. Treat this as allocator-level smoke only; exact output and no errors are necessary but not sufficient for production or competition eligibility.

## 2026-11-17 UTC: expandable allocation longer 16K pressure diagnostic (planned)
Prior one candidate-first 16K 128-output screen gave +0.46% throughput, zero preemptions and 130944 prefix hits in each mode; insufficient to measure cache-growth effect. New control-first test uses unchanged server flags and model, 16384 input, 1024 output, concurrency/prompts 64, seed42; fresh servers, no preceding short workload. Sample live KV, preemptions and running/waiting; compare allocation, completion, TTFT/TPOT, throughput. It is not an official half-run and allocator mode has unconfirmed competition eligibility.

### Research checkpoint 2026-11-17T19:27:29+00:00: rejected

- HEAD before checkpoint: ecc9fee944c2d107d7ba7962b9cc69429b062f2d
- Budget remaining: unknown
- Finding: Expandable allocator long 16K diagnostic regresses
- Evidence and artifact paths: Original v15 production control 063f1fb7, unchanged server flags. Fresh control-first 16K diagnostic: 16384 input, 1024 output, 64 prompts/concurrency, seed42. Expandable/control 1678.40/1878.24 total tok/s (-10.64%), TTFT 228617.09/218196.08 ms (+4.78%), TPOT 264.75/222.17 ms (+19.17%). Both completed 64 requests, zero failures, zero prefix hits, and two preemptions each. Single pair is not causal proof or official half-run, but fails throughput and latency guardrails; no verified gain. Standalone allocation smoke: 80 reuse iterations, ten graph replays and matching output SHA256 9fb6b2c687b0d115530224460470c3cd950842303acb0c6f636ae44b403cc227 for both modes. This does not prove full KV lifecycle correctness. Allocator-policy eligibility remains unresolved; diagnostic only. Confirmed allocation expired 2026-11-17 16:00 UTC. Raw evidence backup required.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Do not promote allocator mode on these screens. Restore original source, back up logs and continue future work only on a new confirmed allocation.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-17 UTC: fused SwiGLU with weight-reuse tile ordering (planned)
Prior fused 256x128 two-accumulator kernel had exact BF16 output but took 1.895 ms versus 1.547 ms for original MM plus explicit single-launch activation. Previous v7/v11 grouping tested unfused GEMM, not the dual-accumulator fusion; v15 grouped serving was inconclusive. New causal hypothesis: the fused kernel loads separate gate/up weight tiles per row-major M tile; grouping M tiles can increase weight reuse while still avoiding the intermediate. Test one fused grouped layout against the production-like original chain using identical inputs and alternating warm event timings; skip broad grouping sweeps. Reject if incorrect or not faster than the production-like baseline. This does not rerun earlier unfused grouped MM benchmarks.

### Research checkpoint 2026-11-17T19:30:54+00:00: rejected

- HEAD before checkpoint: 5a59283f98443a08fc7c49b9b1431f050bbb4790
- Budget remaining: unknown
- Finding: Grouped fused SwiGLU is slower than original chain
- Evidence and artifact paths: User granted two additional hours; same host, budget extended to 2026-11-17T21:29:00Z preserving old deadline history. Original v15 production trees; grouped fused 256x128 two-accumulator BF16 kernel with 16 warps, 2 stages vs original MM plus explicit single-launch activation proxy. Alternating 12-event median production shape M2048 K2048 N6144: grouped fused 2.04581 ms, control 1.54150 ms (+32.7% latency); M33 K96 N257 0.01316/0.00986 ms. Outputs exactly match for both sampled BF16 cases; FP16, graph, full stride coverage and production serving not tested. The activation proxy is not the exact compiled production kernel. Reject candidate, no production change.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Investigate a distinct source-owned projection, memory-lifetime or small-M dataflow mechanism; preserve 25-minute reserve and do not attempt unadmitted half-runs.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-17 UTC: grouped MM production kernel attribution (planned)
Prior v15 grouped candidate wide GEMM direct latency 1.39-1.41 to 1.17ms, but serving A/B -7.08% and +1.02% by order; previous worker-attribution attempt stopped before request. Missing evidence: an actual production 4K worker launches a grouped-specialized MM kernel. Diagnostic uses the existing candidate commit 342e8d35 in isolation against original v15, duplicates its kernel with a distinct profiling name only for the existing grouped shape/layout predicate, and captures a short torch profiler window during a 4K diagnostic. Kernel naming and profiler are diagnostic changes, not scoring; no repeated microbenchmarks or serving performance claims. Decision: establish or disprove production execution before considering further grouped work. Remove temporary source changes and profiler after capture.

### Research checkpoint 2026-11-17T19:37:09+00:00: provisional

- HEAD before checkpoint: 1e8cc4a87dc83e3225c8bbb6f2b6bfc0130ccdfb
- Budget remaining: unknown
- Finding: Verify grouped MM GPU execution in 4K worker
- Evidence and artifact paths: Diagnostic original v15 control source plus isolated grouped candidate 342e8d35; duplicate identical grouped kernel renamed mm_kernel_grouped_probe and selected only for M>=1024 N>=8192 stride_bk=1, profiler enabled, no performance claim. A 4K random diagnostic with 8/8 successful requests generated 672 rank0 GPU kernel events matching mm_kernel_grouped_probe, summed duration 784050.465 microseconds in profiler window; this establishes grouped-specialized worker execution in that diagnostic. Async frontend trace has zero GPU events; correct worker rank0 trace SHA256 8f46da15738f8063bc94aef4c12bb00738af80001a3e0e1f28175cacc2ef4a57; 4k client log SHA256 1241260ebaeeac420a9d3e1577f4bc94d3ef4efbbd58514740a3f9aba21e9313. Trace perturbed timings and excludes unseen graph replay; no claim of throughput or accuracy gain. Temporary source restored, owned server stopped.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Analyze new independent GEMM/epilogue mechanism and preserve baseline; prior balanced serving remains inconclusive and half-runs are not admitted in remaining time.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-17 UTC: interleaved single-dot fused SwiGLU (planned)
Previous fused SwiGLU dual-dot variants were exact but slower: row-major 1.895 ms and grouped 2.046 ms vs original MM plus single-launch activation 1.547 ms. New mechanism: load gate/up weight columns interleaved into a single 256-wide tensor-core dot, reshape and split its accumulator into gate/up lanes after BF16 rounding, then fuse activation. This preserves the original 256x256 accumulator footprint while using one dot per K-step instead of two. Compare 256x128, 16-warp implementation against original chain on production and ragged BF16 shapes with alternating warmed timings. Reject on mismatches or no improvement; proxy activation still requires production validation.

### Research checkpoint 2026-11-17T19:39:21+00:00: rejected

- HEAD before checkpoint: f103820d6efd88521527a5993a49a9979163208a
- Budget remaining: unknown
- Finding: Reject interleaved single-dot SwiGLU fusion
- Evidence and artifact paths: Original v15 source; interleaved gate/up columns, single 256-wide dot and split BF16-rounded accumulator vs original MM plus one-launch pointwise proxy. For M2048 K2048 N6144, alternating 12-event median fused 3.01379 ms vs baseline 1.53813 ms (+95.9%); M33 K96 N257 0.01843 vs 0.01199 ms. Sampled BF16 outputs exactly matched, but FP16/strides/graph/production integration were not tested. Reject this kernel; no serving or accuracy result.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Find a genuinely different mechanism; do not retest this interleaved fusion without new causal evidence.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-17 UTC: fused SwiGLU native resource attribution (planned)
Prior dual-dot fused 256x128 latency 1.895 ms vs original MM plus activation proxy 1.547 ms; interleaving and grouped scheduling also regressed. Missing evidence: hardware resource and utilization measurements for the production-sized fused kernel, which may explain the regression and identify whether a distinct dataflow is plausible. Profile one existing fused launch on M2048 K2048 N6144 with native analyzer; compare qualitatively with the archived original MM profile. Diagnostic only: profiler replay perturbs timing, no source or serving modifications. Stop fusion exploration if measurements expose no actionable bottleneck.

## 2026-11-17 UTC: smaller-N dual-dot fusion resource test (planned)
Nearest prior 256x128 dual-dot fused kernel: 1.895 ms versus MM plus pointwise proxy 1.547 ms. Native profile showed 78.31% LSU utilization, 55.77% TCU efficiency, 102 vector registers/thread and 65536 bytes shared/block, vs archived original MM 54.59%, 76.41%, 97 and 65536 respectively (separate profiler captures; counters are diagnostic, not a causality proof). New single variant halves N tile to 64 and warps to 8, reducing per-block accumulators and potentially improving resident blocks; 256 M and 2-stage K32 unchanged. Test exact sampled BF16 outputs and alternating timings vs the production-like original proxy, on ragged and production shapes. Reject if no speedup; no broad sweep.

### Research checkpoint 2026-11-17T19:42:50+00:00: rejected

- HEAD before checkpoint: 369af371168c7acf69b128ba0ae2679c13231639
- Budget remaining: unknown
- Finding: Reject smaller-N fusion and record resource attribution
- Evidence and artifact paths: Original v15 source; native profile fused dual-dot 256x128 LSU 78.31%, TCU 55.77%, 102 vector registers/thread, 65536 shared bytes/block, no local memory; archived original MM 54.59%, 76.41%, 97 and 65536 respectively. Independent profiler captures, no causality established. Profiler forced termination after capture triggered pybind shutdown error; report completed. Smaller-N fused 256x64 eight warps 2 stages exact sampled BF16 on production and ragged shapes; production M2048 K2048 N6144 median 2.59466 ms vs original MM plus activation proxy 1.53635 ms (+68.9%); ragged 0.01366 vs 0.01196 ms. Reject; proxy differs from actual compiled production activation; no production serving claim.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Leave fusion absent a materially distinct mechanism; investigate other priorities within session budget.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-17 UTC: exact division-argmax sampling fusion (planned)
Prior log-domain sampler fusion was faster but failed finite near-tie and nonfinite exactness; prior in-place softmax preserved sampled tokens but failed serving screens. New mechanism keeps original softmax and exponential RNG unchanged and fuses only FP32 division and argmax in a two-stage reduction, avoiding the scores write/read. Must match native torch div rounding, first-index tie, NaN propagation, infinities and zero noise; FP64 Gumbel falls back. Compare precomputed same probs/noise with native div and argmax, edge and random cases, and alternating CUDA-event timing on actual vocabulary size. Reject if any mismatch or no speedup; no claim on startup KV capacity until integration and profiling. This is distinct from log-domain rewrite and does not alter RNG.

## 2026-11-17 UTC: division-argmax adversarial correctness and graph screen (planned)
Prior same-noise FP32 fusion matched sampled NaN, infinity, zero/zero and exact tie cases and random rows; production vocabulary 256x130560 median 0.41023 ms versus native preallocated div plus argmax 1.27503 ms. The reference uses an out buffer instead of native div_ but same arithmetic; source inputs and allocations differ from production, and there is no serving result. Missing evidence is near-tie rounding at scale, broader chunk boundaries, signed zeros/NaNs and CUDA graph replay. Compare native in-place div_ and argmax with identical buffers on large adversarial batches and repeated graph replay; reject on any mismatch, retain FP64 and unsupported layouts fallback.

### Research checkpoint 2026-11-17T19:48:40+00:00: provisional

- HEAD before checkpoint: 300c3fa7df988894c197001e99b198927e42e6f8
- Budget remaining: unknown
- Finding: Promising isolated division-argmax fusion requires serving validation
- Evidence and artifact paths: Same FP32 probs and exponential noise, original softmax and RNG outside prototype, two-stage divide with round-to-nearest and first-index NaN/tie reduction. 256x130560 median preallocated torch div plus argmax 1.27503 ms vs fused 0.41023 ms (16 alternating event samples; approximately 68% lower operation latency). Repeat 1.27517 vs 0.40996 ms. Compared against native in-place div_ plus argmax: zero token mismatches for 65536x16 constructed finite near-tie rows, 2048x2047/2048/2049 and 256x130560 with zeros/zero, NaNs, infinities and ties; three CUDA graph replays each passed. Timing reference uses preallocated out= division rather than in-place div_; no RNG, integration, fallback, FP64, noncontiguous, seeded generator or production serving validation; no end-to-end gain claimed. Temporary workspace is small; no KV-block claim. Code in autoresearch/probes/sampler_divargmax_*.py, raw logs archived separately.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Audit exact division and mutation contract; implement isolated backend integration with fallback only if legal, then test seeded sampling, multiple layouts, graph, and short balanced serving screens. Do not use half-runs without budget admission.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-18 UTC: review correction for division-argmax evidence

The 65536-by-16 adversarial batch contained **65532 constructed finite near-tie rows and four deliberately overwritten edge-case rows** (zero divided by zero, an exact tie, NaNs, and infinities), not 65536 finite near-tie rows. All 65536 rows matched the native result in the recorded run. Three graph replays per shape used unchanged input buffers; they do not establish correctness when graph inputs or request lifecycles change. The second timing pair was a repeated screen in the same test process, not an independent serving replication. Raw archive v15-divargmax.tgz SHA256 f698c51144b7f59f98108fa66c2649a1afee2a348d5027eb06f02ea46595bb8e is backed up outside the PR. The benchmark used preallocated out= division; production uses in-place division. No end-to-end throughput gain, complete exactness proof, FP64 support, or eligibility is established. The self-contained probe import was repaired after the run without rerunning GPU tests; the original executed scripts remain in the archive.

## 2026-11-18 UTC: sampler integration boundary audit (CPU-only)

Read-only audit of the installed vLLM sampler: forward_native applies top-k/top-p, optionally returns processed logits/logprobs, computes softmax, and calls random_sample. random_sample creates exponential noise, performs default-generator draws unless every row has its own generator, then overwrites individually seeded rows; sample_with_exponential_noise mutates probs in place when dtypes match, or mutates q when they differ. A package-text search also found a speculative-decoding proposer calling sample_with_exponential_noise on a cloned tensor and a separate HIP path. Replacing that helper globally with the prototype would change its mutation contract and affect additional call sites; no such integration has been made. A candidate should retain original filtering and RNG call order, limit interception to a proven internal native-sampling path whose probability tensor is dead after selection, and fall back for FP64, unsupported shapes/layouts and any unproven semantics. Verify seeded and mixed generators, processed outputs, aliasing, rounding, nonfinite inputs, graph replay with changing inputs and actual production dispatch before serving screens. This is a source audit, not correctness or performance evidence; installed packages and serving parameters were not modified.

The previous allocation deadline has passed. No new GPU job or benchmark was started on November 18; a confirmed new allocation and budget admission are required before further GPU experiments.

## 2026-11-18 UTC: integrated divide-argmax sampler correctness (planned)
Prior isolated FP32 256x130560 division-argmax prototype measured 1.275/0.410 ms and matched sampled adversarial selection; no integration/RNG/serving validation. New candidate 9b870223 intercepts only native sampler raw modes, retains filtering/softmax and exact default/per-row generator order, uses a new two-stage division/argmax kernel, and falls back to original native function for FP64, processed-output and unsupported input paths. General random-sampling helper and its mutation contract are untouched. Before serving test: compare full seeded native sampler including no/mixed/all row generators, top-k/top-p, nonfinite inputs, layout and FP64 fallbacks, output and next RNG state; graph replay and further rounding/mutation checks follow. Reject on any mismatch; no change to model/server/benchmark flags.

## 2026-11-18 UTC: integrated sampler full-path timing and changing graph inputs (planned)
Prior isolated division-argmax improved latency, and candidate 9b870223 passed nine direct seeded token/RNG cases including filtering, mixed/all per-row RNG, FP64, processed modes, noncontiguous fallback and nonfinite inputs; actual production execution and serving are unknown. New evidence: alternate native full softmax+noise+selection and integrated full path on vocabulary 130560 at batch sizes 8,64,256; include memory allocation and RNG work. Independently replay the two-stage reduction graph with changed finite, tied, and nonfinite inputs and compare native in-place division/argmax each time. Decision: proceed to short serving only if full path improves and graph correctness passes. No user-visible throughput claim from microbenchmarks.

### Research checkpoint 2026-11-18T10:18:41+00:00: provisional

- HEAD before checkpoint: b5e80789bea59511040ab2433ece3cdbbe66e6de
- Budget remaining: unknown
- Finding: Integrated sampler passes initial correctness and full-path GPU screen
- Evidence and artifact paths: Candidate source 9b870223; nine full native sampler cases passed exact tokens, default and per-row generator state for no/top-k/top-p filtering, mixed/all/no row RNG, FP64 fallback, truly noncontiguous input fallback, processed outputs and sampled NaN/inf. Six graph replays with changing input data matched in-place native division and argmax. Alternating 14 CUDA-event samples, vocab 130560, full softmax+noise+selection baseline/candidate medians: batch8 0.41853/0.13910 ms; batch64 1.10934/0.45963 ms; batch256 2.45832/1.59297 ms. Baseline is a hand-constructed native path with no filtering, not a measured production worker or end-to-end result; correctness coverage is limited. Candidate does not alter shared helper mutation behavior, and FP64/processed modes fall back. Raw logs and tests saved; actual worker dispatch, serving, MATH-500 and competition eligibility remain unverified.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Verify actual worker candidate dispatch and run matched short 4K/16K pressure screens before considering an admitted half-run. Retain original source as control.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-18 UTC: candidate worker attribution and short serving screen (planned)
Prior candidate 9b870223 passed direct seeded/graph tests and improved full-path sampling latency on sampled sizes, but actual worker execution and serving impact are unknown. Temporarily log the first call for each shape in the candidate-owned reduction, start the standard server and run 4K then 16K 8-request diagnostic and 4K/16K concurrency-64 pressure screens, with matched control restart and identical seed/flags/cache order. Logging is diagnostic and may perturb timing; any apparent gain requires uninstrumented replication and a fresh matched half-run. Capture actual worker source path/marker, successes, throughput, TTFT/TPOT, preemptions and prefix counts. Remove instrumentation after stopping the owned server; no official benchmark changes.

### Research checkpoint 2026-11-18T10:33:23+00:00: provisional

- HEAD before checkpoint: 710a53911af4827830d830a0ae94c38d7500a9f6
- Budget remaining: unknown
- Finding: Candidate serving diagnostics completed; matched control pending
- Evidence and artifact paths: Candidate source 9b870223 plus temporary first-shape logging in divargmax.sample; worker log observed 1..8x130560 during live 4K requests and 256x130560 during startup. Candidate 4K/16K low and 4K/16K concurrency64 screens completed 8/8,8/8,64/64,64/64. Candidate 4K pressure 2300.91 tok/s TTFT13211.58 TPOT625.04 ms; 16K pressure 3476.71 tok/s TTFT122484.05 TPOT612.82 ms. Logging and warmed cache make these diagnostic; no control, no comparison and no causal gain yet. Live monitor began after 4K pressure started. Candidate server stopped and temporary logging restored.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Run matched original source control with identical order, flags, seeds and metrics; assess screening guardrails before any half-run.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

## 2026-11-18 UTC: clean reverse-order sampler serving repeat (planned)
Prior candidate-first pair used temporary first-shape worker logging and control without logging. All requests succeeded. 4K pressure candidate/control 2300.91/2169.98 total tok/s (+6.03%); TTFT 13211.58/13286.78ms and TPOT 625.04/691.31ms; prefix hits 32640 each, zero new preemptions. 16K candidate/control 3476.71/3382.25 (+2.79%); TTFT 122484.05/129160.66ms, TPOT 612.82/603.53ms; prefix hits 196448/184160, zero new preemptions. Different 16K cache hits and candidate instrumentation confound comparison, one pair only, not a score. Repeat candidate without logging immediately after the clean control, same server flags, workload order, seed and metric snapshots. Decision: require promising 4K and no material 16K regression plus eligibility/correctness/budget before half-runs.

### Research checkpoint 2026-11-18T10:51:05+00:00: provisional

- HEAD before checkpoint: 0c6598327d4620696520ae73b55dd4e1e2101eeb
- Budget remaining: unknown
- Finding: First sampler serving pair promising; clean reverse order in progress
- Evidence and artifact paths: Instrumented candidate-first, original control second, same model/server/benchmark flags, seed and 4K8/16K8/4K64/16K64 sequence. All cases 8/8,8/8,64/64 succeeded. Candidate/control 4K64 2300.91/2169.98 tok/s (+6.03%), TTFT13211.58/13286.78 ms, TPOT625.04/691.31 ms, prefix hits32640/32640, zero new preemptions. 16K64 3476.71/3382.25 (+2.79%), TTFT122484.05/129160.66, TPOT612.82/603.53, prefix hits196448/184160, zero new preemptions. Candidate attribution logger was enabled, control was not; 16K prefix-hit counts differ. Diagnostic only, no half-run or verified improvement; raw archive SHA256 fcf90fbf92bb29c4b304bd95be6b5e84f489d56d419f2fbcda1dbd7e470d02f5 backed up locally. Uninstrumented candidate repeat running after clean control.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Complete clean repeat, compare both workloads and cache state, then check two-run budget and all correctness/rules gates before half-run.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.
