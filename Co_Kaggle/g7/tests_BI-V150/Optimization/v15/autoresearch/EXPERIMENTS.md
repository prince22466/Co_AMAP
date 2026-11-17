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
