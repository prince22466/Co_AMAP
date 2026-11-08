# v10 Codex autoresearch experiment log

## Frozen external performance reference (NOT a matched A/B)

- Source: `profiling/v9/summary.txt` (user-reported 2-run means).
- 4K `[4096,1024,64,256]`: **2256.31** Total tok/s; +6% goal **2391.6886**.
- 16K `[16384,1024,64,128]`: **1518.06** Total tok/s; +6% goal **1609.1436**.
- Both workloads must complete two successful half-run repetitions. These fixed reported values specify the target, but their raw CSVs/benchmark hash/run machine are NOT present in this tracked summary. Source/benchmark/config compatibility and a fresh v10 matched control must be verified before attributing a change to an optimization.

## Starting code / historical evidence

- v10 is initialized from the v9 **two-stage** eligible BF16 GQA prefill kernel, with the same mixed-paging test. The v10 setup PR only copies these files byte-for-byte and configures research; no new performance improvement is claimed.
- v9 experiment #001 showed ~31% prefill attention call reduction for a stage-3 -> stage-2 change, but achieved occupancy remained near 15.35%; reported bank-conflict change is a clue, not a proven mechanism. An attempted v9 independent confirmation was interrupted; the separately published v9 half-run summary is a numerical reference without original raw CSV artifacts attached there.
- Earlier isolated 4K decode GPU traces attributed ~64.5% kernel time to Linear, but this is NOT a fresh v10 profile or full-serving time share. The v7 grouped GEMM candidate lacked a demonstrated serving win; the v4 grouped page lookup regressed. v9 reported 4K repetitions of **2044.14** and **2468.48 tok/s**, so test cold/warm or prefix-cache effects with independent matched A/B rather than treating a single favorable average as a +6% gain. Treat history as hypothesis guidance, not accepted v10 evidence.

## Required evidence ledger for each experiment

Record date/time, user-provided deadline (or unknown), remaining GPU time, baseline and candidate SHA/source hash, exact operator and production shapes, kernel trace frequency, GPU *summed kernel* versus wall-clock bottleneck share, proposed causal mechanism and Amdahl bound. Validate:

1. v10 GPU operator + attention/decode/mixed paging correctness, edge/tail/fallback checks.
2. Matched warmed-up kernel A/B with medians/variance, registers/SMEM/spills/occupancy and traffic/bank metrics when available.
3. Proof that the changed code path executes in real serving.
4. Matched short 4K and 16K serving diagnostics (not a half-run), successful requests, TTFT/TPOT/output rate and cache variance.
5. Written go/no-go, budget and reason; only then run unchanged BOTH-workload half-run when >=90 min remain.
6. Original raw CSV and summary with exactly 2 SUCCESS rows per workload, complete requests; independent repeat/interleaved control if needed.
7. Two separate decisions: >=5% **progress** versus last accepted matched v10 control; >=6% **final goal** versus fixed v9 reference **plus positive repeated matched-v10 uplift**; crossing the static goal alone is insufficient. Report the other workload and any regressions. Kernel correctness and human-owned MATH-500 remain separate.

All bulk artifacts are stored in `v10/.autoresearch-work/`, never checked in. One half-run may take 1.5–2h. Budget ~300+ minutes if new control plus repeat needed. No benchmark-script edits, hyperparameter tuning, dispatch-only swaps, speculative sampling, quantization or latest-upstream FlagOS merges.

| ID | Date | Accepted control SHA | Hypothesis & target | Kernel/serving A/B | 4K A->B | 16K A->B | Decision & PR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 000 | pending | v10 starting snapshot (hash pending) | Profile 4K decode GEMM/Linear, mixed attention, residual 16K attention; establish fresh matched baseline | Not run | pending | pending | **No v10 performance claims** |

### 001 — Increase BF16 GQA prefill query tile (in progress; not accepted)

- User reported approximately 2.5 hours of GPU availability before this work; exact expiry and remaining time are unverified. Control repository HEAD: `36858e32cf74613508af601ea1a21af062d7ce40`; remote control attention SHA-256 `c651c1f0172feadf1ec1f286fe4466b25c45e25496dc1b1b31237a25946fd55b`. Candidate attention SHA-256 `6e3e8f7cbc24140a9cfd3ef3d2ed8d99cf7ae0f4e0147ac3d9d9b0775925931a`. Unmodified benchmark SHA-256 `873f17f932d4643222be9903ba3b569c1863c4e2d246ceafbe6f75ebe7163ffc`, `RUNS=2`, `SKIP_FIRST=0`.
- Baseline and candidate GPU suites: 18 passed each, including BF16 prefill/decode, mixed paged-KV and window fallback; logs in remote `.autoresearch-work/tests/`. The existing two-stage BF16 GQA specialization was present in the control.
- Short control diagnostics, **not half-runs**: 4K 16/16 success, 2309.59 total tok/s, mean TTFT 6549.03 ms, TPOT 63.03 ms; 16K 8/8 success, 2324.62 total tok/s, mean TTFT 28226.67 ms, TPOT 220.37 ms. Raw logs: remote `.autoresearch-work/quick/control_KpzKNU5i/`.
- Short profiled serving workloads: 4K attention 55.64% and MM 34.93% of **summed recorded GPU kernel duration**; 16K attention 84.00% and MM 11.60%. These are reduced diagnostic workloads with profiler overhead, not wall-time shares or full half-run results; CUDA graph replay limits visibility of decode. Traces: remote `.autoresearch-work/profiles/rank0.1794080972322616724.pt.trace.json.gz` and `rank0.1794081031125988567.pt.trace.json.gz`.
- Hypothesis: increasing the specialized query tile from 32 to 64 rows reuses each loaded paged KV tile over twice as many query rows. The 84% GPU-sum share provides an upper bound only for that short 16K trace, not an end-to-end prediction.
- Isolated event-timed alternating A/B on identical BF16 inputs, shuffled page table, 12 samples per variant, reference checked at `atol=rtol=0.03`: 2048 query tokens and context 0: 2.1521 -> 1.2968 ms; context 4096: 10.1361 -> 5.7951 ms; context 14336: 30.4801 -> 17.4209 ms. A 64-token query was 0.0203 -> 0.0204 ms. GPU was otherwise idle. Probe and full ranges: remote `.autoresearch-work/probes/attention_block64.log`. This does not prove production invocation or a serving gain.
- First candidate short serving diagnostics (same inputs and server flags following restart), **not half-runs**: 4K 16/16 success, 2214.43 total tok/s, mean TTFT 7510.18 ms, TPOT 60.41 ms; 16K 8/8 success, 3206.92 total tok/s, mean TTFT 19019.92 ms, TPOT 169.75 ms. Compared with the single control run these are -4.12% and +37.96%, respectively; noise, ordering, and the potential 4K regression require repeated matched runs. Raw logs: remote `.autoresearch-work/quick/candidate_1hdwSjle/`.
- Second candidate run after another restart: 4K 16/16 success, 2477.80 total tok/s, mean TTFT 6032.96 ms, TPOT 59.26 ms; 16K 8/8 success, 3208.37 total tok/s, mean TTFT 19029.22 ms, TPOT 169.54 ms. Logs: remote `.autoresearch-work/quick/candidate_repeat_1pLjTBm8/`. These are screening runs, not competition results.
- Candidate production 16K trace confirms the attention kernel with the wider query tile executes: dominant grids `[257..260, 2, 1]` versus control `[513..516, 2, 1]`, with 114 registers per thread in both, shared memory 24,576 -> 32,768 bytes and profiler-estimated occupancy 16% -> 13%. The traces contain different numbers of launches, so aggregate elapsed GPU durations are not directly paired. Spills and bandwidth remain unmeasured. Candidate trace: remote `.autoresearch-work/profiles/rank0.1794082190925456122.pt.trace.json.gz`.
- Reverse-order control repeat after server restart: 4K 16/16 success, 2304.36 total tok/s, TTFT 6549.60 ms, TPOT 63.29 ms; 16K 8/8 success, 2334.32 total tok/s, TTFT 28190.54 ms, TPOT 218.79 ms. Logs: remote `.autoresearch-work/quick/control_repeat_DLOdgZQs/`. The two 16K control/candidate screening runs are consistent with a substantial benefit; the 4K candidate varies and a material full-workload regression is not ruled out.
- Go decision: production trace, reference correctness, paired isolated timings and repeated short serving diagnostics justify collecting the starting control. User reported approximately 1h40m remaining shortly before 20:05:37 UTC; the unchanged control half-run actually started at 20:15:41 UTC, roughly at the 90-minute floor given the approximate allocation. A complete 90–120-minute run and a matched candidate repeat are not guaranteed before expiry. Run and raw outputs: remote `.autoresearch-work/bench_runs/control_4IyqUx3J/`. Both complete matched half-runs and spill/bandwidth analysis remain **pending**. No promotion, PR, final score, or MATH-500 claim.
- Control 4K repetitions both succeeded with all 256 requests, at 1852.74 and 2411.83 total tok/s (mean 2132.285). The large spread requires investigation and repeated interleaved A/B.
- Completed control half-run: all four raw rows SUCCESS with 256/256 requests for each 4K run and 128/128 for each 16K run. 16K repetitions 1523.20 and 1537.20, mean **1530.20** total tok/s; 4K reported summary mean **2132.28**. The self-comparison through `autoresearch/evaluate.py` validates CSV structure and completion only, not a speedup. Both CSVs are preserved locally in `.autoresearch-work/bench_control/` and remotely in the run directory above. A candidate half-run remains necessary for any threshold claim.
- Expanded candidate GPU suite, including 7-, 9- and 65-token query/page-boundary cases: **21 passed**; log `.autoresearch-work/tests/candidate_block64_boundaries.log` on the remote.
- Higher-concurrency 4K diagnostic (4096 input, 128 output, concurrency 64, 64 prompts, separate matched server restarts): control 1906.50 total tok/s, mean TTFT 30059.43 ms and TPOT 700.67 ms; candidate 2026.42 total tok/s, mean TTFT 25559.75 ms and TPOT 670.15 ms; both 64/64 successful. This single short diagnostic is not a half-run or proof of a repeatable 4K gain. Logs: remote `.autoresearch-work/quick/highconc_control_PCvz5kbr/` and `highconc_candidate_ytn2Yps6/`.
- Draft PR requested by the user: https://github.com/prince22466/Co_AMAP/pull/229. The candidate has not met the matched half-run promotion gate; keep the PR in draft pending that evidence.

### 001 continued — 2026-11-08 validation

- Review fix: pad the single-sequence GPU test's cache and page table with one spare 16-token page, matching the existing decode/mixed tests. The pinned reference performs an unmasked page-index load across each 32-token KV tile; the 98-token tail case previously allocated seven entries while the final tile could access entry seven. Inference code is unchanged by this test repair.
- Corrected candidate suite: **21 passed**. Saved control with the same corrected inputs and original 32-row expectation: **21 passed**. Remote logs: `.autoresearch-work/validation_20261108/candidate_correctness.log` and `control_correctness.log`.
- User reported 5.8 hours available around 16:03 UTC on 2026-11-08 (approximate expiry 21:51 UTC). Plan: fresh control A1, candidate B1, then independent A2/B2 repeats if time permits; never begin a half-run below the 90-minute start floor. The previous allocation's CSVs remain historical evidence, not a current paired control.
- Remote machine remains `84d64a9dab4f`, with matching control/candidate attention and unchanged benchmark SHA-256 values recorded above. Model files and package versions are captured in remote `.autoresearch-work/validation_20261108/provenance.json`. First fresh control run: `.autoresearch-work/validation_20261108/A1_GXj1WMMX/`. Results pending.

## Results template

### NNN — title
- Deadline/allocation and starting/remaining time:
- Source / benchmark hashes, source import/invocation:
- Actual serving breakdown: 4K vs 16K, prefill / mixed / decode, GPU kernel sum vs wall:
- Hypothesis, estimated attributable Amdahl maximum, expected end-to-end effect:
- Tests (including numerical tolerances and fallbacks):
- Repeated identical-input kernel A/B; resources, emitted compiler evidence:
- Production invocation trace:
- Short diagnostic serving A/B 4K and 16K (NOT scored):
- Go/no-go and budget:
- Unchanged half-run paired CSVs (both two-run means, ranges, TTFT/TPOT, completion):
- Comparison to last accepted v10 control (+5% PR gate):
- Comparison to **fixed** reported v9 reference (+6% final goal):
- Independent repeatability, matched uplift, other-case regressions and competition-score tradeoff:
- Decision / relevant commit / PR / outstanding MATH-500:
