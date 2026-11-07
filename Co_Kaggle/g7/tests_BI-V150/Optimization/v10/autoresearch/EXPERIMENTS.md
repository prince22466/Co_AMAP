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
