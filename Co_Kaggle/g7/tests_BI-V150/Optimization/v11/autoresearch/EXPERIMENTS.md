# v11 Codex Terra 6.1 — autoresearch experiment ledger

## Frozen reference and goals

The v10 source of the reported two-workload development half-run is `profiling/v10/summary.txt`; original raw CSVs and equivalent current-machine baseline provenance are not tracked with that summary. **Do not call these numbers a matched control.**

| Workload | Input/decode/concurrency/prompts | Reported v10 Total tok/s | Min +5% | Stretch +10% |
| --- | --- | ---: | ---: | ---: |
| 4K | `[4096,1024,64,256]` | 2372.38 | 2490.999 | 2609.618 |
| 16K | `[16384,1024,64,128]` | 1872.70 | 1966.335 | 2059.970 |

- Require fixed goal **AND** independently verified >=5% or >=10% uplift over a fresh *v11 starting-code* matched A/B control in one workload. Other workload must be reported without an unapproved material regression. Confirm every half-run has two successful 4K repetitions of 256/256 requests and two successful 16K repetitions of 128/128 requests; `RUNS=2`, `SKIP_FIRST=0`.
- At >=3% reproducible gain vs latest accepted v11 control: draft progress PR. At >=5% verified fixed+matched minimum: milestone PR and keep researching toward 10% if resources allow. At >=10% verified fixed+matched stretch: final PR and stop. No auto-merge; human owns MATH-500.
- v10 performance noise: control A1 4K individual repetitions 2103.78 / 2474.28; candidate B1 2078.02 / 2677.88. Report run-level spread and independently paired confirmation; a single high 4K mean is insufficient.
- v10 short profiled GPU-sum shares (NOT wall time, NOT v11 evidence): 4K attention 55.64%, MM 34.93%; 16K attention 84%, MM 11.60%. Reprofile. v10 widened BF16 GQA prefill query tile `BLOCK_M=32 -> 64`, reducing one long prefill kernel 30.4801 -> 17.4209 ms, although SMEM 24KB -> 32KB and estimated occupancy 16% -> 13%; these are historical clues only.

## Start-of-run provenance checklist

- Authorized GPU owner, wall deadline/allocation (or unknown), observed clock/start time, remaining time. Half-run start floor >=90min; may need 120min, and repeated A/B may require 300+min.
- Git branch, HEAD, exact v11 operator SHA256 and tracked/untracked status; compare known v10 starting-source trees without *executing* v10. Record source and checkpoint hashes after every acceptance.
- BI-V150 hardware/drivers, Torch/Corex/Triton/vLLM/FlagGems package versions, model and weights hashes, loaded `inspect.getfile` locations, launch and serving flags, optimizer path invocation trace.
- Exact unmodified external benchmark SHA, benchmark CLI inputs, fresh named raw CSV and summary, successful requests/error counts, cache capacity/prefix hits, GPU telemetry.
- All bulk raw evidence retained in ignored `/root/Optimization/v11/.autoresearch-work/`; preserve each complete A/B pair, including failures and negative experiments.

## Research queue (re-rank after v11 production traces)

1. Mixed prefill/decode page-index/KV address reuse inside current Triton unified attention; preserve 2D/3D/fallback and causal semantics.
2. Real 4K decode small-M BF16 GEMM/Linear matrix tiling, weight memory coalescing and accumulator dependencies; no dispatch-only swap, no naive v7 grouped-MM copy.
3. Residual 16K BF16 prefill memory pipeline / KV tile and score accumulation reuse with current v10 `BLOCK_M=64` as control; measure SMEM/register/spill effects first.
4. KV/cache wait, idle and launch gaps only when serving wall evidence and allowed kernel/data-path change support it.
5. Minor ops only when current Amdahl bound supports >noise floor.

## Experiment evidence format

| ID | UTC | Baseline SHA | Hypothesis/operator | Kernel A/B + correctness | Short serving A/B | Full 4K A->B | Full 16K A->B | Decision/PR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 000 | pending | v11 starting snapshot (hash pending) | Fresh production profile, immutable matched control | Not run | Not run | Pending | Pending | **No v11 performance claim yet** |

### NNN — Title

- Authorized run time; current GPU deadline/remaining time:
- Previous accepted v11 and frozen original starting-source hashes:
- Real production shapes / attention launch phases / CPU-GPU idle and KV waits:
- GPU summed-kernel share vs end-to-end critical path; estimated Amdahl bound:
- Kernel/data-movement hypothesis; reason it is allowed by competition rules:
- Correctness tests (tails, mixed, BF16, fallbacks) and numerical tolerances:
- Interleaved GPU kernel A/B medians/spread, registers/SMEM/spills/bandwidth:
- Proof new kernel executes in real v11 serving:
- Short matched 4K and 16K serving A/B; cache hits, TTFT/TPOT/output tok/s:
- Half-run go/no-go, deadline budget, provenance and raw filenames:
- Unmodified two-case half-run A1/B1 + independent A2/B2; full request counts:
- Frozen reported v10 +5%/+10% comparison **separate** from >=5%/+10% matched-v11-starting-code comparison:
- Incremental >=3% checkpoint comparison vs latest accepted best:
- Other workload regression/competition impact and remaining MATH-500:
- Accepted/rejected, rollback proof, commit SHA, draft PR URL and next direction:

Never erase a negative result or label interrupted half-runs as complete. `autoresearch/evaluate.py` exit 0 means valid CSVs only; numeric threshold fields are **not** acceptance certificates.
