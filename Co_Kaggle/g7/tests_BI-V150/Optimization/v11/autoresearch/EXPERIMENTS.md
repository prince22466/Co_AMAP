# v11 Codex Terra 6.1 — autoresearch experiment ledger

## Frozen reference and goals

The v10 source of the reported two-workload development half-run is `profiling/v10/summary.txt`; original raw CSVs and equivalent current-machine baseline provenance are not tracked with that summary. **Do not call these numbers a matched control.**

| Workload | Input/decode/concurrency/prompts | Reported v10 Total tok/s | Min +5% | Stretch +10% |
| --- | --- | ---: | ---: | ---: |
| 4K | `[4096,1024,64,256]` | 2372.38 | 2490.999 | 2609.618 |
| 16K | `[16384,1024,64,128]` | 1872.70 | 1966.335 | 2059.970 |

- Require fixed goal **AND** independently verified >=5% or >=10% uplift over a fresh *v11 starting-code* matched A/B control in one workload. Other workload must be reported without an unapproved material regression. Confirm every half-run has two successful 4K repetitions of 256/256 requests and two successful 16K repetitions of 128/128 requests; `RUNS=2`, `SKIP_FIRST=0`.
- At >=3% reproducible gain vs latest accepted v11 control: draft progress PR. At >=5% verified fixed+matched minimum: milestone PR and keep researching toward 10% if resources allow. At >=10% verified fixed+matched stretch: final PR and stop. No auto-merge; the human runs the unchanged MATH-500 accuracy evaluation and must verify the project's >=95% acceptance threshold before submission. v10's reported 97.1% is not proof that v11 passes.
- v10 performance noise: control A1 4K individual repetitions 2103.78 / 2474.28; candidate B1 2078.02 / 2677.88. Compare first-vs-first and second-vs-second 4K repetitions under matched cache/warmup/order conditions; report run-level spread and independent A/B confirmation. A single high 4K second repetition or mean is insufficient.
- v10 short profiled GPU-sum shares (NOT wall time, NOT v11 evidence): 4K attention 55.64%, MM 34.93%; 16K attention 84%, MM 11.60%. Reprofile. v10 widened BF16 GQA prefill query tile `BLOCK_M=32 -> 64`, reducing one long prefill kernel 30.4801 -> 17.4209 ms, although SMEM 24KB -> 32KB and estimated occupancy 16% -> 13%; these are historical clues only.

## Start-of-run provenance checklist

- Authorized GPU owner, wall deadline/allocation (or unknown), observed clock/start time, remaining time. Time gate is enforced by `autoresearch/session_budget.py`: initially 105min measured +20min per-run margin +25min final preservation reserve (>=150min for one, >=275min for two); update estimate from actual runs. Four complete A1/B1/A2/B2 half-runs require >=525min under the initial budget model and cannot fit a 330min session. A single A/B pair is provisional.
- Git branch, HEAD, exact v11 operator SHA256 and tracked/untracked status; compare known v10 starting-source trees without *executing* v10. Record source and checkpoint hashes after every acceptance.
- BI-V150 hardware/drivers, Torch/Corex/Triton/vLLM/FlagGems package versions, model and weights hashes, loaded `inspect.getfile` locations, launch and serving flags, optimizer path invocation trace.
- Exact unmodified external benchmark SHA, benchmark CLI inputs, fresh named raw CSV and summary, successful requests/error counts, cache capacity/prefix hits, GPU telemetry.
- All bulk raw evidence retained in ignored `/root/Optimization/v11/.autoresearch-work/` and backed up to persistent off-instance storage before GPU expiry; Git-only notes must include hashes/paths plus a candid availability status. Capture failures and negative experiments too. Use `autoresearch/checkpoint.py --push` periodically (<=30min), before expensive runs and at time/credit pressure.

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

Never erase a negative result or label interrupted half-runs as complete. Checkpoints in this file are append-only; PR updates are not evidence of benchmark success. `autoresearch/evaluate.py` exit 0 means valid CSVs only; numeric threshold fields are **not** acceptance certificates.

### Research checkpoint 2026-11-10T04:07:07+00:00: blocked

- HEAD before checkpoint: not available
- Budget remaining: unknown
- Finding: initial_remote_access_checkout_missing_git
- Evidence and artifact paths: matching_three_operator_hashes_external_benchmark_sha_873f17f9
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: establish_git_PR_and_confirm_allocation_deadline
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T04:07:57+00:00: blocked

- HEAD before checkpoint: not available
- Budget remaining: unknown
- Finding: checkpoint_CSV_shadowing_bug_fixed_tests_passed
- Evidence and artifact paths: 33_harness_tests_pass_git_checkout_missing_benchmark_hash_873f17f9
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: establish_PR_branch_confirm_deadline_and_backup
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T04:10:36+00:00: provisional

- HEAD before checkpoint: not available
- Budget remaining: unknown
- Finding: initial_GPU_correctness_21_tests_passed
- Evidence and artifact paths: .autoresearch-work/initial_correctness.log_and_33_harness_tests_passed
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: restore_Git_push_access_and_PR_checkout_before_benchmark
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T04:17:30+00:00: provisional

- HEAD before checkpoint: 3b930fa7a8afa7ae2858497390d46beb6df697a7
- Budget remaining: unknown
- Finding: Remote v11 inference sources match 2812 local working tree files; BI-V150 imports v11 modules and port 9031 is free
- Evidence and artifact paths: Remote .autoresearch-work/source-bytes-check.log and probe-v11.log; prior 21 GPU and 33 harness tests passed; Git commit 3b930fa7; benchmark sha256 873f17f932d4643222be9903ba3b569c1863c4e2d246ceafbe6f75ebe7163ffc
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Start owned control server, capture model provenance, and prepare baseline guarded half-run using local Git checkpoint preservation
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T04:19:54+00:00: provisional

- HEAD before checkpoint: 707be5f26575fe4fe5c251d326a4b9c9a3c433f4
- Budget remaining: unknown
- Finding: Preparing first unchanged v11 control half-run; no performance result yet
- Evidence and artifact paths: Source working-tree comparison 2812 files matched; BI-V150; model weights sha256 14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c; benchmark sha256 873f17f932d4643222be9903ba3b569c1863c4e2d246ceafbe6f75ebe7163ffc; 21 GPU tests pass; allocation ends 2026-11-10T09:48:00Z; local backup .autoresearch-work/remote-handoff
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: After server health and deadline admission, run guarded half-run and sync all checkpoints and evidence via local Git as operator requested
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T04:21:10+00:00: provisional

- HEAD before checkpoint: 07a2276a254879b42bad2f08c0465d54ed099ed8
- Budget remaining: unknown
- Finding: Unchanged v11 control half-run started at 2026-11-10T04:20:48Z; result pending
- Evidence and artifact paths: Remote .autoresearch-work/bench_runs/guard_20261110T042048_3897/run.json and benchmark.log; GPU telemetry .autoresearch-work/gpu-telemetry-control.log; fixed two-workload benchmark and source/model hashes recorded in prior checkpoint
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Monitor run; back up raw CSVs and logs, validate request counts, and publish results via local Git
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T04:32:51+00:00: provisional

- HEAD before checkpoint: d64d5801045030d2d27fde8e4b31e422d824d54e
- Budget remaining: unknown
- Finding: Starting control 4K repetition 1 SUCCESS 256/256, total 2211.04 tok/s; remaining repetitions still running
- Evidence and artifact paths: Remote guard_20261110T042048_3897/benchmark_results/raw_runs_20261110_122048.csv; first 4K duration 592.81 seconds; prefix cache reported 0%; raw partial CSV backed up locally; not a completed half-run
- PR-preserved small benchmark CSVs: autoresearch\evidence\.autoresearch-work_9c0166097d0d_raw_runs_20261110_122048.csv (sha256 9c0166097d0def13caafa506d81328b21b8ee1c370b3806a7ba3fc736222d6b0)
- Next action: Continue control, preserve all four raw rows and both workload summaries
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.
