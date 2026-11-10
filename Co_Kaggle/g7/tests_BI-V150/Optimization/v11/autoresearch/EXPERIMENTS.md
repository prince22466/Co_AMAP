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

### Research checkpoint 2026-11-10T04:40:50+00:00: provisional

- HEAD before checkpoint: 52ba452f840c699654a28b8594681a5bc3c47d07
- Budget remaining: unknown
- Finding: Starting control 4K repetitions both SUCCESS 256/256: 2211.04 then 2526.52 total tok/s; 16K pending
- Evidence and artifact paths: Remote guard_20261110T042048_3897 raw_runs_20261110_122048.csv; durations 592.81 and 518.78 seconds; server cumulative prefix cache hit 0.0 to 0.2 percent; incomplete half-run
- PR-preserved small benchmark CSVs: autoresearch\evidence\.autoresearch-work_7027890d743c_raw_runs_20261110_122048.csv (sha256 7027890d743cd9df4402a35c5cfa82db3d7095d2a57095cbb56a248d9b4550af)
- Next action: Monitor both 16K repetitions and back up full raw, summary, telemetry and logs
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T04:55:05+00:00: provisional

- HEAD before checkpoint: 5f6d931b712ebea239d1f35778944b0cb9dbe056
- Budget remaining: unknown
- Finding: First 16K control repetition in progress; server metrics show 3 cumulative preemptions and KV cache near capacity
- Evidence and artifact paths: Prometheus vllm:num_preemptions_total=3 at 2026-11-10 approximately 04:54 UTC; server log .autoresearch-work/serve_runs/v11-control.log; both 4K rows preserved; causal wall-time attribution not established
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Finish and validate the control, preserve full logs, then profile production phases before selecting a hypothesis
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T05:01:58+00:00: provisional

- HEAD before checkpoint: 5007973b49eb3d9f4a49cde6ff8993c924692d31
- Budget remaining: unknown
- Finding: Control 16K repetition 1 SUCCESS 128/128, total 1824.16 tok/s; final 16K repetition pending
- Evidence and artifact paths: Raw CSV contains 4K 2211.04 and 2526.52, 16K first 1824.16; duration 1221.51 s; 3 preemptions observed earlier; no matched candidate or speedup claim
- PR-preserved small benchmark CSVs: autoresearch\evidence\.autoresearch-work_87b0bdb9c657_raw_runs_20261110_122048.csv (sha256 87b0bdb9c6570fbcaa628129e0a58f78caee96576bfcf31ddc6b0884288db208)
- Next action: Finish 16K repetition 2 and back up full evidence; profile production behavior
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T05:09:46+00:00: provisional

- HEAD before checkpoint: 8fa3dbc01137f3c3eece5d90911f4b0a75756d7b
- Budget remaining: unknown
- Finding: Final 16K control repetition running; cumulative server preemptions increased from 3 to 5
- Evidence and artifact paths: Server vllm:num_preemptions_total=5 sampled near 05:09 UTC; cache near capacity; three successful raw rows already preserved; no wall-time attribution
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Complete control and back up all evidence before production profiling
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T05:22:51+00:00: provisional

- HEAD before checkpoint: f53d6d915ae4c3f909f110fa1d6a04a66c59120f
- Budget remaining: unknown
- Finding: Fresh starting-code control complete: 4K mean 2368.78 tok/s, repeats 2211.04 and 2526.52; 16K mean 1859.34, repeats 1824.16 and 1894.52; all requests successful
- Evidence and artifact paths: run guard_20261110T042048_3897 elapsed 60.71 min; four SUCCESS rows validated by evaluate.py; off-instance local backup .autoresearch-work/remote-handoff/guard_20261110T042048_3897 plus server and GPU telemetry logs; source and model hashes in preceding checkpoints; no speedup or accuracy claim
- PR-preserved small benchmark CSVs: autoresearch\evidence\guard_20261110T042048_3897_4d33913621d4_raw_runs_20261110_122048.csv (sha256 4d33913621d4d7db32b7c00be5afffa169a8fc7d8f9b965e85b0dc38945aa3c0); autoresearch\evidence\guard_20261110T042048_3897_83f6edc1c9ec_summary_20261110_122048.csv (sha256 83f6edc1c9ec39dcd0e6b3c74e9f8816d908bd86fd580f2238c0b9075211f064)
- Next action: Profile production phases and screen one operator hypothesis; admit candidate half-run only if justified and time permits
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T05:28:44+00:00: provisional

- HEAD before checkpoint: ef21fd93d45cbe18c2dcab9b8493877693cb60fc
- Budget remaining: unknown
- Finding: Initial 4K short profile reports sampled summed GPU time mm 46.43 percent and unified attention 45.33 percent; diagnostic capture is not serving wall time
- Evidence and artifact paths: Remote .autoresearch-work/profiles/profiler_out_0.txt and rank0.1794288498430610818.pt.trace.json.gz; first 20 profiled engine iterations with warmup and graph visibility limitations; short probe in progress
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Back up traces, profile 16K and mixed phases, then rank kernel hypotheses with an Amdahl bound
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T05:32:08+00:00: provisional

- HEAD before checkpoint: e1a7b59a3e43ff90eb227f29fa8155960d14e2e9
- Budget remaining: unknown
- Finding: Short prefill profiles: 4K sampled GPU-sum mm 46.43 percent, attention 45.33 percent; 16K attention 79.25 percent, mm 17.81 percent
- Evidence and artifact paths: Off-instance .autoresearch-work/remote-handoff/profiles holds both compressed torch traces and summaries; 2048x2048 BF16 projection shapes and attention Q shape 2048x16x128 observed; 16K diagnostic cache hit rate rose to 16.7 percent; profiles include initial warmup and graph replay limits, not wall-time shares
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Capture mixed/decode phases, assess Amdahl bound and select one real operator change
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.
