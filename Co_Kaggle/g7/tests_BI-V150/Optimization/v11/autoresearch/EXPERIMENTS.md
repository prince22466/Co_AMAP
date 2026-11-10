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

### Research checkpoint 2026-11-10T09:01:17+00:00: provisional

- HEAD before checkpoint: 9f42717e8ea13859d3afc0c4cf2c269f1e8910f7
- Budget remaining: unknown
- Finding: User reports 50 minutes remaining; retained earlier guarded expiry 09:48 UTC, now approximately 47 minutes remain; no further half-run admitted
- Evidence and artifact paths: Control completed and CSVs pushed; 4K and 16K initial profiles backed up; mixed 4K diagnostic completed remotely at .autoresearch-work/profiles/quick-mixed-4k.log and rank0.1794288777900331932.pt.trace.json.gz; candidate not yet tested
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Immediately back up mixed profile and all diagnostic logs, analyze mixed attention and run a brief kernel screen if time remains
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:03:02+00:00: provisional

- HEAD before checkpoint: e598461cbcc8d15b6d3bf8b458c0405d49aefadd
- Budget remaining: unknown
- Finding: Deadline threshold reached: mixed 4K trace captured graph replay with no aten::mm or unified attention CPU events; phase attribution remains unresolved
- Evidence and artifact paths: Backed up local .autoresearch-work/remote-handoff/profiles/rank0.1794288777900331932.pt.trace.json.gz and all diagnostic logs; full control evidence backed up; 4K and 16K initial prefill profiles do not establish wall-time gain
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Timebox a single paged-KV kernel screen and preserve reserve; no full benchmark admitted
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:03:32+00:00: hypothesis

- HEAD before checkpoint: 9acbbd03a9fc9469adaee1c8dbf4e7b7feee025e
- Budget remaining: unknown
- Finding: Screen hypothesis: deduplicate KV page-table loads within existing BF16 GQA prefill attention; 16K short profile sampled 79.25 percent unified attention GPU-sum
- Evidence and artifact paths: Hypothesis only, no causal wall share or mixed-phase attribution; profile trace hidden by graph replay; fixed benchmark control 16K mean 1859.34; baseline source tree and diagnostics retained
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Use one guarded operator change, run numerical tests and paired kernel A/B; revert if it fails or remains unverified
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:05:59+00:00: provisional

- HEAD before checkpoint: d8a0e99de350a34faff114a4e7a890a4efa10cea
- Budget remaining: unknown
- Finding: Experimental prefill page-table reuse passed 21 attention/decode/mixed GPU tests; alternating same-input kernel A/B is in progress
- Evidence and artifact paths: Remote .autoresearch-work/candidate-tests.log; candidate preserves decode lookup and BLOCK_M=64, checks BF16 GQA tails and mixed paging; correctness alone is not performance evidence
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Inspect interleaved GPU timings; revert candidate if weak or unverified; no half-run under deadline
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:07:45+00:00: rejected

- HEAD before checkpoint: fe8c3f5ac31d387417ce4721d67c5555aaccc842
- Budget remaining: unknown
- Finding: REJECTED: BF16 GQA prefill page-index reuse passed 21 tests but regressed paired kernel medians by 29.99 percent at context 4096 and 29.88 percent at context 14336; reverted
- Evidence and artifact paths: Identical BF16 query=2048 and randomized 16-token pages; alternating 10 samples: 5.7616 to 7.4898 ms and 17.4620 to 22.6797 ms; max output difference 0; remote .autoresearch-work/kernel-screen.log and candidate-tests.log; original source SHA256 6e3e8f7cbc24140a9cfd3ef3d2ed8d99cf7ae0f4e0147ac3d9d9b0775925931a restored; no serving test
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Back up negative raw logs; inspect another safe hypothesis only if time permits; preserve 25-minute handoff reserve
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:09:39+00:00: hypothesis

- HEAD before checkpoint: 9bc87e97de15053bf1aa02486eba99234d58b095
- Budget remaining: unknown
- Finding: Second bounded hypothesis: explicitly pipeline the 16K prefill KV tile loop with two in-flight stages while preserving non-prefill loop defaults
- Evidence and artifact paths: 16K initial prefill trace: attention 840 launches and 9.995 s sampled GPU sum, median 11.337 ms range 5.822 to 19.975 ms; this is not wall time; first page-gather candidate rejected
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Run GPU correctness and alternating same-input kernel screen; reject if slow or unverified before 25-minute reserve
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:12:34+00:00: rejected

- HEAD before checkpoint: 2572fd9949695eada86badefbcfc10e20d4e0d00
- Budget remaining: unknown
- Finding: REJECTED: explicit two-stage prefill loop passed 21 GPU tests but did not improve paired kernel medians
- Evidence and artifact paths: Alternating 10 samples, BF16 query 2048 with randomized 16-token pages: context 4096 control 5.7891 ms candidate 5.7782 ms; context 14336 control 17.4541 ms candidate 17.4999 ms; outputs identical; remote .autoresearch-work/pipeline-tests.log and pipeline-screen.log; control SHA256 6e3e8f7cbc24140a9cfd3ef3d2ed8d99cf7ae0f4e0147ac3d9d9b0775925931a restored
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Back up both screen logs and traces, verify PR and stop at preservation reserve; no candidate promoted
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:14:36+00:00: provisional

- HEAD before checkpoint: e3330c0ddfe1117259da795509771cedcaad10cc
- Budget remaining: unknown
- Finding: Final restoration verified: 2812 tracked inference working-tree files match remote bytes; both experimental operators rejected and no v11 speedup verified
- Evidence and artifact paths: 4K control 2211.04/2526.52 and 16K 1824.16/1894.52 total tok/s; all 4 rows SUCCESS; two candidate tests 21/21 and alternating A/B logs backed up in local .autoresearch-work/remote-handoff; profiler logs and compressed traces backed up; checkpoint evidence committed to draft PR; no owned tmux sessions remain
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Next allocation: capture phase-resolved serving traces including graph replay and real wall shares; investigate 16K KV pipeline or real small-M GEMM only with evidence; reproduce baseline and matched independent A/B; human MATH-500 >=95 percent for chosen candidate
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:15:00+00:00: provisional

- HEAD before checkpoint: 206bfb25f117da8519b2953f3fda9650dc3f8015
- Budget remaining: unknown
- Finding: Baseline server-log sampling shows sustained 16K KV pressure and queues; this does not identify a permissible or causal optimization by itself
- Evidence and artifact paths: Approximate 16K time window: 238 active 10-second server samples, 201 with KV usage >=90 percent and 225 with requests waiting; 4K window: 110 active samples, zero with KV usage >=90 percent and 14 with requests waiting. Local .autoresearch-work/analyze-server-log.py and remote-handoff/v11-control.log. Windows are approximate and neither GPU kernel shares nor attribution of wall delay
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Next session measure phase-resolved critical path and preemption cost; investigate only allowed equivalent-semantics operator/data-path changes, without altering cache capacity or scheduler settings
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:15:28+00:00: rejected

- HEAD before checkpoint: 2be7351f32fba02e816092b1c73674b3d6b91a59
- Budget remaining: unknown
- Finding: Reproduction material preserved for both rejected screens; no candidate promoted
- Evidence and artifact paths: Off-instance local .autoresearch-work/kernel-screen.py contains alternating test inputs and timings; recreate-rejected.py and recreate-rejected-pipeline.py reconstruct the experimental changes from the verified original source; regenerated candidate files are approximations, not verified original byte snapshots; full raw logs and traces are in .autoresearch-work/remote-handoff
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Use a new confirmed allocation for graph-aware phase profiling, matched control/candidate serving and independent replication; preserve existing dispatch and model settings
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:16:25+00:00: provisional

- HEAD before checkpoint: cdb6b66642d97c287ba727a8585f8c1b90cd9ffc
- Budget remaining: unknown
- Finding: Off-instance backup checksum verification complete; no tested speedup and no owned GPU processes remain
- Evidence and artifact paths: Local backups match remote SHA256: 4K trace 4f4ded08, 16K trace e1b20abb, mixed trace ffa3029f, control server log 4a6ce4ba, GPU telemetry fcbe45d0, negative kernel logs 0a4f1ce2 and 683dc067; full control raw and summary CSVs committed to PR; restored source 2812 files matched
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Preserve 25-minute reserve; next allocation needs graph-aware mixed/decode traces, a justified operator candidate, matched A/B and independent replication; human MATH-500 remains pending
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:33:07+00:00: provisional

- HEAD before checkpoint: 94ae4e60137be29681ea8f5c48d5a13609343100
- Budget remaining: unknown
- Finding: Operator requests 20-minute follow-up to identify other optimization directions; earlier guarded deadline remains 09:48 UTC, approximately 15 minutes remaining
- Evidence and artifact paths: Two prior prefill hypotheses rejected; both traces and full control saved; no owned server running; half-run admission denied
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Inspect real projection paths and baseline trace shapes, screen a narrowly scoped kernel hypothesis only if time allows, preserve before expiry
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:33:43+00:00: hypothesis

- HEAD before checkpoint: b4afd39b41faa9de2962d244b4510e75aafdbac6
- Budget remaining: unknown
- Finding: Follow-up 4K trace: BF16 MLP-sized projection (2048x2048) x (2048x12288) contributes 942.19 ms of 1806.45 ms sampled mm kernel time
- Evidence and artifact paths: Trace CPU-to-GPU external-id attribution: 672 such calls, median 1402.18 us; second (2048x6144)x(6144x2048) 487.28 ms of sampled mm sum; first 20 profiled iterations, CUDA graphs and serving wall share unresolved; source .autoresearch-work/trace-associate.py
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Screen a shape-specific tile-order data-reuse idea against identical BF16 inputs; require correctness and reject regressions; no serving claim
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:35:36+00:00: hypothesis

- HEAD before checkpoint: ae40e1dbc1db816daf2f450da27f6691cead993e
- Budget remaining: unknown
- Finding: Temporary grouped tile ordering under test for traced 2048x2048 by 2048x12288 BF16 projection; no result yet
- Evidence and artifact paths: Remote temporary FlagGems mm.py backed up as .autoresearch-work/control-mm.py sha256 4cb0cff3addfa452e85724bc400b39ee354f5c79168f3df63f1b0a6c1a708ae8; .autoresearch-work/mm-screen.log; bounded 360-second screen
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Check correctness and paired timings, restore original source and back up evidence before 09:48 UTC
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:36:26+00:00: provisional

- HEAD before checkpoint: 75a2074564f1cfc83be1de3fc83c7b287c6bfd53
- Budget remaining: unknown
- Finding: PROVISIONAL lead: shape-specific grouped MM tile ordering 1.40223 ms control vs 1.16975 ms candidate, approximately 16.6 percent kernel-time reduction on traced 2048x2048 by 2048x12288 BF16 projection
- Evidence and artifact paths: 8 alternating GPU event samples on identical transposed weights; outputs matched control and torch reference; fallback 57x2048x12288 and 7x2051x257 also matched; .autoresearch-work/mm-screen.log; no independent repetition, complete test coverage, production invocation proof for candidate or serving A/B
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Repeat kernel screen independently, back up full source diff and logs, restore unverified candidate before deadline; no speedup or progress gate claim
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:38:14+00:00: provisional

- HEAD before checkpoint: ce54f19b5842ad8949e6aeddc72c1b3362a5a241
- Budget remaining: unknown
- Finding: Promising but unpromoted MM lead replicated in separate process: 2048x2048 by 2048x12288 BF16 grouped tile mapping reduces kernel median 1.40223 to 1.16975 ms and 1.41153 to 1.17371 ms
- Evidence and artifact paths: Two runs of eight alternating GPU event samples, separate seeds; candidate outputs match control and torch reference for observed shape, small-M and tail cases; additional transposed-input shape 1.45587 to 1.17392 ms and numerically matches. 4K trace sampled this shape at 942.19 of 1806.45 ms MM kernel sum. Remote mm-screen.log and mm-screen-2.log; candidate backup .autoresearch-work/mm-candidate.py and .patch local. No complete correctness/fallback coverage, production candidate trace, serving A/B or independent serving replication.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Restore control; next allocation run focused operator tests and matched short serving probes, then guarded independent A/B only if justified
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:39:51+00:00: provisional

- HEAD before checkpoint: 31b8d461dfc53e61cc6419472044830ce63012ea
- Budget remaining: unknown
- Finding: Final follow-up: replicated 16.6 to 16.8 percent kernel-time reduction on observed BF16 2048x2048 by 2048x12288 MM shape; candidate unpromoted and original MM restored
- Evidence and artifact paths: Two exact local off-instance logs .autoresearch-work/remote-handoff/mm-screen.log sha256 41b6905a and mm-screen-2.log sha256 cc7f5271 match remote. Saved provisional candidate .autoresearch-work/mm-candidate.py sha256 a6edcade and mm-candidate.patch sha256 f869ab51, plus mm-screen-first.py and mm-screen.py scripts. Original mm sha256 4cb0cff3 restored and Git inference tree clean. Traced MM projection consumed 942.19 ms of 3891 ms sampled 4K GPU sum; not a wall-time estimate.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Next allocation apply candidate only in v11; add focused noncontiguous, fallback, numerical and real-shape tests; verify candidate executes in production, matched short 4K and 16K serving diagnostics, then guarded matched A/B with independent replication if justified; human MATH-500 remains pending
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T09:54:01+00:00: provisional

- HEAD before checkpoint: 55675bc55c1a67fa7dfc9e50de134f3c871b7dcb
- Budget remaining: unknown
- Finding: User ended session: provisional shape-specific grouped MM candidate is saved but not applied; original v11 inference code remains restored
- Evidence and artifact paths: Draft PR #236; fresh control 4K 2211.04/2526.52 and 16K 1824.16/1894.52 total tok/s with all four rows successful. Independent kernel screens 1.40223 to 1.16975 ms and 1.41153 to 1.17371 ms on BF16 2048x2048 by 2048x12288. Saved .autoresearch-work/mm-candidate.patch and mm-candidate.py; logs .autoresearch-work/remote-handoff/mm-screen.log and mm-screen-2.log. No production candidate proof or serving A/B.
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: In next confirmed GPU allocation restore candidate from saved patch into remote v11, verify hashes, add real-shape numerical and fallback tests, prove production invocation, run matched 4K/16K short diagnostics; only if justified and admitted run guarded unchanged half-runs, preserve and independently replicate. Human runs unchanged MATH-500 for chosen candidate.
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.

### Research checkpoint 2026-11-10T16:04:39+00:00: provisional

- HEAD before checkpoint: 5eda087954f3d3b0b8dd1c67c9c8d0e593447495
- Budget remaining: unknown
- Finding: Prepared unrun GPU candidate tests for traced BF16 projection, transposed and strided inputs, FP16 and small/tail fallback cases; previous GPU deadline expired
- Evidence and artifact paths: Local ignored .autoresearch-work/test_grouped_mm_candidate.py passes Python compilation only; candidate remains unapplied; prior repeated kernel A/B and full control evidence in PR #236
- PR-preserved small benchmark CSVs: none (external backup required)
- Next action: Once operator confirms new allocation and deadline, initialize its budget, verify provenance, apply candidate in v11, execute candidate tests on BI-V150, and run matched diagnostics if correctness passes
- Verification status is researcher-reported; this tool does NOT certify benchmark validity, model accuracy, or independently replicated gains.
