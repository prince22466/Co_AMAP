# v12 autoresearch — append-only results

**Initial status: no v12 GPU experiments have run for this setup PR.** Numbers below are reported historical anchors, NOT independently matched evidence.

| Scenario | v10 reported tok/s | >=5% goal | Fresh v12 matched control |
|---|---:|---:|---|
| 4K [4096,1024,64,256] | 2372.38 | 2490.999 | Not run |
| 16K [16384,1024,64,128] | 1872.70 | 1966.335 | Not run |

The standalone unmerged PR #237 has a **default-off, untested** sparse kernel prototype. It is not part of v12 main or proof of a speedup.

## Experiment entry template (append, do not erase failures)

### [UTC] [E1/E2/E3/S1/S2/S3] — [hypothesis / provisional / verified / rejected]
- Code HEAD; diff and source/model/benchmark hashes; original dense control SHA:
- Actual GPU bottleneck and estimated end-to-end effect; workload/cache/graph state:
- Correctness and numerical metrics vs **dense**; if sparse, vs its explicit sparse oracle:
- Kernel A/B: median/spread, bytes or tile/FLOP counts, repeats, resource counters, logs:
- Actual v12 production invocation and short 4K/16K serving A/B:
- Long-context ability versus dense (beginning/middle/end across seeds), when approximate:
- Half-run: original four SUCCESS rows, 256/128 request counts, TTFT/TPOT, matched replication:
- MATH-500 (human) and organizer eligibility, if selected; no undocumented assumptions:
- Decision, accepted SHA or rollback, checkpoint/PR, raw evidence off-instance, next step:

The helper scripts check deadlines and CSV structure; they **cannot** certify model quality, reproducibility or eligibility. Preserve negative evidence, failures and rejected candidates.

### 2026-11-11 10:23 UTC — session setup, provisional
- Allocation confirmed until 16:00 UTC (17:00 Warsaw); budget guard initialized once.
- Remote checkout was an extracted source tree without research files or its own Git checkout; missing research files were copied from local main bbdfba2665e6f3aa8c9d17d7acca8173a181ee8b. The attention source SHA-256 matches local: `6e3e8f7cbc24140a9cfd3ef3d2ed8d99cf7ae0f4e0147ac3d9d9b0775925931a`. Complete tree identity is not yet verified.
- One idle BI-V150, PyTorch 2.10.0; plugin and FlagGems import from `/root/Optimization/v12`; external half-run SHA-256 `873f17f932d4643222be9903ba3b569c1863c4e2d246ceafbe6f75ebe7163ffc`.
- 21 research helper tests passed; baseline attention correctness is running. No throughput or accuracy results yet.
- Next: verify full source provenance, finish correctness, profile production attention, screen exact and sparse candidates. GitHub branch checkpoint uses connector because remote Git access is not yet established.

### 2026-11-11 10:25 UTC — baseline correctness and source verification
- Remote attention, decode, and mixed-paging suite: **21 passed**, one Triton benchmark fallback warning, 14.97 s. Research helper suite: 21 passed.
- SHA-256 compared 5,177 tracked files from local main: all 5,139 present remote files matched; 38 `FlagGems/.github` metadata files were absent. No source content mismatch observed. The remote extracted tree still is not a Git checkout; source identity should not be described as a checked-out Git SHA.
- A dedicated baseline server was started with fixed runbook flags; its first graph capture is in progress. Baseline quick serving diagnostics are queued behind health readiness. No A/B measurements yet.
- One physical GPU was detected, so multi-GPU Ring Attention is not applicable in this allocation.

### 2026-11-11 10:29 UTC — E3 prefill page-table reuse, rejected
- Hypothesis: reuse the existing decode page-table lookup in standard BF16 GQA prefill, reducing redundant physical-page lookups. Isolated experimental copy; original production source unchanged. Imported from `.autoresearch-work/screen`, baseline/variant text SHA-256 after Python newline normalization `263c73746e48fd16989b6c6587a485a1f7505f303ba93ff78d377159f4fecac6` / `7c7c4556a0f2441fd48c92296515d0e6317611b226a551cb4f48fbd6a288d162` (raw production source SHA recorded above).
- 12 interleaved paired GPU-event measurements per shape, identical random BF16 tensors with shuffled 16-token pages; comparisons were bit-identical for these cases. q=63/context=4097: 0.5355 vs 0.6190 ms (+15.6%); q=2048/context=2048: 3.5781 vs 4.6124 ms (+28.9%); q=2048/context=14336: 17.5197 vs 22.8435 ms (+30.4%). q=1/context=16377 is unchanged code and measured 2.3934 vs 2.4323 ms. These are isolated operator timings, not production traces; registers, spills and detailed production GPU share remain unmeasured.
- Reject; no serving A/B justified. Not a verified speedup. Logs and candidate/harness are in `.autoresearch-work/screen/`; off-instance backup pending.
- Baseline non-scoring quick run (concurrency 8, output 128): 4K 16/16 successful, 2459.48 total tok/s, mean TTFT 6038.05 ms, TPOT 60.03 ms; 16K 8/8 successful, 3196.27 total tok/s, mean TTFT 19050.60 ms, TPOT 170.62 ms. No matched restart and not an official workload; cache conditions are not controlled. 4K/16K logs under `.autoresearch-work/quick/baseline/`.
- Next: obtain production attention profile and test independent exact dataflow hypothesis; explore sparse separately with a middle-context quality gate.

### 2026-11-11 10:31 UTC — S1 sparse prototype reference gate, inconclusive / blocked
- Isolated source and tests from PR #237 under `.autoresearch-work/screen/`; serving source not modified. Pytest: **4 failed, 3 passed**. All four explicit FP32 PyTorch sparse-reference comparisons failed at atol/rtol 0.05; mismatches 2.0–3.5%, maximum absolute errors 0.26–0.42. This includes the window-covering-all-context case.
- Diagnostic on the full-context case: original v12 dense vs the same FP32 oracle also fails (2,641/133,120 elements, max absolute 0.41796875, mean absolute 0.01613); sparse output vs original dense is bit-identical in that case. Thus the failures do **not** isolate a sparse-kernel defect; oracle/baseline discrepancy is unresolved. Prior dense regression tests compare vLLM's implementation, not this FP32 oracle.
- Do not promote or time this approximate candidate as valid until the oracle discrepancy is explained and sparse-reference correctness is demonstrated. No middle-context model retrieval, serving, accuracy or organizer eligibility testing was performed. Failure logs and diagnostic source are under `.autoresearch-work/screen/`.

### 2026-11-11 10:33 UTC — dense/FP32 oracle cross-check
- Recomputed the full-window case with the same BF16 inputs and explicit causal GQA reference on CPU in FP32. GPU PyTorch reference vs CPU: maximum absolute difference 0.0004883, zero failures at atol/rtol 0.05. Original dense Triton output vs CPU: maximum absolute 0.41796875, 2,641/133,120 tolerance failures, mean absolute 0.01613. This confirms the GPU/CPU oracle agreement for this case, but the source of the discrepancy with the existing dense implementation is not yet established; do not extrapolate to model quality or causation.
- Source and logs: `.autoresearch-work/screen/oracle_cpu_check.py` and `oracle_cpu_check.log`. Production code remains unchanged. Investigate the dense implementation and reference semantics before interpreting any sparse quality result.

### 2026-11-11 10:39 UTC — production GPU trace, provisional
- Restarted owned v12 server with the prescribed flags under `ixsys -t cuda_device -d 140 -l 180`; completed short 4K and 16K serving diagnostics. The first client attempted health just before startup and exited; a second client ran both workloads, though its SSH connection dropped after the 16K run began. Logs show 16/16 and 8/8 successful requests respectively. Traced runs are diagnostic only, with imperfectly controlled cache and tracing overhead.
- GPU engine SQLite recorded 51,574 kernel events, 41.7355 s summed kernel time in the captured interval. `kernel_unified_attention`: 3,696 launches, 29.9128 s, **71.672% of summed kernel time**; `mm_kernel`: 9.1002 s, 21.805%. Attention reported max VRFs 114, SRFs 99, dynamic shared memory 32,768 bytes, localmem 0 (profiler fields, not measured occupancy or confirmed spill counts). Graph activity and the incomplete phase boundaries make this aggregate unsuitable for separate 4K/16K, prefill/mixed/decode, or wall-clock attribution. Kernel name alone does not prove the loaded source implementation.
- Illustrative Amdahl calculation only: if a 7% attention-time reduction applied uniformly to this GPU-only profile, total summed GPU-kernel time would fall by about 5.0%; actual serving throughput effect is unknown and will be smaller if other time or workload phases dominate.
- `production` trace, `engine.sqlite`, `kernel_summary.txt`, server/client logs in `.autoresearch-work/profile/`; backing up off-instance. Next separate workload phases and prove actual operator invocation; screen exact attention changes with correct reference and interleaved A/B.
- Dense oracle diagnostic: v12 and imported upstream callable were bit-identical for q=1/9/65 cases, and both differ from CPU reference (q=65/context=33: 2,641/133,120 errors at 0.05 atol+rtol, max 0.41797). The upstream callable may be affected by plugin registration, so this is not yet proof of an independent upstream defect. Raw `.autoresearch-work/screen/dense_oracle_diagnostic.log`.

### 2026-11-11 10:40 UTC — E1 prefill query-tile geometry, rejected
- Following attention's 71.7% aggregate GPU kernel-time share, screened isolated `BLOCK_M=128` and `BLOCK_M=32` variants against the v12 `BLOCK_M=64` prefill path. This changes actual query/KV reuse and work partition, not operator dispatch. Each uses identical tensors and 12 interleaved event timings; sampled outputs were bit-identical to baseline. The independent CPU reference discrepancy remains unresolved.
- q=63/context=4097: baseline 0.5335 ms, 128 0.6967 ms (+30.6%); separately baseline 0.5352 ms, 32 0.5520 ms (+3.1%). q=2048/context=2048: baseline 3.5867 vs 4.4641 ms (+24.5%) for 128, and 3.5757 vs 6.1593 ms (+72.3%) for 32. q=2048/context=14336: baseline 17.5266 vs 21.3181 ms (+21.6%) for 128, and 17.4821 vs 30.3777 ms (+73.8%) for 32.
- Reject both; no serving A/B warranted. Exact operator source remains untouched. Logs and scripts under `.autoresearch-work/screen/tile_128.log`, `tile_32.log`, and `tile_screen.py`; source text normalized SHA-256 128 `526f8f25b237465d93a635c52e133aee4507e0593eac4b9cc2caea644a0e139a`, 32 `4312129abc22a3f6b3a76b4a8b3698f5c1af4a9c26a01b7642d96e56cd89def3`. Resource counters for these variants not yet measured.

### 2026-11-11 10:41 UTC — E1 wider KV tile, mixed result; global change rejected
- In an isolated source copy, used `TILE_SIZE_PREFILL=64` for the BF16 GQA prefill path versus baseline 32. This changes attention loop work and KV tile geometry. Twelve alternating event timings, identical tensors and shuffled pages; candidate outputs passed comparison with baseline, max absolute differences <=0.00098 on sampled prefill cases (not a CPU-oracle resolution).
- q=63/context=4097: median 0.5344 -> 0.4847 ms, **9.3% faster**. q=2048/context=2048: 3.5750 -> 6.0033 ms, **67.9% slower**; q=2048/context=14336: 17.5005 -> 29.9223 ms, **71.0% slower**. Rejected as a blanket prefill change. Source normalized SHA `ffe9b8af56dacc25c5b0fc77ae146e45e09cdb9caa347b140e70b8f827604832`; log `.autoresearch-work/screen/kv_tile_64.log`.
- Potential research lead: gate the wider tile to short prefill queries only, preserving larger/mixed behavior and validating 2D/3D, tail/page correctness. The captured production trace is dominated by larger attention launch grids: grids 257–264 account for ~29.72 s of 29.91 s attention kernel time. Small grids 15, 19, 29 account for ~0.187 s combined. Grid alone does not establish query length or a serving-phase breakdown, so these numbers are only a reason to expect limited opportunity in this diagnostic, not a production impact estimate. No serving gain demonstrated. Original production source untouched.

### 2026-11-11 10:42 UTC — E1 prefill pipeline depth, promising / not yet accepted
- Isolated BF16 GQA prefill `launch_num_stages=1` vs original 2; changes Triton software-pipeline scheduling inside the existing attention kernel. 12 alternating GPU-event timings per shape with identical shuffled-page inputs. Outputs were bit-identical against baseline in sampled cases; independent FP32 oracle discrepancy remains unresolved.
- q=63/context=4097: 0.53264 -> 0.45380 ms (-14.8%); q=2048/context=2048: 3.58159 -> 3.19208 ms (-10.9%); q=2048/context=14336: 17.52477 -> 15.12698 ms (-13.7%). q=1 decode path unchanged. This is kernel-only evidence; no serving or end-to-end improvement established. Normalized source SHA candidate `909b4b423dfa561a06a19549e6295909e10b7ba937d8d10654cc723ab6e5dc2a`, raw `.autoresearch-work/screen/stage_1.log`.
- A depth of 3 regressed q=63 by 51%, and the long cases by 90–92%; reject. Raw `.autoresearch-work/screen/stage_3.log`.
- Next: preserve baseline, apply stage-1 candidate in v12, run comprehensive correctness including mixed pages and decode/fallback plus CPU reference diagnostic, inspect kernel resources, prove serving invocation, and alternate matched 4K/16K short-serving restarts. Only proceed to unchanged half-runs after the serving go gate and budget admission.

### 2026-11-11 10:43 UTC — stage-1 candidate regression gate
- Candidate applied to remote source, preserving originals under `.autoresearch-work/stage1/`. Raw operator SHA-256 `1c65864df8333a6ff91f005b29547441ef128a33c847466c09accca0cf6a0d28`; only prefill software pipeline stage count changed. Test assertion changed from two to one stage for the optimized mixed case.
- Required attention/decode/mixed paging GPU suite: **21 passed**, 1 warning, 16.34 s; raw `.autoresearch-work/stage1/correctness.log`. Existing tests compare against imported vLLM and have not resolved the independent FP32 oracle discrepancy. Further FP16/fallback and numerical coverage remains necessary.
- Candidate source and corresponding test preserved separately on this draft branch in commits `26b4cbc320f6f8648c35f6ab5ab9f24582c9dcf0` and `d60c295317b6459bea545f6aa8ab100d2edde34b`; raw remote uses CRLF, normalized source content corresponds. Candidate server with fixed flags is starting for short-serving B1; operator execution/source hash still needs proof. No end-to-end claim.

### 2026-11-11 10:45 UTC — trace backup
- GPU trace and engine SQLite were copied off-instance to the workspace under `Optimization/v12/.autoresearch-work/remote_backup/profile/`. Local SHA-256: `production` = `712265ab64ad567945658788c9ad25cf905caccefe190fd311ad8262ea58b8b8`; `engine.sqlite` = `0090d2936e4c5358dd95d97881bf4d526390200bca39b30001eeb51eb8fe2be2`. These hashes are local backup hashes; source hashes were not independently compared after copy. Traced 16K log reports 8/8 completed despite the SSH client session dropping.

### 2026-11-11 10:48 UTC — stage-1 candidate B1 short-serving diagnostic, mixed
- Candidate B1, fixed server flags and same non-scoring quick configuration as initial control A1; all requests completed. 4K: 16/16, 2263.25 total tok/s, mean TTFT 7245.44 ms, TPOT 59.91 ms; A1 2459.48, 6038.05 ms, 60.03 ms. 16K: 8/8, 3567.85 total tok/s, mean TTFT 16710.78 ms, TPOT 155.12 ms; A1 3196.27, 19050.60 ms, 170.62 ms. Unreplicated deltas about -8.0% / +11.6% total throughput; no inference of causality or official score.
- Material 4K regression if repeatable; do not advance to half-run on this single mixed result. Alternate fresh A2/B2 restarts, same flags/order and inspect cache/source/production invocation. Source/logs `.autoresearch-work/stage1/quick_b1/`, `serve/candidate_b1.log`; baseline in `.autoresearch-work/quick/baseline/`.

### 2026-11-11 10:54 UTC — fresh control A2
- Restored original v12 operator SHA-256 `6e3e8f7cbc24140a9cfd3ef3d2ed8d99cf7ae0f4e0147ac3d9d9b0775925931a`, restarted server with same flags, 4K then 16K short diagnostics. 4K 16/16: 2460.35 total tok/s, mean TTFT 6022.39 ms, TPOT 60.10 ms; 16K 8/8: 3195.17 total tok/s, mean TTFT 19069.86 ms, TPOT 170.55 ms. Both closely reproduce A1; raw `.autoresearch-work/stage1/quick_a2/`.
- Restored stage-one candidate SHA `1c65864df8333a6ff91f005b29547441ef128a33c847466c09accca0cf6a0d28`; candidate B2 server starting. Do not treat first candidate's 16K gain as sufficient if 4K regression repeats.
- Additional oracle diagnostic: GPU-ordered K/V pages exactly matched CPU reconstruction from the same block table; the observed CPU-reference discrepancy is not explained by a misordered PyTorch GPU gather. Raw `.autoresearch-work/screen/check_gather.log`.

### 2026-11-11 11:00 UTC — stage-1 candidate B2 short-serving diagnostic
- Independent restart; same settings and order as A1/B1/A2. B2 4K: 16/16, 2500.87 total tok/s, mean TTFT 5926.09 ms, TPOT 59.13 ms; A2 2460.35, 6022.39 ms, 60.10 ms. B2 16K: 8/8, 3546.01 total tok/s, TTFT 16764.00 ms, TPOT 156.52 ms; A2 3195.17, 19069.86 ms, 170.55 ms.
- The 16K diagnostic gain repeats (~11.0% and ~11.6% relative to controls); the 4K results differ substantially across candidate restarts (2263.25 and 2500.87 vs controls 2459.48 and 2460.35). The first run had peak 12 concurrent requests vs 11 on the others; scheduling/cache dynamics and actual production source invocation require investigation. Short workloads do not establish official 64-concurrency results, TTFT limits, competition score or causal replication at full half-run. No half-run launched.
- Raw `.autoresearch-work/stage1/quick_b2/` and candidate server log; next repeat/control 4K and prove execution, then apply the full gate without ignoring the material first-run regression.

### 2026-11-11 15:07 UTC — stage-1 candidate third 4K restart
- B3 short diagnostic: 16/16 successful, 2512.00 total tok/s, mean TTFT 5883.39 ms, TPOT 58.99 ms. A1/A2 controls 2459.48/2460.35; candidate B1/B2/B3 2263.25/2500.87/2512.00. B3 did not run 16K. Preserve the first-run 4K regression as observed variability, not a proven reproducible regression or something to discard.
- User updated remaining research window to 40 minutes at approximately 15:07 UTC; actual GPU allocation still ends 16:00 UTC. Guard status: about 53 minutes left, 150 required for a half-run, admission denied. No official half-run was started. Preserve at least 25 minutes for evidence, off-instance backups and PR handoff. No >=5% official goal or accuracy/eligibility claim.
