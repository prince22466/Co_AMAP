# Codex Terra 6.1 autoresearch — MiniCPM5-2B BI-V150 v11

**Authority:** read this contract, `program.md`, and `autoresearch/EXPERIMENTS.md` before touching inference code. `program.md` contains executable commands; this file defines acceptance gates. The user will run Codex in an existing, authorized Remote-SSH IDE; committing these files does **not** start GPU work or an autonomous agent.

## Workspace and starting code

- Work **only** in `/root/Optimization/v11`, mirrored by `Co_Kaggle/g7/tests_BI-V150/Optimization/v11`. The initial `FlagGems` and `vllm-plugin-FL` source trees are byte-identical by Git tree SHA to v10 at v11 creation (2026-11-09). Preserve v10's two-stage BF16 GQA prefill optimization (`BLOCK_M=64`), decode page-lookup optimization, and functional tests.
- Do **not** edit, serve from, or copy patches into v0–v10 or `/workspace/FlagGems` or `/workspace/vllm-plugin-FL`. Invoke the *unchanged* externally provided half-run benchmark at `/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py` in place, read-only. Do not modify/copy it for scoring.
- Preserve all untracked work, user edits, and unrelated services. Only terminate your own server and benchmark. Record the remote source SHA/tree/hash before the first experiment. Confirm imported inference modules and kernels really come from v11, not the external benchmark checkout or stale v10.
- Store full profiles, raw benchmark CSVs, hardware traces, intermediate probes, and scratch logs in `v11/.autoresearch-work/`, which is ignored. Commit only verified v11 operator changes and tests. Commit and push research notes even for negative, incomplete, or unverified experiments, explicitly labeling their status. Ignored raw artifacts need an off-instance backup; a GitHub PR containing only artifact paths is not that backup.

## Frozen performance objectives (development half-run only)

Reported reference: `profiling/v10/summary.txt` (2026-11-08, not matched current-machine control):
- 4K `[4096,1024,64,256]`: **2372.38** Total tok/s; **minimum +5% = 2490.999** (display 2491.00); **stretch +10% = 2609.618** (display 2609.62).
- 16K `[16384,1024,64,128]`: **1872.70** Total tok/s; **minimum +5% = 1966.335** (display 1966.34); **stretch +10% = 2059.97**.

**OR, not AND:** reaching the minimum in *either* case is the first goal; 10% or more in either is the desired stretch. Always measure and disclose *both*, and never accept a material overall regression or unapproved two-case tradeoff. Compare unrounded values when possible; a two-decimal printed result of 1966.33 does not cross the 16K minimum. Never silently re-anchor targets to a changing accepted baseline.

These v10 numbers are **reported anchors**, not evidence that any v11 code is faster. First acquire a fresh, unchanged-v11-control A half-run with original CSVs and immutable source/benchmark/model provenance. Perform full matched A/B with equivalent model, server configuration, GPU, dataset, temperature, benchmark, workload order, and restart/warmup conditions. Follow with independent A/B replication (prefer A1/B1/A2/B2 or balanced ordering). The 4K result is particularly cache/order-sensitive: earlier v10 4K individual repetitions spanned ~2100–2680 tok/s. Compare **first vs first** and **second vs second** repetition separately, document prefix-cache hit behavior and warmup/restart order, and balance A/B order when possible. Do not promote a candidate solely because its second 4K repetition is unusually fast. No one-off mean, percentage, or fixed-target crossing proves a causal win.

**Promotion gates:**
1. **Progress checkpoint:** a repeatable, independent **>=3% improvement over the last accepted matched v11 control** in at least one workload, positive overall tradeoff and full correctness -> commit and open a *draft* evidence-backed optimization PR even below the fixed +5% minimum. Small gains may accumulate; weak/noisy gains are hypotheses, not checkpoints. Do not auto-merge.
2. **Minimum achieved:** candidate meets the frozen **+5% v10 threshold** AND demonstrates **>=5% uplift versus a fresh matched v11 starting-code control** for at least one workload, confirmed by independent matched half-runs, all requests and tests successful. Open a milestone PR and **continue toward +10%** if resource budget and a viable next hypothesis remain; do not abandon a verified minimum gain.
3. **Stretch achieved:** candidate meets frozen **+10% v10 threshold** AND **>=10% matched uplift versus unchanged v11 starting code** in either workload, independently replicated, while other workload/accuracy remain acceptable. Open final evidence PR and stop optimization. Do not require both cases to improve by 10%.
4. On allocation/deadline, stop and report the best verified candidate plus missing evidence. Never claim official benchmark scores, independent replication, or MATH-500 pass without those actual results. Human handles MATH-500; kernel-level numerical correctness is Codex's responsibility.

**Accuracy gate:** the project's MATH-500 acceptance threshold is **>=95%**. Do not tune the MATH-500 evaluator or claim that v11 passes from v10's reported 97.1%. The human runs the actual unchanged MATH-500 evaluation of the chosen candidate; a faster candidate failing the required accuracy is not submission-ready. Codex still must ensure numerical/operator correctness before any performance experiment is promoted.

Every required half-run uses unmodified `benchmark_throughput_serve_half.py`, `RUNS=2`, `SKIP_FIRST=0` and exact two workload arrays. These are development half-runs, not competition official full-runs. Preserve prompt success counts (4K 256/256 and 16K 128/128 *per repeat*). Capture TTFT, TPOT, output tokens/sec, total tokens/sec, error counts, GPU utilization and prefix-cache effects. The read-only `autoresearch/evaluate.py` checks CSV structure and numeric thresholds, *not* reproducibility, causal improvement or accuracy.

## Hypothesis queue — profile v11 production before choosing

1. **4K mixed prefill/decode paged attention:** the existing `DECODE_PAGE_LOOKUP` optimization is enabled only for invocation-wide `max_seqlen_q == 1`; mixed calls may lose page-index reuse. Measure frequency, runtime share and actual sequence/query distributions first. Try per-page KV address reuse and independent page-index-load reduction **inside the existing attention operator**. Preserve 2D/3D and all fallbacks. Earlier v4 grouped lookups regressed; measure real dependency chains, not instruction counts alone.
2. **4K decode small-M GEMM/Linear:** inspect real MiniCPM QKV/out/MLP shapes and execution path, tile occupancy, K/N tails, warp mapping, memory coalescing, reuse and accumulator dependencies. The current `mm.py` retains commented grouped mapping while `linear.py` already uses `GROUP_M=8`. A naive grouped-MM port from v7 was *not* proven at serving level. Compare shape-specific operator computation/data movement, not dispatch-only switches.
3. **16K long-prefill attention:** preserve v10's `BLOCK_M=64`, stage-2 specialization and verified page-table contracts. Try safer KV page-index reuse, pipeline overlap, score/softmax accumulator lifetime, and resource-efficient query tiling. The v10 wide tile reported 114 registers/thread, 32KB shared memory and ~13% estimated occupancy; it accelerated prefill despite lower occupancy. Do not blindly increase `BLOCK_M` to 128 without spill/SMEM/serving evidence.
4. **CPU/GPU gaps, preemption/KV-cache memory behavior:** investigate only when fresh v11 traces attribute enough actual wall-clock delay. Look for an allowable operator/data-path correction, **not** capacity, scheduler or benchmark-setting manipulation.
5. **RMSNorm, SiLU and small kernels:** only if fresh Amdahl estimate supports meaningful end-to-end uplift.

Historic v10 short profiled GPU summed-kernel time was 4K attention 55.64%, MM 34.93%; 16K attention 84.00%, MM 11.60%. Those were reduced diagnostics, **not** official wall-time shares; CUDA graph replay limited decode visibility. Reprofile and re-rank, rather than treating them as current v11 facts.

Operator candidates:
- `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py`
- `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150.py`
- `FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/mm.py`
- `FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/linear.py`

## Hard competition constraints

- **禁止通过修改超参开启量化、投机采样。** No quantization, speculative sampling or generation/serving hyperparameter changes for score optimization.
- **禁止benchmark脚本调优。** Do not alter the provided benchmark, accuracy evaluator, dataset, prompts, lengths, concurrency, scheduling of scored workloads, or success accounting.
- **禁止没有算子优化的情况下切换算子，删掉框架主要算子选择逻辑等方式来进行模型性能优化。** No operator-selection-only or backend-switch speedup; no deletion of major dispatch/fallback logic. Every candidate must implement real equivalent-semantics kernel/operator compute or data-movement improvement.
- **禁止通过直接merge FlagOS推理框架最新分支代码获得性能优化。** No wholesale latest FlagOS merge.
- No weakening tests/tolerances, editing model weights, changing `FULL_DECODE_ONLY`, GPU memory fraction, max model length, or environmental policy for benchmark-score manipulation. Do not use a diagnostic probe as a replacement for the scored half-run.
- Rules: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn

## Time-aware autonomous iteration, within the interactive SSH session

An operator-confirmed allocation/deadline is mandatory. Use `autoresearch/session_budget.py init` once per GPU session, `status` before each experiment and `admit --runs N` to plan complete paired runs; only `run-half` may launch an expensive unchanged half-run. Initial estimate: **105 minutes observed runtime + 20 minutes per-run safety margin + 25 minutes preserved for handoff**. Thus, admit one run only at >=150 minutes remaining and two at >=275 minutes, rising automatically with slower observed completed runs. These are conservative estimates, not completion guarantees. If the deadline is unknown, expensive runs are prohibited. A full independent A1/B1/A2/B2 requires approximately 4 x 125 + 25 = **525 minutes** initially, beyond a 330-minute session; never promise independent confirmation in one allocation. Explicitly label an A/B pair as provisional. Timebox the benchmark subprocess so that the final preservation reserve is protected. No unattended GPU allocation, remote access, or self-relaunch is implied.

For each hypothesis:
1. Profile production shapes/stages, GPU *summed kernel* time vs serving wall time, idle/scheduler/KV behavior, and compute an Amdahl upper bound. Record source hashes and why this hypothesis ranks.
2. Make **one** small v11 operator-level change. Run existing attention, decode, mixed-paging, BF16, shape-tail, mask, fallback and numerical tests; add focused coverage for altered paths. For MM/Linear add real-shape tests. Failed correctness -> revert *only* your experiment.
3. Run paired warmed kernel A/B on identical inputs and actual pages; report medians/spread, resource usage, registers, SMEM, spills and bandwidth when available. Prove optimized kernel executes during production.
4. Run matched short 4K and 16K serving diagnostics with restarts/order controls; measure errors, cache behavior, total/output tok/s, TTFT and TPOT. These are screening tests, never official scores.
5. Write a go/no-go in `autoresearch/EXPERIMENTS.md`. Spend ~20–30 minutes on an initial kernel screen per hypothesis, unless strong profiling evidence justifies more. Commit a full half-run only for likely serving gains with enough GPU allocation.
6. Run **both** unmodified workloads and preserve all four SUCCESS raw rows, summary CSV, source/benchmark/model hashes and full traces. Replicate control/candidate independently when considering promotion. A syntactically valid CSV is not proof.
7. Record outcome and regressions. Keep accepted code; discard negative experiments without rewriting history. Open *draft* PRs for confirmed progress (>=3%), milestones (>=5%) and stretch (>=10%); link raw evidence locations. Never auto-merge; keep researching toward stretch while allocation permits.

**Continuous preservation contract:** At session start, after every valuable positive or negative finding, at least every 30 minutes of active research, before any half-run, and immediately when remaining time is <=45 minutes or Codex credit pressure is suspected, call `autoresearch/checkpoint.py` with a nonempty finding, evidence/artifact paths, next action and honest verification status. Use `--push` on the intended open PR branch to commit only the tracked experiment ledger and handoff; confirm the commit appears on the PR. Codex cannot reliably read remaining credit balance, so periodic preemptive checkpoints are mandatory. If pushing is unavailable, clearly report local-only persistence and do not gamble the remaining budget on long evaluation. The benchmark runner can checkpoint before/after a run, but does not inspect results or independently certify them. Preserve raw CSVs/traces off-instance before expiration. Do not auto-merge.

Stop cleanly at stretch success, exhausted budget, user instruction, or an unsolved blocker. Do not promise background progress or claim work occurred on the remote machine until actual SSH/hardware runs prove it.
