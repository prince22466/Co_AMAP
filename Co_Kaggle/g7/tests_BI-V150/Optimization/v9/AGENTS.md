# Codex instructions — MiniCPM5-2B BI-V150 v9

Applies to the v9 subtree. Read autoresearch/README.md and autoresearch/GOAL.md first.

## Goal

Optimize total tokens/s with the EXACT official workloads on the authorized BI-V150:
4K [4096,1024,64,256] >= 2700 OR 16K [16384,1024,64,128] >= 1500.
Stop only on a verified target, a user-set deadline, or a hard blocker. Open a
checkpoint GitHub PR for each confirmed improvement even below the target,
and continue the next experiment from the last accepted best commit.

Historical v8 full-run reference only: 4K 2331.29; 16K 1251.22 total tok/s;
math_500 98.1%. Never treat these as a fresh A/B control.

## Inviolable competition constraints

- Use only MiniCPM5-2B, provided FlagOS vllm-plugin-FL and FlagGems.
- Never edit benchmarks, accuracy datasets, evaluation scripts, official
  workload shapes, warmup/runs, or score extraction logic.
- Never enable quantization or speculative decoding by hyperparameter changes.
  Do not change serving arguments, precision, concurrency, batching knobs,
  model configuration, graph mode, model path, or generation settings to game
  throughput.
- Do not obtain a gain solely by switching selected operators, deleting
  framework dispatch logic, or merging recent upstream FlagOS code.
- An accepted implementation must actually optimize a kernel, an operator,
  or its data movement within v9; preserve unsupported-shape fallbacks and
  normal operator selection. A narrowly needed launch integration is allowed
  only to expose that genuine kernel-level optimization.
- Never alter v8/other versions or external vendor/official source directories.
  Do not commit private credentials, SSH keys, model weights, raw profiler
  datasets or bulky generated logs.

Competition: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn

## Ordered experiment queue

1. 16K attention: global-memory dependencies, page-table accesses, shared
   memory footprint, register spills, K/V and Q tile reuse, occupancy.
2. Mixed prefill/decode: phase interference and scheduling causality; profile
   mixed versus isolated phases without changing the allowed workload.
3. 4K attention and prefill GEMM: memory locality, grouped M tiling, reuse,
   efficient address generation and accumulators inside existing kernels.

Primary files:
- vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py
- vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150.py
- FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/mm.py
- FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/linear.py
- vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py
- vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py

Learn from previous negative results: v4 grouped lookup, v6 KV tiling and
v7 GEMM stacking failed to produce a robust serving gain. Do not repeat
without a new profiler-supported causal hypothesis. Verify optimized code
actually runs via production traces, not imports alone.

## Mandatory research cycle

1. Record best Git SHA, machine, load, kernel trace and fixed serving config.
2. Measure a FRESH full official 4K/16K v9 baseline on the BI-V150.
3. Profile before patching; attempt one precise operator-level hypothesis.
4. Rsync the v9 code only to an agent-owned remote scratch path and restart
   ONLY the agent-owned server. Use remote.py; never overwrite original v9.
5. Run GPU regression tests and fallback correctness. Prove production kernel
   invocation. Execute FULL official benchmark unchanged (RUNS=4, skip=1)
   on BOTH workloads; evaluate candidate accuracy on full MATH-500.
6. Compare matched official raw and summary CSVs with evaluate.py. Requires
   MATH-500 >= 0.95; 0 failed requests; at least 1% reliable throughput
   gain in one case; no other workload throughput regression beyond 1%;
   no mean TTFT regression beyond 1% for either case. Repeat noisy A/B.
   The helper checks numbers, not causal provenance: inspect actual logs.
7. For accepted gains: commit only v9 kernel/tests/finding notes, push to a
   uniquely named branch, immediately open a GitHub PR with full evidence,
   and base the next experiment on the accepted commit (stacked PRs okay).
   Do not wait until target is achieved. Never automatically merge PRs.
8. For rejected experiments: log the negative result, selectively restore
   only experiment modifications, and move to next hypothesis. Preserve
   unrelated user work and the last accepted best commit.
9. At deadline or blocker: stop, retain logs, report best branch/PRs and
   actual results; never invent evidence or claim future work ran.

Microbench improvements alone are insufficient. A threshold without full
benchmark, accuracy and regression gates is NOT success.

## Remote safety and reporting

Only use preapproved V9_SSH_HOST and verified known_hosts. Remote writes
must stay in V9_REMOTE_ROOT (a dedicated /root/Optimization/v9_autoresearch
scratch directory), never /root/Optimization/v8, original /v9, or /workspace
source. Stop only agent-owned PID. Never expose private keys, force-push main,
kill unrelated services, or bypass host-key checking.

Update autoresearch/EXPERIMENTS.md every round. Also write
.autoresearch-work/round-report.json (format in GOAL.md). Keep raw artifacts
off-repo; include reproducible paths/hashes in every checkpoint PR.
