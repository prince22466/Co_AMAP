# v11 — Codex Terra 6.1 autoresearch runbook (Remote-SSH)

Read [AGENTS.md](AGENTS.md), [autoresearch/EXPERIMENTS.md](autoresearch/EXPERIMENTS.md), and [readme.md](readme.md). Work **only in `/root/Optimization/v11`**. This repository PR configures a task; it does not remotely launch Codex or GPU work. Starting operator sources (FlagGems + vLLM plugin) must remain the **v10 optimized checkpoint**, including BF16 GQA prefill `BLOCK_M=64`, until an evidence-backed v11 operator experiment changes them.

## Targets and acceptance

Official-format **development half-run** (not full-run), both `[4096,1024,64,256]` and `[16384,1024,64,128]`, unmodified benchmark:
- Frozen reported v10 reference: **4K 2372.38** / **16K 1872.70** Total tok/s.
- **Minimum +5% (OR):** 4K >= **2490.999**, or 16K >= **1966.335**.
- **Stretch +10% (OR):** 4K >= **2609.618**, or 16K >= **2059.970**.

Preserve raw precision in comparisons. Both cases must pass all requests, and a fresh independent matched v11 starting-code control must establish >=5% causal uplift for minimum or >=10% for stretch. Repeated independent A/B pairs are required. Disclose the other workload and refuse unapproved material regression. At repeatable >=3% incremental improvement vs last accepted best, open a *draft progress PR*, then keep searching. At +5% milestone open a PR but keep searching for +10% if feasible. At independently confirmed +10% open final PR, stop. **No auto-merge.** Human owns MATH-500; the project's accuracy acceptance threshold is **>=95% on the unchanged MATH-500 evaluation**. Never infer a v11 accuracy pass from historical v10 results.

## 0 — Check current source, workload and GPU budget

```bash
cd /root/Optimization/v11
pwd
git status --short
git rev-parse HEAD
git ls-tree HEAD FlagGems vllm-plugin-FL
sha256sum /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py
grep -n 'BLOCK_M = 64\|decode_page_lookup =' vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py
```

Verify remote directories, tracked local edits, free port 9031, model files, GPU status, remaining allocation time or explicitly record unknown. Do not assume remote v11 matches GitHub simply because it exists. The benchmark path is *only* used for running the untouched benchmark. Do not run v10 accidentally.

## 0a — Confirmed GPU deadline and continuous PR checkpoints

This is an executable gate, not a reminder. The **operator** supplies the actual duration/deadline for each new GPU allocation; never invent remaining time. For a verified 5.5-hour session only:

~~~bash
cd /root/Optimization/v11
python3 autoresearch/session_budget.py init --duration-minutes 330
python3 autoresearch/session_budget.py status
python3 autoresearch/session_budget.py admit --runs 2
~~~

Or use `init --deadline-utc 2026-11-10T16:00:00Z` with the real operator-confirmed expiry; do not combine both options. `admit` exits 2 if there is insufficient verified time, 1 on invalid state. The initial model is 105 minutes measured half-run + 20 minutes safety per run, and a protected 25-minute preservation reserve: require >=150 minutes for one and >=275 for two. `record --elapsed-minutes N` records a known historical run and increases future estimates when slower. A 330-minute session does **not** fit four independently replicated half-runs (A1/B1/A2/B2); mark an A/B-only result provisional. Never backdate, reset, or reinitialize the deadline during a session merely to admit a run.

**Before experimenting and every <=30 minutes**, after every important failure/result, before each costly run, and immediately when time <=45 minutes or Codex credits might expire, publish an append-only, honestly labeled checkpoint:

~~~bash
python3 autoresearch/checkpoint.py \
  --summary "Exact finding or failure; provisional unless replicated" \
  --evidence "source SHA; matched metrics; raw filenames/checksums; errors" \
  --next-step "Single highest-priority next action" \
  --status provisional \
  --push
~~~

`--push` commits **only** the tracked experiment ledger and handoff, then pushes the current non-main branch. Use it only on the reviewed branch backing the intended open PR, and verify that the PR shows the new commit. If GitHub push/auth fails, local files remain but **PR preservation is not complete**; fix access or report the blocker before consuming another long run. Codex cannot observe its remaining credit balance reliably; do not postpone the handoff until the end. The guarded half-run can preserve small (<=1 MB), schema-checked raw_runs/summary CSV files in tracked PR evidence when Git push works; optional eligible files can be attached with repeated `--csv /path/inside/.autoresearch-work/.../raw_runs.csv` parameters to `checkpoint.py`. Verify the PR commit contains the CSVs. **Copy large traces, logs and unsnapshotted raw evidence off-instance** before the rental expires. A note containing only ephemeral paths is not a backup.

## 1 — Start an owned v11 serving process (Terminal A)

```bash
cd /root/Optimization/v11
mkdir -p .autoresearch-work/serve_runs
serve_log=$(mktemp "$PWD/.autoresearch-work/serve_runs/serve_XXXXXXXX.log")
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
unset VLLM_FL_USE_FLAGGEMS_ATTN VLLM_ATTENTION_BACKEND VLLM_BATCH_INVARIANT
set -o pipefail
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  2>&1 | tee "$serve_log"
```

Do not kill an unrelated server on port 9031. Keep the owned session in a visible terminal or an inspectable named `tmux` session. Confirm health with `curl -f http://127.0.0.1:9031/health` before benchmarking. In a separate terminal, record actual source import locations and confirm the v11 kernels execute in a production GPU trace.

## 2 — Mandatory GPU correctness tests

```bash
cd /root/Optimization/v11
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
python3 -m pytest -q -s --noconftest \
  vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_mixed_paging.py
python3 -m unittest discover -s autoresearch/tests -v
```

When changing `mm.py` or `linear.py`, add and execute focused GPU tests for real BF16/FP16 projection shapes, small-M, K/N tails, non-contiguous inputs if supported, fallback behavior, numerical tolerances and NaNs. When changing attention, test 7/9/65-token query tails, mixed pages, 16K prefill, pure decode, mixed prefill/decode and fallback. Tests must actually pass on BI-V150; a docs-only PR cannot verify them.

## 3 — Phase-resolved production profile and paired kernel screening

Start with **real v11** production traces for uncached 4K/16K prefill, mixed prefill/decode, and pure decode. Separate prefix-cache hits from uncached work. Record per-operator real shapes and launch counts, GPU active versus idle/wait, CPU scheduling, preemption and KV-cache statistics. Distinguish `summed GPU kernel time` from *end-to-end serving wall time*. CUDA graph replay can hide decode activity from naive traces. Re-rank hypothesis queue using current evidence.

First hypotheses, if supported: mixed-attention per-page KV address/index reuse; 4K small-M MM/Linear tile utilization and weight accesses; 16K prefill KV memory dependency and pipeline utilization. Every candidate **must change real kernel/operator computation or data movement**, never framework dispatch alone.

For one operator experiment at a time: A/B identical tensor shapes/dtypes/page layouts with repeated warmup, alternating order, resource counters, output comparisons and production invocation trace. If correctness or kernel-level improvement fails, revert *only* your experimental edit. Record rejected hypotheses and reasons in `autoresearch/EXPERIMENTS.md`.

## 4 — Short serving diagnostics (screen, never score)

```bash
cd /root/Optimization/v11
mkdir -p .autoresearch-work/quick
quick_dir=$(mktemp -d "$PWD/.autoresearch-work/quick/probe_XXXXXXXX")
vllm bench serve --backend vllm --model minicpm \
  --tokenizer /workspace/MiniCPM5-2B --endpoint /v1/completions \
  --host localhost --port 9031 --dataset-name random --ignore-eos \
  --random-input-len 4096 --random-output-len 128 \
  --max-concurrency 8 --num-prompts 16 \
  2>&1 | tee "$quick_dir/smoke_4k.log"
vllm bench serve --backend vllm --model minicpm \
  --tokenizer /workspace/MiniCPM5-2B --endpoint /v1/completions \
  --host localhost --port 9031 --dataset-name random --ignore-eos \
  --random-input-len 16384 --random-output-len 128 \
  --max-concurrency 8 --num-prompts 8 \
  2>&1 | tee "$quick_dir/smoke_16k.log"
```

Repeat with matched restarts, unchanged inputs and preferably balanced A/B order. These small short-output probes do *not* reproduce the scored workload or its concurrency/long decode; do not compare their tokens/sec directly with the target table. Escalate to half-run only on a documented causal go/no-go.

## 5 — Deadline-gated, unchanged development half-run (Terminal B)

**Do not invoke the benchmark directly** during autoresearch: use the guard, which checks the confirmed deadline, checks server health, snapshots pre-run evidence, executes the unchanged external benchmark, limits runtime to protect the preservation reserve, and checkpoints its exit state. It does **not** change benchmark code, inputs, repeat count, or server settings.

~~~bash
cd /root/Optimization/v11
python3 autoresearch/session_budget.py status
python3 autoresearch/session_budget.py admit --runs 1

# Requires Git authentication on the checked, non-main PR branch.
# If PR push cannot succeed, preserve evidence first and do not gamble a long run.
python3 autoresearch/session_budget.py run-half --push-checkpoint
~~~

The command uses the external, untouched `benchmark_throughput_serve_half.py` with fixed `RUNS=2`, `SKIP_FIRST=0`, and both workloads `[[4096,1024,64,256],[16384,1024,64,128]]`. Each run gets a new ignored `.autoresearch-work/bench_runs/guard_*/` directory with `benchmark.log`, `run.json` and the benchmark-generated CSVs; eligible CSVs are copied into tracked `autoresearch/evidence/` and pushed on the final checkpoint. Inspect `tail -f .../benchmark.log` for progress. An exit code of zero or an automatically published checkpoint is **not** acceptance; raw CSVs still need explicit verification, matched A/B and independent reproduction.

Use new run directories and fresh owned-server restarts for A1/B1/A2/B2. To switch control/candidate, first stop **only** the owned server, restore the exact saved source snapshot, confirm manifest/checksum, then restart with identical flags. Do not modify v10. Compare source SHA, model hash, benchmark checksum, installed packages and cache capacity; document any drift. For 4K, separately compare first-vs-first and second-vs-second repetitions, cache-hit/warmup conditions and reverse-order A/B controls where possible; a favorable second repetition alone is not enough.

Each full run must have 4 raw `SUCCESS` rows: two 256/256 request 4K repetitions and two 128/128 request 16K repetitions. Interrupted runs are incomplete evidence, even if their finished rows look good.

## 6 — Post-hoc read-only CSV evaluation

```bash
cd /root/Optimization/v11
python3 autoresearch/evaluate.py \
  --baseline-summary /actual/current_accepted_v11_control/benchmark_results/summary_DATE.csv \
  --baseline-raw /actual/current_accepted_v11_control/benchmark_results/raw_runs_DATE.csv \
  --starting-summary /actual/fresh_v11_starting_code_control/benchmark_results/summary_DATE.csv \
  --starting-raw /actual/fresh_v11_starting_code_control/benchmark_results/raw_runs_DATE.csv \
  --candidate-summary /actual/v11_candidate/benchmark_results/summary_DATE.csv \
  --candidate-raw /actual/v11_candidate/benchmark_results/raw_runs_DATE.csv \
  --json-out .autoresearch-work/comparison.json
```

Exit code 0 means only valid CSV structure, not confirmed improvement or accuracy. The helper reports numeric progress, fixed +5/+10 targets and matched uplift; independent repeats, production proof, no material two-case tradeoff and human MATH-500 are separately required.

## 7 — Copy/paste task to Codex Terra 6.1

> Operate only in /root/Optimization/v11 on the authorized Remote-SSH BI-V150 machine. Read AGENTS.md, program.md and autoresearch/EXPERIMENTS.md. v11 FlagGems/vLLM operator code starts identical to optimized v10; retain its stage-2 BLOCK_M=64 prefill and decode page lookup. Fixed v10 development half-run: 4K 2372.38, 16K 1872.70 Total tok/s. Minimum OR target: +5% = 4K >=2490.999 or 16K >=1966.335. Stretch OR target: +10% = 4K >=2609.618 or 16K >=2059.970. Require both frozen thresholds AND independently repeatable matched-v11-starting-code uplift of the same magnitude. Profile real v11 production phases before selecting a kernel target (mixed paged attention, 4K small-M GEMM/Linear, residual 16K KV pipeline); make genuine compute/data-movement changes and preserve framework dispatch/fallback logic. Never alter provided benchmark, model, dataset, generation settings, concurrency or serving flags for score; no quantization, speculative sampling, or latest FlagOS merge. Funnel through operator numerical correctness, paired kernel A/B, production invocation, repeated matched quick serving tests, and only then full unchanged 4K+16K half-runs. Use an operator-confirmed deadline and session_budget.py for every half-run; initial minimum is 150min for one or 275min for two, with a 25min reserved handoff. A1/B1/A2/B2 cannot fit a 330min session. Every run has RUNS=2, SKIP_FIRST=0, all requests successful. Compare fresh A/B and independent replications, record full raw provenance, TTFT/TPOT and regressions. Open draft PR for repeatable >=3% incremental progress; at >=5% confirmed open milestone PR and continue toward >=10% if budget/hypotheses permit; at independently confirmed >=10% open final PR and stop. Never auto-merge; human handles unchanged MATH-500 with project acceptance >=95%. Use checkpoint.py --push at least every 30min, immediately after high-value positive/negative results and before an expensive half-run; include hashes, raw evidence paths, exact reproduction steps, honest verification state and the single next action. Back up ignored raw data off the GPU host. Codex credits are not directly observable, so checkpoint preemptively. Track all attempts, rejected experiments, deadlines, remaining budget and exact PR links. Do not pretend this instruction launches an unattended process; report verified work only.
