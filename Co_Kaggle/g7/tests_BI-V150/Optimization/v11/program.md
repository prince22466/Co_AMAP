# v11 — Codex Terra 6.1 autoresearch runbook (Remote-SSH)

Read [AGENTS.md](AGENTS.md), [autoresearch/EXPERIMENTS.md](autoresearch/EXPERIMENTS.md), and [readme.md](readme.md). Work **only in `/root/Optimization/v11`**. This repository PR configures a task; it does not remotely launch Codex or GPU work. Starting operator sources (FlagGems + vLLM plugin) must remain the **v10 optimized checkpoint**, including BF16 GQA prefill `BLOCK_M=64`, until an evidence-backed v11 operator experiment changes them.

## Targets and acceptance

Official-format **development half-run** (not full-run), both `[4096,1024,64,256]` and `[16384,1024,64,128]`, unmodified benchmark:
- Frozen reported v10 reference: **4K 2372.38** / **16K 1872.70** Total tok/s.
- **Minimum +5% (OR):** 4K >= **2490.999**, or 16K >= **1966.335**.
- **Stretch +10% (OR):** 4K >= **2609.618**, or 16K >= **2059.970**.

Preserve raw precision in comparisons. Both cases must pass all requests, and a fresh independent matched v11 starting-code control must establish >=5% causal uplift for minimum or >=10% for stretch. Repeated independent A/B pairs are required. Disclose the other workload and refuse unapproved material regression. At repeatable >=3% incremental improvement vs last accepted best, open a *draft progress PR*, then keep searching. At +5% milestone open a PR but keep searching for +10% if feasible. At independently confirmed +10% open final PR, stop. **No auto-merge.** Human owns MATH-500.

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

## 5 — Unmodified official-format development half-run (Terminal B)

A complete two-case half-run historically takes **90–120 minutes**. **Never start below 90 minutes remaining**; 90 minutes may be insufficient to complete. Prefer ~300+ minutes for fresh control + candidate + validation. If deadline cannot be verified, state it as unknown and do not guarantee the run. The external benchmark must be unmodified with `RUNS=2`, `SKIP_FIRST=0`.

```bash
curl -f http://127.0.0.1:9031/health
cd /root/Optimization/v11
mkdir -p .autoresearch-work/bench_runs
run_dir=$(mktemp -d "$PWD/.autoresearch-work/bench_runs/run_XXXXXXXX")
cd "$run_dir"
sha256sum /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py > benchmark_sha256.txt
set -o pipefail
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]' \
  2>&1 | tee benchmark.log
printf 'Outputs: %s\n' "$run_dir/benchmark_results"
```

Use new run directories and fresh owned-server restarts for A1/B1/A2/B2. To switch control/candidate, first stop **only** the owned server, restore the exact saved source snapshot, confirm manifest/checksum, then restart with identical flags. Do not modify v10. Compare source SHA, model hash, benchmark checksum, installed packages and cache capacity; document any drift, and never conceal 4K first/second-run variation.

Each full run must have 4 raw `SUCCESS` rows: two 256/256 request 4K repetitions and two 128/128 request 16K repetitions. Interrupted runs are incomplete evidence, even if their finished rows look good.

## 6 — Post-hoc read-only CSV evaluation

```bash
cd /root/Optimization/v11
python3 autoresearch/evaluate.py \
  --baseline-summary /actual/fresh_v11_control/benchmark_results/summary_DATE.csv \
  --baseline-raw /actual/fresh_v11_control/benchmark_results/raw_runs_DATE.csv \
  --candidate-summary /actual/v11_candidate/benchmark_results/summary_DATE.csv \
  --candidate-raw /actual/v11_candidate/benchmark_results/raw_runs_DATE.csv \
  --json-out .autoresearch-work/comparison.json
```

Exit code 0 means only valid CSV structure, not confirmed improvement or accuracy. The helper reports numeric progress, fixed +5/+10 targets and matched uplift; independent repeats, production proof, no material two-case tradeoff and human MATH-500 are separately required.

## 7 — Copy/paste task to Codex Terra 6.1

> Operate only in /root/Optimization/v11 on the authorized Remote-SSH BI-V150 machine. Read AGENTS.md, program.md and autoresearch/EXPERIMENTS.md. v11 FlagGems/vLLM operator code starts identical to optimized v10; retain its stage-2 BLOCK_M=64 prefill and decode page lookup. Fixed v10 development half-run: 4K 2372.38, 16K 1872.70 Total tok/s. Minimum OR target: +5% = 4K >=2490.999 or 16K >=1966.335. Stretch OR target: +10% = 4K >=2609.618 or 16K >=2059.970. Require both frozen thresholds AND independently repeatable matched-v11-starting-code uplift of the same magnitude. Profile real v11 production phases before selecting a kernel target (mixed paged attention, 4K small-M GEMM/Linear, residual 16K KV pipeline); make genuine compute/data-movement changes and preserve framework dispatch/fallback logic. Never alter provided benchmark, model, dataset, generation settings, concurrency or serving flags for score; no quantization, speculative sampling, or latest FlagOS merge. Funnel through operator numerical correctness, paired kernel A/B, production invocation, repeated matched quick serving tests, and only then full unchanged 4K+16K half-runs. Half-run start floor >=90min remaining (90–119min still risky). Every run has RUNS=2, SKIP_FIRST=0, all requests successful. Compare fresh A/B and independent replications, record full raw provenance, TTFT/TPOT and regressions. Open draft PR for repeatable >=3% incremental progress; at >=5% confirmed open milestone PR and continue toward >=10% if budget/hypotheses permit; at independently confirmed >=10% open final PR and stop. Never auto-merge; human handles MATH-500. Track all attempts, rejected experiments, deadlines, remaining budget and exact PR links. Do not pretend this instruction launches an unattended process; report verified work only.
