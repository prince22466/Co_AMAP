# BI-V150 v10 — Codex in Remote-SSH IDE

Use only `/root/Optimization/v10` for source edits, serving and data. Read [AGENTS.md](AGENTS.md) and [autoresearch/EXPERIMENTS.md](autoresearch/EXPERIMENTS.md). This v10 tree incorporates v9's stage-2 optimized prefill attention path; do not restart from pre-v9 attention by accident.

**Fixed +6% of reported v9 HALF-run target:** 4K >= **2391.6886** (display 2391.69) OR 16K >= **1609.1436** (display 1609.15) Total tok/s. Baselines were 2256.31 / 1518.06. Always measure BOTH exact cases; a fresh matched v10 control is required for claims of a causal kernel win. Repeatable >=5% vs last accepted matched v10 control -> progress PR; repeatable fixed +6% goal **with positive matched-v10 uplift and no unapproved net-negative tradeoff** -> final PR then stop. MATH-500 remains for the human.

## Terminal A — dedicated v10 server (same fixed serving configuration as v9)

Confirm port 9031 is free; do not stop another person's server. Run the server only from v10 and keep it attached to an owned terminal or a dedicated inspectable tmux session (do not let an SSH reset silently invalidate benchmark completion). Avoid overwriting previous logs.

~~~bash
cd /root/Optimization/v10
mkdir -p .autoresearch-work/serve_runs
serve_log=$(mktemp "$PWD/.autoresearch-work/serve_runs/serve_XXXXXXXX.log")
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
unset VLLM_FL_USE_FLAGGEMS_ATTN VLLM_ATTENTION_BACKEND VLLM_BATCH_INVARIANT
set -o pipefail
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  2>&1 | tee "$serve_log"
~~~

The server must import v10 `FlagGems/src` and `vllm-plugin-FL` (verify actual import paths and GPU trace). The external `/workspace/vllm-plugin-FL` tree is **not** the serving source. Restart only YOUR owned server with Ctrl+C; never `pkill` unrelated processes.

## GPU regression tests

~~~bash
cd /root/Optimization/v10
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
python3 -m pytest -q -s --noconftest \
  vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_mixed_paging.py

# CSV comparison helper tests: no GPU needed.
python3 -m unittest discover -s autoresearch/tests -v
~~~

If `mm.py` or `linear.py` is modified, also add/run focused real-shape BF16/FP16 MM/Linear GPU correctness and fallback tests. Include small M, edge K/N, real v10 projection shapes and accumulated error tolerances. Record actual PASS output; these setup docs do not claim tests have been run.

## Profile and screen BEFORE expensive half-runs

1. Measure fresh v10 production traces for uncached/mixed/decode 4K and 16K, plus KV-cache active/wait/preempt metrics. Distinguish GPU summed kernel duration from wall time and 4K prefix-cache reuse. If a 4K small-M MM bottleneck does NOT reproduce, re-rank rather than forcing that direction.
2. Isolate one legitimate operator/data-movement edit, with reference-output tests and warmed, interleaved baseline/candidate kernel A/B; record resource counters, compiler behavior, and production invocation.
3. Run short, matched, standalone serving diagnostics for 4K and 16K. These are screening probes, not benchmark results.

Optional diagnostic examples (run with a healthy v10 server; preserve distinct outputs for each A/B repetition):

~~~bash
cd /root/Optimization/v10
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
~~~

These reduced samples do NOT match 64-concurrency full-length decode, can be distorted by prefix caching, and must never be reported as scored 4K/16K half-run tok/s. Use fresh named logs per repeat, preserve matched order and server restarts, and create targeted mixed-batch probes if needed.

## Run unchanged 4K + 16K half-run ONLY after a documented go decision

Historical runtime is 90–120 minutes for both cases. Record the actual allocation/deadline first. **Do not start if remaining time <90 minutes**; 90–119 minutes still risks incomplete runs. Prefer ~300+ minutes for baseline plus repeat(s). If v9 original CSV provenance is missing, measure a fresh v10 starting-code control A (same stage-2 kernel) rather than assuming a reported summary proves matched A/B.

Do not modify, copy, patch, wrap for score optimization, or replace `benchmark_throughput_serve_half.py`. It is installed externally and must remain `RUNS=2`, `SKIP_FIRST=0`.

~~~bash
cd /root/Optimization/v10
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
printf 'Half-run outputs: %s\n' "$run_dir/benchmark_results"
~~~

For SSH reliability, an **owned, attachable tmux session** may host the same unmodified command and its standard logs; only reconnect/inspect it, not change any benchmark inputs. Verify all four raw rows show SUCCESS and full successful-request counts (256 for each 4K run; 128 for each 16K run), with both summary rows. Failed/incomplete work is invalid even if the script exits 0.

## Evaluate raw/summary CSVs (diagnostic, never an acceptance certificate)

~~~bash
cd /root/Optimization/v10
python3 autoresearch/evaluate.py \
  --baseline-summary /actual/v10_control/benchmark_results/summary_DATE.csv \
  --baseline-raw /actual/v10_control/benchmark_results/raw_runs_DATE.csv \
  --candidate-summary /actual/v10_candidate/benchmark_results/summary_DATE.csv \
  --candidate-raw /actual/v10_candidate/benchmark_results/raw_runs_DATE.csv \
  --json-out .autoresearch-work/comparison.json
~~~

The validator compares candidate to the current matched v10 control for the >=5% intermediate checkpoint and to the FIXED reported v9 numbers for the +6% final target. Exceeding the fixed target with zero/negative improvement over the matched control is NOT a promotion candidate. Exit 0 means only valid CSV structure/completion, not repeatable improvement. Repeat interleaved A/B under matched conditions (especially 4K cache-sensitive runs), require kernel correctness and production trace, document both workloads and TTFT/TPOT, then prepare PR.

## Copy/paste task for Codex

> Work **only** in /root/Optimization/v10. Read AGENTS.md, program.md and autoresearch/EXPERIMENTS.md. Keep inherited v9 stage-2 attention intact. Goal: +6% over reported v9 half-run: 4K >=2391.6886 OR 16K >=1609.1436 Total tok/s, both unchanged cases tested, valid requests, independently repeatable and correct. First profile current v10 4K decode small-M GEMM/Linear, mixed prefill/decode, and 16K prefill kernel/data-movement time shares; re-rank on production evidence. Only genuine operator/data-movement changes; no dispatch-only shortcuts, quantization, speculative decoding, hyperparameter changes, benchmark edits or latest FlagOS merge. Use correctness, paired kernel A/B, production invocation and short serving diagnostic gates before a costly half-run. A half-run requires >=90 min remaining; reserve more for matched repeat. Open evidence-backed PR at every repeatable >=5% incremental checkpoint, continue, and open a final PR once fixed +6% goal and matched uplift are confirmed. Never auto-merge. Do not run MATH-500. Do not accept a materially worse two-case competition result merely because one goal threshold was crossed. At deadline/blocker record best verified results and stop. No unattended execution is presumed.
