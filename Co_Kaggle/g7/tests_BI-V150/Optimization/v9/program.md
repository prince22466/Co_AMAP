# MiniCPM5-2B BI-V150 v9 — Codex in Remote-SSH IDE

**Open /root/Optimization/v9 in the existing Remote-SSH Codex IDE extension.** It is the ONLY remote source workspace. Local repository mirror: Co_Kaggle/g7/tests_BI-V150/Optimization/v9 (Windows suffix: g7\tests_BI-V150\Optimization\v9).

Development **half-run** objective: **4K [4096,1024,64,256] >=2450** OR **16K [16384,1024,64,128] >=1450** Total tok/s. Test BOTH. The provided read-only benchmark is /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py (RUNS=2, SKIP_FIRST=0). Do not run MATH-500; the user will perform it later.

Read AGENTS.md and autoresearch/GOAL.md before beginning. No SSH bridge, rsync mirror, local Codex CLI, or bench.py wrapper.

## Terminal A: run vLLM from v9 (foreground)

Check port 9031 is available; never interrupt someone else's server. Keep this dedicated terminal running during measurements; stop YOUR server with Ctrl+C.

~~~bash
cd /root/Optimization/v9
mkdir -p .autoresearch-work
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:$PYTHONPATH"
export VLLM_PLUGINS=fl
unset VLLM_FL_USE_FLAGGEMS_ATTN VLLM_ATTENTION_BACKEND VLLM_BATCH_INVARIANT
set -o pipefail
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  2>&1 | tee .autoresearch-work/serve.log
~~~

The v9 source paths must be first in PYTHONPATH. Never start vLLM using the separate /workspace/vllm-plugin-FL tree. Don't kill unrelated processes with pkill/killall.

## Test the v9 kernels (another terminal)

~~~bash
cd /root/Optimization/v9
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:$PYTHONPATH"
export VLLM_PLUGINS=fl
python3 -m pytest -q -s --noconftest \
  vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py
~~~

## Terminal B: unchanged 4K/16K half-run

Run separately for each baseline and candidate. Each run writes its original CSVs and log in a unique v9-ignored directory.

~~~bash
cd /root/Optimization/v9
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

The provided benchmark may produce exit code 0 but omit a failed workload's summary. Always inspect raw CSVs and validate two successful runs per workload, complete requests, and both summary rows.

## Compare baseline/candidate (optional read-only checker)

Substitute actual filenames in the following example; all four paths must refer to the files from their respective run directories.

~~~bash
cd /root/Optimization/v9
python3 autoresearch/evaluate.py \
  --baseline-summary /actual/base/benchmark_results/summary_DATE.csv \
  --baseline-raw /actual/base/benchmark_results/raw_runs_DATE.csv \
  --candidate-summary /actual/candidate/benchmark_results/summary_DATE.csv \
  --candidate-raw /actual/candidate/benchmark_results/raw_runs_DATE.csv \
  --json-out .autoresearch-work/comparison.json
~~~

The checker exits 0 for valid CSVs, NOT for a verified gain. A one-off positive delta or a numerically crossed threshold is not a reproducible victory. Repeat paired/interleaved A/B, verify GPU kernel correctness and real production invocation, report TTFT and both throughputs. A repeatable true gain warrants an evidence-backed PR even below goal.

## Codex IDE task

    Work only in /root/Optimization/v9. Read AGENTS.md, program.md and
    autoresearch/GOAL.md. Profile and improve BI-V150 attention/GEMM kernels
    with one causal hypothesis at a time. Establish a fresh v9 half-run
    baseline with the fixed v9 vLLM command and unchanged /workspace half
    benchmark above. Run both 4K and 16K each time. Reach >=2450 4K OR
    >=1450 16K Total tok/s, verified by repeat tests. Run focused GPU
    correctness tests and prove the optimized kernel executes in serving.
    At EVERY repeatable serving gain, even below goal, commit only v9
    changes and open a PR recording measured evidence, then continue.
    Stop at a reproducible target, explicit deadline or hard blocker.
    Do not alter benchmark scripts, evaluation data, hyperparameters,
    non-v9 sources or normal dispatch. Do not run MATH-500; the user will.

Codex IDE is interactive and does not automatically provide an independent unattended daemon or global timeout. Specify a deadline if needed. Do not claim benchmarks or PRs without actual evidence.
