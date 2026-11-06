# MiniCPM5-2B BI-V150 v9 — Codex in Remote-SSH IDE

**Open /root/Optimization/v9 in the existing Remote-SSH Codex IDE extension.** It is the ONLY remote source workspace. Local repository mirror: Co_Kaggle/g7/tests_BI-V150/Optimization/v9 (Windows suffix: g7\tests_BI-V150\Optimization\v9).

Development **half-run** objective: **4K [4096,1024,64,256] >=2450** OR **16K [16384,1024,64,128] >=1450** Total tok/s. Test BOTH. Initial user-reported v9 half-run references: **4K 2130**, **16K 1240 tok/s** (second label assumed 16K; check CSV). **Observable progress = repeatable >=5% gain vs the last accepted best matched half-run**, initially corresponding to 4K >=2236.5 or 16K >=1302 if these reference values are confirmed. Do not confuse a baseline reference with fresh A/B evidence. The provided read-only benchmark is /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py (RUNS=2, SKIP_FIRST=0). Do not run MATH-500; the user will perform it later.

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

## Time planning and fast verification before the expensive half-run

One 4K+16K half-run may take **1.5–2 hours**; a baseline plus one candidate could take **3–4 hours**, and repeat confirmations add more. Do NOT run the 2-hour half benchmark for every small kernel tweak. Start by recording the exact session deadline, monotonic elapsed time, remaining GPU allocation and any already-valid v9 baseline evidence. Avoid rerunning a matching baseline merely out of habit, but never compare to unverified historical values.

**Candidate screening checklist (all required before expensive half-run):** (a) actual production bottleneck and plausible >=5% end-to-end opportunity; (b) original v9 attention/decode and modified-operator GPU correctness/fallback tests for 4K/16K paging and masks; (c) paired, warmed-up, repeated baseline/candidate kernel microbench with medians/variance, occupancy/register/shared-memory/spill and memory evidence; (d) proof the changed kernel is invoked in production; (e) paired short serving A/B smoke under identical diagnostic inputs, request completion and no obvious TTFT/TPOT regression; (f) written go/no-go justification and remaining-time check. Allocate roughly **20–30 minutes to quick screening** per idea, stopping bad candidates early. Diagnostic tests are not half-run scores.

Never start a new complete two-workload half-run with **less than 150 minutes left** (2h observed upper end + 30-minute results buffer); when a fresh control and confirmatory repeat are needed, plan for **~300 minutes or more**. These are planning safeguards, not promises about runtime. If the time left cannot support reliable evaluation, profile further or stop and report best verified result rather than starting an incomplete expensive run.

### Optional small serving A/B diagnostic — not an official benchmark

Use the same running v9 server and identical arguments for baseline and candidate. These commands call the installed vLLM CLI directly, do NOT edit or invoke a modified competition benchmark, and use deliberately fewer requests/tokens to give fast direction only. Save outputs under v9/.autoresearch-work. The diagnostic shape/concurrency differs from the scored benchmark: **NEVER report these tok/s as 4K/16K half-run throughput**.

~~~bash
cd /root/Optimization/v9
mkdir -p .autoresearch-work/quick
set -o pipefail

# 4K short prefill+decode diagnostic (not a 4K half-run).
vllm bench serve --backend vllm --model minicpm \
  --tokenizer /workspace/MiniCPM5-2B --endpoint /v1/completions \
  --host localhost --port 9031 --dataset-name random --ignore-eos \
  --random-input-len 4096 --random-output-len 128 \
  --max-concurrency 8 --num-prompts 16 \
  2>&1 | tee .autoresearch-work/quick/smoke_4k.log

# 16K long-context diagnostic (not a 16K half-run).
vllm bench serve --backend vllm --model minicpm \
  --tokenizer /workspace/MiniCPM5-2B --endpoint /v1/completions \
  --host localhost --port 9031 --dataset-name random --ignore-eos \
  --random-input-len 16384 --random-output-len 128 \
  --max-concurrency 8 --num-prompts 8 \
  2>&1 | tee .autoresearch-work/quick/smoke_16k.log
~~~

Use distinct log names/directories for **each** baseline/candidate diagnostic repetition; do not overwrite the evidence by reusing the example filenames. Reduced requests may not reproduce mixed prefill/decode at 64 concurrency, so add isolated/mixed attention kernel probes as needed before choosing an expensive serving experiment. Microbenchmarks and short smoke tests are screening filters, not proof of overall throughput or regression absence.

## Terminal B: unchanged 4K/16K half-run

**Expensive final-stage measurement; run ONLY after passing quick screening and verifying >=150 minutes remain.** Run separately for a needed baseline and a promising candidate. Each run writes its original CSVs and log in a unique v9-ignored directory.

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

The checker exits 0 for valid CSVs, NOT for a verified gain. A one-off positive delta or a numerically crossed threshold is not a reproducible victory. Repeat paired/interleaved A/B, verify GPU kernel correctness and real production invocation, report TTFT and both throughputs. An observed positive change below 5% is diagnostic and should be logged, not filed as a progress PR. A **repeatable >=5% gain versus the last accepted best** in either workload warrants an evidence-backed progress PR even below goal. A reproducible final goal (4K >=2450 OR 16K >=1450) warrants a final PR even if its last incremental gain is below 5%.

## Codex IDE task

    Work only in /root/Optimization/v9. Read AGENTS.md, program.md and
    autoresearch/GOAL.md. Profile and improve BI-V150 attention/GEMM kernels
    with one causal hypothesis at a time. Establish a fresh v9 half-run
    baseline only if existing matched CSVs cannot be verified. Spend ~20–30
    minutes per hypothesis on correctness, profiled paired kernel A/B,
    production invocation proof and short serving diagnostics. Do not run
    an expensive 1.5–2h half-run without strong causal evidence and at least
    150 minutes remaining. Run BOTH workloads in the official half-run.
    Reach >=2450 4K OR
    >=1450 16K Total tok/s, verified by repeat tests. Run focused GPU
    correctness tests and prove the optimized kernel executes in serving.
    At EVERY repeatable >=5% Total tok/s gain relative to the last
    accepted best half-run in either workload, commit only v9 changes
    and open an evidence-backed PR, then continue. Log smaller gains
    until cumulative >=5% is proven. If a reproducible end goal is
    reached, open the final PR even for a smaller last step.
    Stop at a reproducible target, explicit deadline or hard blocker.
    Do not alter benchmark scripts, evaluation data, hyperparameters,
    non-v9 sources or normal dispatch. Do not run MATH-500; the user will.

Codex IDE is interactive and does not automatically provide an independent unattended daemon or global timeout. Specify a deadline if needed. Do not claim benchmarks or PRs without actual evidence.
