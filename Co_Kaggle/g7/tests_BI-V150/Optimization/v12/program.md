# v12 Codex runbook — BI-V150 Remote-SSH

**Read `AGENTS.md` first.** Its constraints and stop/go rules are authoritative. Candidate queue: `autoresearch/RESEARCH_QUEUE.md`; results: `autoresearch/EXPERIMENTS.md`. Work only in `/root/Optimization/v12`.

## 1. Verify source and allocation before research

```bash
cd /root/Optimization/v12
pwd
git status --short
git branch --show-current
git rev-parse HEAD
sha256sum /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py
sha256sum vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
python3 -m unittest discover -s autoresearch/tests -v
```

If the branch or instructions are missing, fetch the **draft PR branch** safely, preserving local changes. Confirm model, installed plugin/Triton versions, actual imported Python module paths and GPU. Do not kill another user's server. Ask the operator for the *actual* GPU allocation end time; initialize the budget guard **once**:

```bash
python3 autoresearch/session_budget.py init --deadline-utc 'ACTUAL-ISO8601-UTC-DEADLINE'
python3 autoresearch/session_budget.py status
```

Replace the placeholder with the confirmed UTC time. Never invent/reset a deadline to bypass admission. First allocation: use **cheap profiling and kernel tests before spending ~2h on a full half-run**. The fresh dense control half-run is mandatory only when a candidate reaches the half-run stage.

Checkpoint regularly (on a non-main PR branch; the helper commits *evidence only*):

```bash
python3 autoresearch/checkpoint.py \
  --summary 'Candidate / observed result / decision' \
  --evidence 'HEAD, source hashes, actual GPU logs/CSV and sample counts' \
  --next-step 'Specific next test or reason to stop' \
  --status provisional --push
```

Save raw traces off-instance and **commit/push accepted kernel code separately**. The checkpoint helper never commits your candidate source.

## 2. Run original GPU regression tests

```bash
cd /root/Optimization/v12
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
python3 -m pytest -q -s --noconftest \
  vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_mixed_paging.py
```

Capture correctness and import-path proof before optimizing. For approximate attention, additionally implement a sparse PyTorch reference, check dense-output differences, and test nonlocal middle-context retrieval: start with q=2048 after 14,336-token prefix, paged KV=16, BF16 GQA=16Q/2KV/128 head. Include mixed requests, nonaligned tails and invalid block-table entries.

## 3. Profile, experiment, and screen

Measure real 4K and 16K **uncached prefill, mixed prefill/decode and pure decode**, and prefix-cache warm/cold behavior. Report attention GPU *summed-kernel* share separately from *wall-clock* share and end-to-end throughput. Re-rank hypotheses from `autoresearch/RESEARCH_QUEUE.md`.

Try **one exact** operator dataflow/pipeline hypothesis and **one sparse** local/global or query-aware selection hypothesis in the first research cycle (unless evidence rules one out). Keep independent reversible diffs. For each: correct dense or explicit sparse reference, alternating warm kernel A/B, source/loaded-module proof, tile/byte counts, short serving 4K and 16K A/B. Stop quickly if invalid, loses long-range capability or has weak critical-path benefit.

Owned server launch; **same flags for baseline and candidate**:

```bash
cd /root/Optimization/v12
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
```

Confirm `curl -f http://127.0.0.1:9031/health` and actual v12 kernel invocation. Short tests with the owned server:

```bash
mkdir -p .autoresearch-work/quick
quick_dir=$(mktemp -d "$PWD/.autoresearch-work/quick/screen_XXXXXXXX")
for input_len in 4096 16384; do
  if [ "$input_len" = 4096 ]; then n=16; else n=8; fi
  vllm bench serve --backend vllm --model minicpm \
    --tokenizer /workspace/MiniCPM5-2B --endpoint /v1/completions \
    --host localhost --port 9031 --dataset-name random --ignore-eos \
    --random-input-len "$input_len" --random-output-len 128 \
    --max-concurrency 8 --num-prompts "$n" \
    2>&1 | tee "$quick_dir/quick_${input_len}.log"
done
```

These are **non-scoring diagnostics** at concurrency 8/128 output tokens, NOT valid evidence for official 64-concurrency performance. Alternate A/B restarts and retain all logs.

## 4. Full half-run only after go/no-go, never change benchmark

The official **development half-run** is external and unchanged: `RUNS=2`, `SKIP_FIRST=0`, cases `[[4096,1024,64,256],[16384,1024,64,128]]`. No benchmark copying, patching or tuning. It usually takes ~90–120 minutes; hard 90-minute floor but budget guard's **>=150-minute conservative admission** applies.

```bash
cd /root/Optimization/v12
python3 autoresearch/session_budget.py status
python3 autoresearch/session_budget.py admit --runs 1
# After admission, owned healthy server, and pre-run PR checkpoint:
python3 autoresearch/session_budget.py run-half --push-checkpoint
```

The guard invokes the untouched external benchmark and saves unique raw results under `.autoresearch-work/bench_runs/`. Preserve 2 complete SUCCESS repetitions for **each** workload with 256 and 128 requests respectively. Check source hashes and prefix cache states. A zero exit is not validation or proof of a gain.

Repeat matched original-v12 control A and candidate B on identical GPU/server flags (alternating server restarts/order across allocations as needed). Use the read-only evaluator with exactly **two** evidence pairs:

```bash
python3 autoresearch/evaluate.py \
 --control-summary /real/A/summary.csv --control-raw /real/A/raw_runs.csv \
 --candidate-summary /real/B/summary.csv --candidate-raw /real/B/raw_runs.csv \
 --json-out .autoresearch-work/comparison.json
```

It checks CSV integrity and numeric thresholds against the reported v10 anchor and matched control, **not** causal replication, quality or rules. For subsequent incremental improvements use the then-current accepted, freshly rerun control A; also show cumulative change from initial v12.

## 5. Quality, PRs, and handoff

Exact dense-output optimizations: ensure existing reference tests and appropriate full-model smoke. Sparse algorithms: compare against dense on multiple-seed 4K/8K/16K retrieval with randomly placed facts in **beginning/middle/end**, distractors and reasoning. Reject measurable core-ability regression. A human runs the **unchanged official MATH-500 Level-3** test for a selected candidate; >=95% is necessary but is **not sufficient** if outputs become incomparable. Ask organizers before treating approximate attention as contest-eligible. Never quietly enable sparse via serving flag adjustments.

Commit minimal tested operator/test changes to the non-main branch and open evidence-backed draft progress PR for independently repeatable >=3% gains, milestone PR for >=5%, stretch >=10%. No auto-merge. Keep `autoresearch/EXPERIMENTS.md` append-only; update `HANDOFF.md` with best verified SHA, raw artifacts/hashes, negative results, GPU time and next step.

## Paste-ready Codex task

> Work as a GPU inference-kernel autoresearch engineer on the authorized Remote-SSH BI-V150 machine under /root/Optimization/v12 only. Read AGENTS.md, program.md, RESEARCH_QUEUE.md, EXPERIMENTS.md and HANDOFF.md. First verify branch, v12 imports, baseline operator SHA and actual GPU deadline, test the guard/attention suite, and profile 4K/16K production prefill, mixed and decode attention. Aim for reproducible >=5% total tok/s gain in either unchanged half-run case (ideally >=10%) with no material regression or MiniCPM long-context loss. Investigate exact FlashAttention-style operator/dataflow, split-K/Flash-Decoding and KV staging, and screen one gated sparse long-range selection candidate. Ring Attention only if real multi-GPU use exists. Do not use dispatch-only swaps, benchmark edits, quantization/speculation via hyperparameters, other model weights, or latest FlagOS merge. Sparse tests must prove actual KV tile/FLOP elimination and assess *middle-context retrieval versus dense*. Use correctness -> matched kernel A/B -> matched short-serving A/B -> quality/rules gate -> unchanged guarded half-run, not the other way around. Spend no half-run time without a plausible serving win. Checkpoint/push every <=30min; separately commit tested source code, save ignored raw traces off-instance, and never claim unrun results. Human owns MATH-500; approximate submission requires separate eligibility review. Continue within verified GPU/credit budget, preserve a candid handoff at deadline, and do not auto-merge.
