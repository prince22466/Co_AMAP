# v12 — Codex autoresearch execution program (Remote-SSH BI-V150)

**Authority:** read [AGENTS.md](AGENTS.md), [autoresearch/RESEARCH_QUEUE.md](autoresearch/RESEARCH_QUEUE.md), [autoresearch/EXPERIMENTS.md](autoresearch/EXPERIMENTS.md) first. Work inside **/root/Optimization/v12** only. This PR configures research but DOES NOT remotely launch Codex or a GPU.

## Goal and evidence

- Fixed reported v10 reference: **4K 2372.38** / **16K 1872.70** Total tok/s, separate from fresh matched v12 starting-code control.
- Minimum (OR): **4K >=2490.999** or **16K >=1966.335** (+5%). Stretch (OR): **4K >=2609.618** or **16K >=2059.970** (+10%). Must also demonstrate independent matched >=5% or >=10% uplift over original v12 on same hardware; other workload reported and no unapproved material regression.
- 4K case `[4096,1024,64,256]`, 16K case `[16384,1024,64,128]` with `RUNS=2 SKIP_FIRST=0` in the untouched external half-run script. These are **development**, not official full-run benchmarks.
- Preserve the existing dense model exactly where possible. Approximate sparse attention must pass long-context tests and be ruled eligible by organizer before it can be considered a submission, regardless of MATH-500 >=95%.

## 0. Resolve the actual workspace and provenance

```bash
cd /root/Optimization/v12
pwd
git status --short
git rev-parse HEAD
git branch --show-current
git remote -v
sha256sum /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py
sha256sum vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py
sha256sum vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150.py
```

Make sure this checkout includes **v12 AGENTS.md and autoresearch tools from the open draft PR**. If the remote still uses main's stale v10-only instructions, fetch/checkout the PR branch without discarding untracked user work. Check model path/weight hashes, package versions, memory, free port, BI-V150 information and exactly which Python modules are loaded. Do not assume every `/root/Optimization/v12` is a Git repository checkout. In that case, resolve the correct checked-out repository and content path first, without overwriting user changes or running the wrong code.

## 1. Allocation guard and durable checkpoints

Obtain the actual GPU end time from operator. For an explicitly confirmed 330-minute allocation only:

```bash
cd /root/Optimization/v12
python3 autoresearch/session_budget.py init --duration-minutes 330
python3 autoresearch/session_budget.py status
python3 autoresearch/session_budget.py admit --runs 1
```

Alternatively `init --deadline-utc <ACTUAL-UTC-EXPIRY>`; never fabricate it, reset it mid-session, or initialize if unknown. Conservative initial estimate: **>=150 min for one** complete half-run; **>=275 min for two**, with 25 minutes reserved for preserving evidence. A four-run A1/B1/A2/B2 half-run series needs around **525 min** under initial assumptions and generally spans allocations. A partial run is never a completed result.

At startup, every <=30min, before long GPU runs, after meaningful findings and at time/credit pressure, checkpoint on the **intended non-main PR branch**:

```bash
python3 autoresearch/checkpoint.py \
  --summary "Exact tested operator, finding, success/failure, scope" \
  --evidence "HEAD and source SHA256; GPU timings; raw files and checksums; observed errors" \
  --next-step "One next concrete test; deadline/ownership" \
  --status provisional --push
```

`--push` commits only tracked ledger/handoff (and optional small raw CSV evidence), not unreviewed inference edits; verify GitHub shows the commit. Back up bulk ignored raw files to durable off-instance storage. Stop new long tests before preservation reserve.

## 2. Start the owned unmodified v12 serving configuration

Terminal A, **only when the operator owns port 9031 and model is present**:

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

In Terminal B, check `curl -f http://127.0.0.1:9031/health`; record real module source paths and production GPU trace before claiming any candidate path runs. Never terminate an unrelated server or alter the official serving flags.

## 3. Initial correctness, control and research screens

```bash
cd /root/Optimization/v12
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl

python3 -m unittest discover -s autoresearch/tests -v
python3 -m pytest -q -s --noconftest \
  vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_mixed_paging.py
```

Then profile representative **actual production** phases: uncached 4K and 16K prefill, mixed prefill/decode with shuffled paged KV, pure decode, cache-warm and cache-cold. Distinguish GPU-summed time, traced wall time, preemption, active/idle, graph replay visibility, prefix-cache hits, dynamic `max_seqlen_q` and Q chunks. Record resource counters (registers, SMEM, spills, K/V traffic) and estimated Amdahl gains.

**Required first exploration batch:** do not spend the entire session on one fashionably named technique. Screen at least (a) one exact FlashAttention-2-style work-partition/online-softmax or paged-KV pipeline hypothesis, and (b) one intentionally approximate local+sink or query-aware page-selection hypothesis with dense-output comparison and a synthetic middle-context retrieval sanity check. Keep them as separate reversible diffs. If time is insufficient for both, checkpoint the priority and untested second hypothesis explicitly.

Test sparse with an **explicit sparse reference**, page-tail and causal masking, 16-token page shuffling, first/middle/last token position, nonzero existing prefix (e.g. q=2048 following ctx=14336), mixed batch, warm/cache effects. A low-cost sparse-screen is not a green light to use sparse in scoring.

Review the old research-only [PR #237](https://github.com/prince22466/Co_AMAP/pull/237) for ideas **only**. The sparse prototype is not in the main v12 tree. It was never confirmed compiled on BI-V150, and its Python entrypoint is default-off. Inspect for bounds, causal leaks, sink/recent tile overlap, and 2D/3D/mixed-batch cases before porting an **individual real operator change**, never copying the whole PR blindly.

## 4. Interleaved kernel A/B and short serving go/no-go

Run exact operator numerical comparisons first, then warmed alternating A/B on identical real shapes/layouts. Track kernel latency median/p10/p90, load bytes, visited KV tiles, spills, occupancy, compiler options, CPU overhead of page-selection/extra reductions, and all NaNs. Default density and feature fallbacks must be unchanged when a candidate is disabled.

Short serving *diagnostics only* (same settings on fresh A and B owned restarts; unique logs):

```bash
cd /root/Optimization/v12
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

These are standalone diagnostics with shorter decoding and lower concurrency, **not** the official scoring workloads and must never be used to claim 4K/16K half-run gains. Log A/B order and seed and repeat control to detect warmup or compilation drift.

## 5. Go to expensive half-run only on independently convincing serving evidence

Verify remaining time:

```bash
cd /root/Optimization/v12
python3 autoresearch/session_budget.py status
python3 autoresearch/session_budget.py admit --runs 1
# Only after the guard admits, correct v12 server is healthy and a PR checkpoint is pushed:
python3 autoresearch/session_budget.py run-half --push-checkpoint
```

The guard invokes **the external unchanged** `/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py` with exactly `[[4096,1024,64,256],[16384,1024,64,128]]`, controls available time and checkpoints outputs. It writes unique directories in ignored `.autoresearch-work/bench_runs/`. After it exits, inspect original raw CSV **four SUCCESS rows** (two x 256/256 requests for 4K and two x 128/128 for 16K) and summary file. A zero exit from the runner or evaluator does not prove correctness, serving win, accuracy or independent repeatability.

For A/B, restore exact dense-source SHA as control; restart an owned server with the same flags, workload cache conditions and K/V capacity; run candidate B, then independent control/candidate pairs in alternated order when future allocation allows. Keep full source/benchmark/model hashes and raw evidence in PR or off-instance backups.

## 6. Read-only numeric evaluation

```bash
cd /root/Optimization/v12
python3 autoresearch/evaluate.py \
 --baseline-summary /actual/last_accepted_v12_control/benchmark_results/summary_DATE.csv \
 --baseline-raw /actual/last_accepted_v12_control/benchmark_results/raw_runs_DATE.csv \
 --starting-summary /actual/fresh_v12_original_code/benchmark_results/summary_DATE.csv \
 --starting-raw /actual/fresh_v12_original_code/benchmark_results/raw_runs_DATE.csv \
 --candidate-summary /actual/v12_candidate/benchmark_results/summary_DATE.csv \
 --candidate-raw /actual/v12_candidate/benchmark_results/raw_runs_DATE.csv \
 --json-out .autoresearch-work/comparison.json
```

This read-only helper checks CSV schema and numeric thresholds against fixed v10 anchors and matched v12 sources. It does NOT validate output semantics, long-context retrieval, organizer eligibility, causal uplift, or MATH-500. Human owns unchanged MATH-500 (project >=95%) for selected quality-safe candidate; seek organizer ruling for approximate attention.

## 7. Paste-ready Codex Terra task

> Act as an evidence-driven GPU kernel autoresearch engineer inside the authorized /root/Optimization/v12 Remote-SSH BI-V150 workspace. Read AGENTS.md, program.md, autoresearch/RESEARCH_QUEUE.md, EXPERIMENTS.md and HANDOFF.md. You must not touch v0–v11 or competition benchmark code, alter vLLM serve flags, enable quantization/speculation, bypass major dispatch logic or copy upstream framework code. Original v12 kernels are v10-equivalent; the target is a repeatable >=5% end-to-end 4K OR 16K half-run gain (stretch >=10%) versus fixed v10 anchors AND independently matched original-v12 controls, while other workload and MiniCPM accuracy/capabilities remain acceptable. Establish GPU deadline and checkpoint PR branch within 30 min, before half-runs and at credit pressure; push append-only evidence and small raw CSVs, back up large traces off-instance. Profile true v12 prefill/decode/mixed bottlenecks and Amdahl. Explore exact FlashAttention-2-inspired operator internals and split-K/paged-KV pipeline first, but also perform an early focused block-sparse/query-aware screen with long-range middle retrieval, retaining an accurate dense reference. Ring Attention only if verified multi-device and worthwhile. One change at a time, numerical tests, interleaved kernel A/B, production trace, short matched serving A/B; no promoted candidate from isolated kernel speed. Sparse output is intentionally approximate and needs long-context proof, unchanged MATH-500 >=95% (human) and organizer eligibility before submission. Use session_budget.py for every unchanged two-case half-run. Reproducible >=3% progress => draft evidence PR, >=5% => milestone PR, >=10% => stretch PR, no auto-merge. Continue within confirmed time/credit budget; on deadline or blocker publish final verified results, exact source hashes, rejected hypotheses and next action in HANDOFF.md. Do not claim that these instructions themselves launched any GPU job.
