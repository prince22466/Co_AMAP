# v16 Remote-SSH runbook

Read AGENTS.md and autoresearch/RESEARCH_QUEUE.md first. Human-started Codex Terra 6.1 task. Use actual v16 checkout path (not an exported folder under unrelated Git root). No work on v14/v15 or main.

## A. Verify source, runtime and budget
```bash
cd /root/Optimization/v16  # or actual nested v16 path
git rev-parse --show-toplevel
git ls-files --error-unmatch FlagGems/src/flag_gems/__init__.py vllm-plugin-FL/vllm_fl/__init__.py
git status --short
git branch --show-current
git rev-parse HEAD
prefix=$(git rev-parse --show-prefix)
git rev-parse "HEAD:${prefix}FlagGems" "HEAD:${prefix}vllm-plugin-FL"
sha256sum /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py
python3 -m unittest discover -s autoresearch/tests -v
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL"
export VLLM_PLUGINS=fl
ixsmi
```
Match tree hashes in AGENTS.md; pin original SHA. Verify worker imports and execution from v16, not just shell imports. Record runtime/model/config, GPU, flags and script hash. Read external half-run script to confirm RUNS=2, SKIP_FIRST=0, CLI and output path; if not matching, stop, never edit. Initialize once with actual operator-confirmed UTC deadline (placeholder below); never reset same allocation. Archive prior state only on confirmed new allocation.
```bash
python3 autoresearch/session_budget.py init --deadline-utc 'ACTUAL-ISO8601-UTC-DEADLINE'
python3 autoresearch/session_budget.py status
```

## B. Research and quick screens
Apply novelty gate, prioritize actual 4K critical path and test new operator correctness/reference, fallback, graph replay and worker dispatch. Existing attention tests for changed/affected paths: test_biv150_attention.py, test_biv150_decode.py, test_biv150_mixed_paging.py under vllm-plugin-FL/tests/functional_tests; relevant FlagGems tests for changed ops. Do not rerun historical suites indiscriminately. Reuse v15 diagnostic commands in [runbook](../v15/program.md) as reference only, replacing all checkout/output paths with v16. Do not invoke v15 code or helpers. Fixed server:
```bash
cd /root/Optimization/v16
mkdir -p .autoresearch-work/serve_runs
serve_log=$(mktemp "$PWD/.autoresearch-work/serve_runs/serve_XXXXXXXX.log")
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL"
export VLLM_PLUGINS=fl
unset VLLM_FL_USE_FLAGGEMS_ATTN VLLM_ATTENTION_BACKEND VLLM_BATCH_INVARIANT
set -o pipefail
vllm serve /workspace/MiniCPM5-2B \
 --port 9031 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
 --served-model-name minicpm --gpu-memory-utilization 0.85 \
 --max-model-len 131072 2>&1 | tee "$serve_log"
```
Fresh owned server per matched sequence; record PID, readiness, worker paths, cache and metrics. Never kill unrelated processes. Short diagnostics (not official): same seed/order, 4K and 16K, 8 then 64 prompts/concurrency as needed. Use 1024-output pressure diagnostic only if needed for decode/preemption, matched both sides. Remove instrumentation for timing.
```bash
set -euo pipefail
cd /root/Optimization/v16
curl -fsS http://127.0.0.1:9031/health
mkdir -p .autoresearch-work/quick
quick_dir=$(mktemp -d "$PWD/.autoresearch-work/quick/screen_XXXXXXXX")
for input_len in 4096 16384; do
  curl -fsS http://127.0.0.1:9031/metrics > "$quick_dir/${input_len}_before.prom"
  vllm bench serve --backend vllm --model minicpm \
    --tokenizer /workspace/MiniCPM5-2B --endpoint /v1/completions \
    --host localhost --port 9031 --dataset-name random --ignore-eos \
    --seed 42 --random-input-len "$input_len" --random-output-len 128 \
    --max-concurrency 64 --num-prompts 64 \
    2>&1 | tee "$quick_dir/${input_len}.log"
  curl -fsS http://127.0.0.1:9031/metrics > "$quick_dir/${input_len}_after.prom"
done
```
Track live KV/preemptions and prefix hits, compare all completions, TTFT/TPOT. Warm/cold differences and short output may distort results; repeat A/B in reverse order. Do not judge a decode candidate solely on short prefill-heavy screen. Only promising 4K gain and protected 16K proceed.

## C. Matched unchanged half-run
Commit tested source before server startup. Profiler OFF; verify production worker, identical runtime, flags, cache/compile protocol and script. Both cases unchanged [[4096,1024,64,256],[16384,1024,64,128]], two repetitions each. At least 150 min for one or 275 min for two initially; 90 min hard floor not sufficient. Reserve 25 min; do not launch if guard refuses. Reuse fresh exact matched control from same allocation if available.
```bash
python3 autoresearch/session_budget.py status
python3 autoresearch/session_budget.py admit --runs 2
# On each separately restarted owned server with its committed source:
python3 autoresearch/session_budget.py run-half --push-checkpoint
```
Guard checks committed trees, health, deadline, script hash, not worker identity/eligibility/accuracy. Verify four SUCCESS rows, request/token counts, no failures. Read-only evaluator:
```bash
python3 autoresearch/evaluate.py \
 --control-summary /real/A/summary.csv --control-raw /real/A/raw_runs.csv \
 --candidate-summary /real/B/summary.csv --candidate-raw /real/B/raw_runs.csv \
 --json-out .autoresearch-work/comparison.json
```
Zero exit = valid CSV only. Numeric pass requires 4K ratio >=3011.12/2817.15, 16K no loss, mean TTFT/TPOT <=5% increase; verify matched conditions and repeat independent A/B. Official full-run absolute targets and MATH-500 are separate human-run final gates; never claim full-run success from half-runs. If insufficient time or human running manually, hand off rather than duplicating.

## D. Preserve and publish
```bash
python3 autoresearch/checkpoint.py --summary 'Hypothesis and observed result' \
 --evidence 'Source SHA, metrics, artifacts, correctness and limitations' \
 --next-step 'Next experiment or blocker' --status provisional --push
```
Commit source separately; helper only pushes evidence, does not open PR. Open candidate PRs with observable improvement even below target, record limitations and negative results, maintain HANDOFF and off-host backups. Continue until verified development target or actual deadline/blocker. No auto-merge.

## Paste-ready task
> Work as Codex Terra 6.1 via authorized BI-V150 Remote-SSH in the committed v16 checkout. Follow AGENTS.md, program.md and autoresearch/RESEARCH_QUEUE.md. Pin exact v14-derived baseline and confirmed allocation deadline. Do not repeat v0–v15 completed experiments. Optimize 4K for >=3011.12/2817.15 matched throughput ratio with strict 16K non-regression; verify both workloads, correct execution and competition constraints. Publish evidence-backed provisional PRs even below target. Continue until replicated development goal or deadline; human confirms full official +60% vs v0, 16K >=2073.93 and MATH-500 Level 3 before submission.
