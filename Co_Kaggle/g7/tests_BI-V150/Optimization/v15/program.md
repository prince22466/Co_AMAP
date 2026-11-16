# v15 Remote-SSH runbook

Read AGENTS.md first. This is a human-started Codex Terra 6.1 task, not a background daemon.

## A. Verify baseline and budget

Run on the authorized remote host. Use the actual v15 checkout (if nested, locate it without changing other versions). Ensure your branch contains this setup and source trees match the original, then pin the clean baseline SHA. An exported v15 folder with no tracked Git source cannot publish reliable PRs: set up a real monorepo checkout/sparse checkout first. Use its actual v15 path in every command.

```bash
cd /root/Optimization/v15
pwd
git rev-parse --show-toplevel
git ls-files --error-unmatch FlagGems/src/flag_gems/__init__.py vllm-plugin-FL/vllm_fl/__init__.py
git status --short
git branch --show-current
git rev-parse HEAD
sha256sum /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL"
export VLLM_PLUGINS=fl
python3 -m unittest discover -s autoresearch/tests -v
python3 - <<'PY'
import importlib
for name in ("flag_gems", "vllm_fl", "flag_gems.runtime.backend._iluvatar.ops.mm",
             "flag_gems.runtime.backend._iluvatar.ops.linear",
             "vllm_fl.worker.worker", "vllm_fl.worker.model_runner"):
    m = importlib.import_module(name)
    print(name, m.__file__)
PY
ixsmi
```

Verify model/versions, actual worker paths (not just shell imports), memory and owned server. Before initialization, check the external script read-only: verify RUNS=2, SKIP_FIRST=0, actual CLI and output location; if different, stop rather than patch it. Record script hash, model/config fingerprint, server flags, GPU/device and cache state. Initialize once using the operator-confirmed UTC deadline below (replace placeholder, no guessed duration), only after verifying the external script; if a genuinely new allocation is confirmed by the operator, archive the previous state and evidence before initializing a new session; never reset within the same allocation. Do not launch expensive control half-run before promising screen.

```bash
python3 autoresearch/session_budget.py init --deadline-utc 'ACTUAL-ISO8601-UTC-DEADLINE'
python3 autoresearch/session_budget.py status
```

## B. Correctness and profiling

```bash
cd /root/Optimization/v15
(
 cd FlagGems
 python3 -m pytest -q -s tests/test_mm.py tests/test_linear.py tests/test_addmm.py
)
python3 -m pytest -q -s --noconftest \
 vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py \
 vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py \
 vllm-plugin-FL/tests/functional_tests/test_biv150_mixed_paging.py
```

Run targeted subsets first if suite is large; record exact cases and skips. Add direct vendor tests (no fallback) for changed MM/linear, and cache lifecycle/graph tests for memory changes. Use independent torch references, explicit tolerances, nontrivial tensors and repeated runs. Inspect actual installed vLLM APIs read-only. Profile phase-specific shape histograms and timings; measure allocator snapshots and KV block/token counts before/after profile, graph capture, initialization and load; use ixsmi and /metrics with counters as deltas. Use old v7 findings and v14 traces as hypothesis seeds only (links in queue). Remove diagnostic instrumentation before final runs.

## C. Fixed server and non-scoring screens

Use exactly the same server flags on every A/B. Restart only the server you own, using a known PID/process group. Keep baseline and candidate source hashes, logs and readiness evidence. Ensure no stale imports or graph/tuning artifacts contaminate the comparison.

```bash
cd /root/Optimization/v15
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

In a separate shell, verify health and actual candidate execution. For quick screen run these independent diagnostics identically for control/candidate, using a matched clean restart before each two-case sequence, the same 4K→16K order and fixed seed. Do not modify or replace the official benchmark. Stop a screen if budget is insufficient; incomplete screens are not comparable.

```bash
cd /root/Optimization/v15
set -euo pipefail
curl -fsS http://127.0.0.1:9031/health
mkdir -p .autoresearch-work/quick
quick_dir=$(mktemp -d "$PWD/.autoresearch-work/quick/screen_XXXXXXXX")
for input_len in 4096 16384; do
  curl -fsS http://127.0.0.1:9031/metrics > "$quick_dir/${input_len}_before.prom"
  vllm bench serve --backend vllm --model minicpm \
    --tokenizer /workspace/MiniCPM5-2B --endpoint /v1/completions \
    --host localhost --port 9031 --dataset-name random --ignore-eos \
    --seed 42 --random-input-len "$input_len" --random-output-len 128 \
    --max-concurrency 8 --num-prompts 8 \
    2>&1 | tee "$quick_dir/quick_${input_len}.log"
  curl -fsS http://127.0.0.1:9031/metrics > "$quick_dir/${input_len}_after.prom"
done
```

For a promising GEMM candidate, screen 4K at concurrency 64; for a KV-capacity candidate, screen 16K at concurrency 64. Set input_len to the corresponding length before each run. Restart the owned server for a matched clean state and run the following separately on both versions. Sample live /metrics, GPU memory and running/waiting every few seconds in a separate shell (start before the load and stop after completion).

```bash
set -euo pipefail
cd /root/Optimization/v15
mkdir -p .autoresearch-work/quick
input_len=16384 # use 4096 for the 4K GEMM pressure screen
pressure_dir=$(mktemp -d "$PWD/.autoresearch-work/quick/pressure_${input_len}_XXXXXXXX")
curl -fsS http://127.0.0.1:9031/metrics > "$pressure_dir/before.prom"
vllm bench serve --backend vllm --model minicpm \
 --tokenizer /workspace/MiniCPM5-2B --endpoint /v1/completions \
 --host localhost --port 9031 --dataset-name random --ignore-eos \
 --seed 42 --random-input-len "$input_len" --random-output-len 128 \
 --max-concurrency 64 --num-prompts 64 2>&1 | tee "$pressure_dir/pressure.log"
curl -fsS http://127.0.0.1:9031/metrics > "$pressure_dir/after.prom"
```

If the low-concurrency screen does not exercise the optimized production shape or memory pressure, do not reject a candidate on that screen alone. If 128 output tokens do not expose decode growth/preemption, use a matched longer diagnostic only when budget allows; never change the official workload. These screens are not scoring and low concurrency cannot validate capacity improvement. Match cache warm/cold state and capture prefix hits. Record actual completed requests, tokens, TTFT/TPOT and errors. Alternate A/B order and rerun plausible gains; compare memory and kernel evidence, then decide whether to spend on full half-runs.

## D. Guarded unchanged half-run

Official development half-run external script is read-only: RUNS=2, SKIP_FIRST=0, cases [[4096,1024,64,256],[16384,1024,64,128]]. Full official scoring is a separate procedure. Commit tested source, start a fresh owned profiler-OFF server from that exact commit, and confirm matching script hash and actual worker provenance before each run. Do not change source until the server stops. Preflight: at least 150 min for one or 275 min for two initially; 90 min is a hard minimum, not sufficient admission. Preserve 25 min. Run only after quick A/B shows benefit and tests pass.

```bash
python3 autoresearch/session_budget.py status
python3 autoresearch/session_budget.py admit --runs 2
# After matched control/candidate plan and a checkpoint, for each owned server:
python3 autoresearch/session_budget.py run-half --push-checkpoint
```

Run each control/candidate independently with matched restarts. If insufficient time for a matched pair, checkpoint instead unless a comparable control is already archived and one run is admitted. Verify two SUCCESS raw rows per length, 256/128 completed requests, exact workload, summary integrity. Use read-only evaluator:

```bash
python3 autoresearch/evaluate.py \
 --control-summary /real/A/summary.csv --control-raw /real/A/raw_runs.csv \
 --candidate-summary /real/B/summary.csv --candidate-raw /real/B/raw_runs.csv \
 --json-out .autoresearch-work/comparison.json
```

Repeat independent matched A/B before verified claim. Evaluator validates both cases, full token/request counts, raw/summary consistency, mean TTFT and TPOT, and reports a combined numeric screen. A zero exit only means the CSVs are valid, not that the screen passed. Check competition score/latency limits and matched conditions separately. Never count historical official values as matched controls. Human runs unchanged MATH-500 before final acceptance.

## E. Preservation

```bash
python3 autoresearch/checkpoint.py --summary 'Hypothesis and observed result' \
 --evidence 'Source SHA, metrics, artifacts, correctness and limitations' \
 --next-step 'Next experiment or blocker' --status provisional --push
```

Save raw traces off-instance. Commit/push tested inference source separately; the helper only commits evidence. Open a candidate PR when measurable improvements appear even if smaller than goal; label provisional until independent replication/quality. Continue until goal or actual deadline; no auto-merge. Maintain the manual section of HANDOFF with original SHA, best verified SHA, rejected ideas, evidence and remaining risks; the helper replaces only the generated latest-checkpoint section.

## Paste-ready Codex task

> Work as Codex Terra 6.1 on the authorized BI-V150 Remote-SSH host. Follow AGENTS.md, this runbook and autoresearch/RESEARCH_QUEUE.md in the real v15 Git checkout. Pin the baseline and allocation deadline. Investigate grouped MM and genuine KV-capacity savings first; use the specified acceptance loop, preserve evidence and publish candidate PRs. Continue until the verified >=5% throughput goal (stretch >=10%) or the actual resource deadline, without violating competition rules or auto-merging.
