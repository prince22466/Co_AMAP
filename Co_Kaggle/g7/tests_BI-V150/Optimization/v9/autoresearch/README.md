# v9 Codex + SSH autoresearch

This directory makes the MiniCPM5-2B / Iluvatar BI-V150 v9 research task
repeatable, but DOES NOT start any remote process by itself. It is not evidence
that Codex, a GPU test or a new performance PR has already run.

## Layout and permissions

- Local: a real Git checkout of Co_AMAP. Codex CLI edits only v9; GitHub CLI
  can push feature branches and create checkpoint PRs. Run Codex from v9 so
  its AGENTS.md and GOAL.md are in scope.
- Remote: authorized BI-V150 reachable by SSH. remote.py copies the v9
  FlagGems and vllm-plugin-FL source trees via rsync into a dedicated
  /root/Optimization/v9_autoresearch scratch path, NOT the original v9/v8.
  This path is fixed in remote.py to prevent accidental deletion elsewhere.
- The remote bridge only manages its own vLLM process (recorded PID and
  inspected command line). It never kills unrelated servers. The user must
  ensure GPU and port 9031 are otherwise idle.
- Unmodified competition benchmark: benchmark_throughput_serve.py as shipped
  in the v9 copy; both full workloads, RUNS=4, SKIP_FIRST=1. It writes
  /workspace/benchmark_results/raw_runs_*.csv and summary_*.csv.
- Accuracy: evaluate the actual served MiniCPM5-2B candidate with the
  existing MATH-500/evalscope procedure and unchanged dataset at
  /workspace/evalscope-datasets/math_500/test.jsonl. Verify the installed
  evaluator command on the GPU host; no substitute first-50 test counts.
- Approval: network access is required for Codex to use SSH and gh; the
  supervisor uses a workspace-write sandbox, not unrestricted/yolo access.
  This grants outbound shell network connections. Use a trusted dedicated
  workstation identity with least-privilege SSH/GitHub access.

## One-time prerequisites

The operator must install Codex CLI (authenticated), Python 3.10+,
git, gh (authenticated and authorized for PR creation), openssh and rsync
on the local machine. On BI-V150, the model and GPU runtime stack
(vllm, vllm-plugin-FL, FlagGems, PyTorch, pytest) must already be present.

Create an SSH alias in your own ~/.ssh/config using your private HostName,
User and IdentityFile. Verify the fingerprint outside this repo and configure
known_hosts. DO NOT commit keys or credentials. For example, name the
authorized host "bi-v150" and first verify a noninteractive SSH check:

~~~bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes bi-v150 true
~~~

Check out the v9 harness PR branch or merge it into main. Then:

~~~bash
cd Co_Kaggle/g7/tests_BI-V150/Optimization/v9
export V9_SSH_HOST=bi-v150
export V9_REMOTE_ROOT=/root/Optimization/v9_autoresearch

python3 -m unittest discover -s autoresearch/tests -v
python3 autoresearch/remote.py check
python3 autoresearch/remote.py sync
python3 autoresearch/remote.py tests
~~~

The remote bridge runs a separate, fixed vLLM server configuration:
model /workspace/MiniCPM5-2B; port 9031; served name minicpm;
FULL_DECODE_ONLY; gpu-memory-utilization 0.85; max-model-len 131072.
It explicitly clears operator/graph env overrides that could conceal
whether optimized v9 kernels run. These must remain identical A/B.

The baseline collection steps are:

~~~bash
python3 autoresearch/remote.py start
# Wait for the /v1/models readiness endpoint in the server log.
python3 autoresearch/remote.py benchmark
python3 autoresearch/remote.py collect
python3 autoresearch/remote.py stop
~~~

Review log /root/Optimization/v9_autoresearch/.autoresearch-runs/serve.log
over SSH. Before each new candidate, sync new v9 code and restart the
owned server. Repeat baseline/candidate in an interleaved A/B sequence if
run-to-run noise or shared GPU load is detected. The benchmark's fixed
"Total tok/s" is the only acceptance throughput measure.

## Check an observed candidate

Use the untouched FULL output CSVs from baseline and candidate and the
ACTUAL MATH-500 evaluation log (replace illustrative paths and score):

~~~bash
python3 autoresearch/evaluate.py \
 --baseline-summary /path/baseline_summary.csv \
 --baseline-raw /path/baseline_raw.csv \
 --candidate-summary /path/candidate_summary.csv \
 --candidate-raw /path/candidate_raw.csv \
 --math500-score 0.96 \
 --math500-log /path/actual_math500.log \
 --json-out .autoresearch-work/gate-round1.json
~~~

This numerical gate requires:
- BOTH exact official shapes, four raw runs each, with ALL four
  rounds' requests successful;
- >=1% throughput increase in at least one workload and no >1% throughput
  loss in either workload;
- no >1% mean TTFT regression for either workload;
- MATH-500 >=0.95.

The scorer cannot prove that CSVs, typed score, and server correspond to the
right SHA/hardware. Codex must independently check logs, timestamps, server
config, real optimized kernel invocation and repeatability before an
improvement PR. The above score 0.96 is an EXAMPLE, not a measured v9 result.

## Launch the bounded research loop

~~~bash
export V9_SSH_HOST=bi-v150
python3 autoresearch/run.py --hours 8 --round-minutes 90
~~~

run.py starts one noninteractive Codex turn per round with workspace-write
and explicitly enabled network access (needed for SSH and gh). It enforces
the wall-clock limit for each turn and the overall deadline. It looks for
.autoresearch-work/round-report.json and the evaluate.py gate JSON.
Codex is responsible for experiments, official evaluation, logging,
pushing changes and opening a PR on every accepted improvement.
Each next round reads the accepted best state from Git and EXPERIMENTS.md.

If Codex reports a verified 2700 (4K) or 1500 (16K) and a checkpoint PR,
the loop stops. Otherwise it continues until the time budget or a blocker.
If a run crashes or times out, logs remain in .autoresearch-work.
A timed-out local Codex process may leave an agent-owned remote server:
manually run remote.py stop. Do NOT stop another user's server.

Codex Goal-capable versions can alternatively run interactively using
the contract in GOAL.md, but that does not replace the hard wall-clock
bound provided by run.py.

## PR and compliance evidence

For EACH accepted >1% reliable gain, Codex must open a separate PR before
continuing, optionally stacked on the previous best branch. Each PR must
include: best baseline/candidate SHAs; actual kernel or operator code diff;
GPU trace or launch proof; unchanged official benchmark commands; both
full 4K/16K raw and summary metrics with throughput and TTFT deltas;
MATH-500 accuracy and exact evaluation command; no request failures;
fallback unit tests; measurement noise/repeat evidence; risks and rollback.
Do not auto-merge PRs. Do not commit raw logs, credentials or weights.

Historical v8 full-run reference (NOT new v9 test): 4K 2331.29 and 16K
1251.22 tok/s; accuracy 98.1%. Fresh A/B is mandatory for every claim.
