# v9 research — directly inside Remote-SSH Codex IDE

No SSH bridge, local Codex CLI, background supervisor, or rsync scratch clone is required. Codex is installed as an IDE extension on the remote host. Open /root/Optimization/v9 in that Remote-SSH workspace; all edits and serving runs use that path. The local repository mirror is Co_Kaggle/g7/tests_BI-V150/Optimization/v9.

The ONLY allowed remote source changes are within /root/Optimization/v9. The provided benchmark is an immutable external INPUT, not a source workspace:
    /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py

Its provided values are RUNS=2, SKIP_FIRST=0. Do not copy or edit the benchmark. The helper runs it with cwd under v9/.autoresearch-work so even CSVs and logs are under v9.

## Remote terminal

    cd /root/Optimization/v9
    python3 -m unittest discover -s autoresearch/tests -v
    python3 autoresearch/bench.py start
    python3 autoresearch/bench.py ready
    python3 autoresearch/bench.py benchmark
    python3 autoresearch/bench.py stop

The start helper uses v9's FlagGems/src and vllm-plugin-FL via PYTHONPATH, VLLM_PLUGINS=fl. It keeps the established serving config fixed:

    vllm serve /workspace/MiniCPM5-2B \
      --port 9031 \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --served-model-name minicpm \
      --gpu-memory-utilization 0.85 \
      --max-model-len 131072

The half benchmark command executed, with unchanged workload shapes:

    python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py \
      --model /workspace/MiniCPM5-2B \
      --served-model-name minicpm \
      --port 9031 \
      --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'

bench.py writes each run's log, manifest and original raw_runs_*.csv and summary_*.csv to a unique directory under .autoresearch-work/bench_runs. It refuses to launch over an occupied port; stop signals only its own tracked process group. Do not stop any external vLLM instance. If start fails, inspect .autoresearch-work/serve.log; do not silently switch to another source directory or serving configuration.

All benchmark results are development *half-runs*, not official full runs. Do not run MATH-500; user will run accuracy evaluation separately.

## Compare a new kernel candidate

Take actual paths emitted by the baseline and candidate benchmark commands (no illustrative numbers are actual measurements):

    python3 autoresearch/evaluate.py \
      --baseline-summary .autoresearch-work/bench_runs/BASE/benchmark_results/summary_X.csv \
      --baseline-raw .autoresearch-work/bench_runs/BASE/benchmark_results/raw_runs_X.csv \
      --candidate-summary .autoresearch-work/bench_runs/CAND/benchmark_results/summary_Y.csv \
      --candidate-raw .autoresearch-work/bench_runs/CAND/benchmark_results/raw_runs_Y.csv \
      --json-out .autoresearch-work/gate-candidate.json

The evaluator checks both shapes, exactly two successful raw runs per shape, complete requests, and the summary. It reports the numerical improvement and milestones (4K >=2450 OR 16K >=1450) and TTFT/other-case changes. A new PR is justified only for an **observable reproducible** genuine kernel improvement, not noise: repeat interleaved A/B when gains are modest; confirm actual optimized production kernel was launched and GPU tests passed. Historical v8 FULL-run values cannot be compared to v9 half-run values.

## IDE mission

Read AGENTS.md and GOAL.md, run a fresh v9 half baseline, profile, change one real kernel/operator optimization, test, repeat half-run comparisons, and immediately open a PR for each proven serving improvement (even short of milestone), then continue until the performance OR goal, a specified deadline, or a blocker. Log rejected hypotheses in EXPERIMENTS.md. Use GitHub CLI if available on the remote host; if GitHub authentication prevents PR creation, report it as a blocker, not a finished PR.

If /root/Optimization/v9 is a standalone source copy rather than a Git checkout, bench.py still runs and records kernel-file SHA-256 hashes; the Git commit field in its manifest will be null. To create evidence-backed PRs, transfer only the changed v9 files into the corresponding tracked local v9 mirror (or use a correctly configured remote Git checkout) before committing and pushing. Do not change other Optimization versions or imply that a PR was filed when GitHub access is absent.

The IDE extension being available does not imply an independent Codex CLI or autonomous daemon. Keep the IDE agent active during experiments. PRs must include candidate SHA, prior best SHA, half-run CSV paths, repeat runs, relative deltas, both TTFTs, failure counts, proof of the edited kernel in production, correctness tests and competition compliance. Do not claim accuracy was checked.
