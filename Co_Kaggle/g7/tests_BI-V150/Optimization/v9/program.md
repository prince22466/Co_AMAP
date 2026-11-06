# BI-V150 MiniCPM5-2B v9 — Remote-SSH Codex autoresearch

The v9 inference code started as a copy of v8. The Codex IDE extension is **already installed in the Remote-SSH session**: open /root/Optimization/v9 in the remote VS Code workspace, and start the agent from THIS folder. No additional local SSH orchestrator, local Codex CLI, or v9_autoresearch copy is needed.

LOCAL repository: Co_Kaggle/g7/tests_BI-V150/Optimization/v9 (Windows suffix g7\tests_BI-V150\Optimization\v9).

ONLY remote editable/runnable source: /root/Optimization/v9.

**Half-run objective:** 4K total tok/s >= 2450 OR 16K >= 1450 (both desirable); both cases must be measured every round with the unchanged /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py. Accuracy/MATH-500 is reserved for the user after the milestone and is NOT part of the agent loop.

Read AGENTS.md and autoresearch/GOAL.md before each research session. The experiment log is autoresearch/EXPERIMENTS.md.

## Quick start in a Remote-SSH terminal

    cd /root/Optimization/v9
    python3 -m unittest discover -s autoresearch/tests -v
    python3 autoresearch/bench.py start
    python3 autoresearch/bench.py benchmark
    python3 autoresearch/bench.py stop

The helper launches vLLM using the **v9 FlagGems and vllm-plugin-FL** in PYTHONPATH, with VLLM_PLUGINS=fl, and saves each unchanged half-run under .autoresearch-work/bench_runs/. Do not use /workspace/FlagGems or /workspace/vllm-plugin-FL as the serving source.

## Paste this mission into the remote Codex IDE chat

    Work exclusively in /root/Optimization/v9. Read AGENTS.md,
    autoresearch/GOAL.md and autoresearch/README.md. Execute successive
    kernel-level optimization experiments for BI-V150. First record a fresh
    v9 half-run baseline; start vLLM from v9 and execute the untouched script
    /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py
    for BOTH 4K and 16K. Reach >=2450 Total tok/s at 4K OR >=1450 at
    16K. Keep going through evidence-based kernel hypotheses until a
    repeatable milestone, the explicitly given deadline, or a blocker.
    Run GPU correctness tests; capture production kernel execution proof.
    At every reproducible improvement, even below goal, commit only v9
    code and open a PR recording baseline/candidate data, then continue.
    Do not edit benchmark scripts or any non-v9 source. Do NOT run MATH-500;
    I will perform accuracy evaluation myself. Provide actual evidence,
    verified PR links, and report blocking issues without inventing results.

The IDE extension is interactive; a running Codex agent will need to be active for the experiments to continue. There is no implied unattended daemon or guaranteed wall-clock timeout. Give the agent a concrete deadline when needed. The PR is a research setup, not a claimed throughput improvement.
