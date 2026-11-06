# BI-V150 optimization v9 (built upon v8)

# Codex autoresearch entry point

The original v9 kernels are an exact copy of the validated v8 baseline.
This folder is equipped for a **bounded, SSH-controlled Codex research loop**.
It does not automatically run when a PR is merged.

- Read [AGENTS.md](AGENTS.md) for the competition restrictions and
  required evidence gates. It is scoped to the v9 folder.
- Read [autoresearch/GOAL.md](autoresearch/GOAL.md) for objective, ranking,
  iteration protocol, MATH-500 and official throughput criteria.
- Follow [autoresearch/README.md](autoresearch/README.md) to configure an
  approved SSH host and launch the supervised research loop.
- Record hypotheses, rejected experiments, and improvement PRs in
  [autoresearch/EXPERIMENTS.md](autoresearch/EXPERIMENTS.md).

Goal: official **4K total tok/s >= 2700** OR **16K total tok/s >= 1500**,
while keeping accuracy >=0.95 and both workload/TTFT regression guards.
Open a new PR for every repeatable allowed improvement >=1%, even before
either goal is reached; then continue until verified goal or time budget.

The code change must be a genuine kernel/operator/data-movement optimization.
No benchmark script tuning, speculative decoding or quantization toggles,
operator-selection tricks or upstream FlagOS merge shortcuts.

To start (after obtaining user-authorized SSH/GitHub/Codex access):

~~~bash
cd Co_Kaggle/g7/tests_BI-V150/Optimization/v9
export V9_SSH_HOST=bi-v150  # replace with your approved ~/.ssh/config alias
python3 autoresearch/remote.py check
python3 autoresearch/run.py --hours 8 --round-minutes 90
~~~

This PR sets up the program; it does not claim a fresh v9 GPU result.
