# v11 autoresearch handoff — setup state

Status: **research harness prepared; no v11 GPU execution verified**.

- Branch/PR: [Codex Terra 6.1 autoresearch setup PR #235](https://github.com/prince22466/Co_AMAP/pull/235)
- Source starting point: v10-derived v11 FlagGems and vLLM plugin with BF16 GQA prefill BLOCK_M=64; this PR does not modify inference kernels.
- Frozen reported v10 half-run anchors: 4K 2372.38 Total tok/s, 16K 1872.70 Total tok/s. These are **not** matched v11 controls.
- Frozen minimum: 4K 2490.999 OR 16K 1966.335. Stretch: 4K 2609.618 OR 16K 2059.970. Both workloads must be disclosed.
- Verified in a previous setup review: existing evaluator's 20 synthetic tests; **new budget/checkpoint tests have not been run on the GPU host**.
- Missing: confirmed operator-provided GPU allocation/deadline, GPU health, current source/import hashes, fresh production profile, v11 matched control, candidate experiments, repeated independent A/B, and human MATH-500.
- Next action: synchronize this PR branch to the authorized /root/Optimization/v11 workspace, confirm GPU health and Git push access; run session_budget.py init with actual remaining time. Before a costly run, use checkpoint.py --push and session_budget.py admit. Prioritize production-profile-based high-confidence kernel experiments. Ensure all ignored raw evidence is backed up off-instance.
- The researcher must update this file on each checkpoint with the latest exact evidence, source commit/hash, provisional/verified status, budget remaining, failures, and the single best next action.

This note preserves **setup state only**. No model speedup or accuracy improvement is claimed.
