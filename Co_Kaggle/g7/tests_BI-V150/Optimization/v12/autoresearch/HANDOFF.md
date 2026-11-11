# v12 autoresearch — latest handoff

Updated: 2026-11-11 (setup PR)

Status: **not started on GPU**. This change prepares a v12 Codex research protocol and copied/adapted evidence-budget tools; it does not launch Codex or run benchmarks.

Main source operator trees are unchanged from v10-equivalent v12. First tasks:
1. Confirm this PR branch is checked out in the authorized /root/Optimization/v12 workspace; preserve all uncommitted remote user changes.
2. Obtain real GPU allocation expiration from the operator; run `session_budget.py init` once per allocation.
3. Verify v12 source/benchmark/model provenance, run budget helper and original GPU regression tests, then obtain a fresh dense v12 control.
4. Profile actual prefill/decode/mixed critical path; screen one exact kernel hypothesis and one sparse long-context hypothesis; write evidence to EXPERIMENTS.md.
5. Checkpoint and push on a non-main draft PR branch at least every 30 minutes and at meaningful test outcomes; off-instance back up large traces.

Do not claim GPU test results, MATH-500 pass or model ability preservation from this handoff. Candidate sparse masks intentionally change model behavior and remain research-only until validated and judged eligible.
