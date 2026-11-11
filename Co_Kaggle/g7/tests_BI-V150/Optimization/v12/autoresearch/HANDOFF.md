# v12 autoresearch handoff

**Status: SETUP ONLY — no v12 GPU tests or throughput measurements performed by this PR.**

1. Open the draft PR branch in the authorized `/root/Optimization/v12` Remote-SSH workspace without overwriting user edits; verify v12 imported modules and source/benchmark SHA.
2. Obtain actual GPU allocation deadline; initialize `session_budget.py` (do not invent or reset time). Run offline helper tests and baseline attention correctness.
3. **Before any expensive half-run**, profile 4K/16K prefill/mixed/decode and cheaply screen one exact attention-dataflow hypothesis plus one sparse/nonlocal-retrieval hypothesis.
4. Reject invalid or implausible candidates early. Only after real kernel and short-serving matched A/B evidence, measure a **fresh original-v12 half-run control** and a candidate using the untouched external benchmark.
5. Approximate sparse results must be checked against dense middle-context retrieval and human-owned unchanged MATH-500; organizer comparability rules still apply.
6. Checkpoint/push evidence at least every 30 minutes, commit candidate operator code separately, back up ignored raw traces off-instance. On deadline leave best verified source SHA, rejected options, raw log hashes and next action.

For procedures read `AGENTS.md` and `program.md`; results go in append-only `autoresearch/EXPERIMENTS.md`. Neither this file nor the tools starts Codex or runs a GPU job.
