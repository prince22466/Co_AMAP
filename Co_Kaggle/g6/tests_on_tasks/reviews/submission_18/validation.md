# Submission 18 validation

Submission 17's archived 30-task run had no nonempty patches or successful workbench edits.
This candidate changes the execution interface, keeping the E4B model and declared budget.
It does not claim to repair every possible model failure or improve the benchmark score yet.

| Observed Submission 17 failure | Change / regression evidence |
| --- | --- |
| 393 calls omitted required skill wrapper arguments | No skill toolset; native tool schemas are exposed directly. CPU tests verify the dispatcher is absent; runtime tests invoke native reads/edits without wrappers. |
| 133 nonexistent script calls; repository paths confused with skill scripts | Actual read_file reads synthetic `scripts/add_latest_release_date.py` as a repository path. |
| 29 dispatched edits rejected for missing module/check code | Native edit_file accepts filepath/old_string/new_string; runtime test edits and captures a nonempty patch without an assertion argument. |
| Rejected calls bypassed workbench accounting | Runtime test repeats nonexistent-file reads and verifies native budget exhaustion. Calls rejected before native execution can still bypass tool accounting; the model-turn cap remains. |
| Verification accepted printing plus assert True | The external smoke gate requires a fixed output-sensitive command and an independent final oracle. The exact archived vacuous-check code cannot satisfy that gate. This is NOT an assertion-quality guarantee in the submitted agent. |
| Finalizer starvation and empty submissions | One agent owns submission from its first turn. The smoke gate rejects empty patches, missing submission and duplicate submission. The model can still fail to submit. |
| Stale verification | External smoke checks the final patch hash against the last passing behavioral check; regression changes the hash and expects rejection. Arbitrary benchmark evidence freshness remains a prompt instruction. |

`submission17_failures.json` contains five real trace excerpts with original report filenames
and SHA-256 hashes. A regression reads the archived ZIP and checks that every excerpt actually
occurs in its named report. These excerpts are development evidence and never packaged or
provided to the agent. They are not gold patches or evaluator tests.

## Executed validation

- 8 CPU regressions passed, including archive provenance, negative smoke-gate cases and
  deterministic three-file packaging.
- 6 official-runtime regressions passed using adk-submission 0.2.13, adk-eval-core 0.1.0,
  google-adk 1.36.1, google-genai 2.11.0 and swegemma 0.2.11.
- The packaged candidate completed a scripted ADK Runner exercise using actual native tools:
  baseline output assertion fails; deliberately wrong edit; same assertion fails; repair;
  same assertion passes; diff check; one nonempty submission. Seven counted tool calls and
  ten scripted model calls. The independent final behavior check passed.
- The official compiler's LiteLlm configuration was checked for enable_thinking=false.
- GitHub CPU CI runs the 8 CPU tests and rebuilds/checks the manifest. The 6 runtime tests
  require the competition wheelhouse and were run locally, not in that CI job.

No actual E4B inference, GPU smoke or competition benchmark was run. The scripted model chooses
valid direct calls deliberately; it cannot prove E4B will make those choices. The README gives
an optional real-model smoke invocation using the same helper and compiled candidate. Run that
before another multi-task GPU evaluation.

Only new Submission 18 files and its CI workflow are introduced. Existing notebooks,
submission_records.md, previous agents and evaluator grading are unchanged.
