# Submission 16 validation

Validated locally on 2026-10-25. No GPU inference or competition score was produced.

- 19 regression tests passed: mandatory/recoverable arguments, counted rejected calls, repeatable
  audits, role actions, atomic edits, automatic assertions, failed-check repair, stale evidence,
  import provenance, timeout/mutation detection, protected files, deterministic packaging,
  notebook defaults and fail-closed smoke evidence.
- The packaged ZIP passed the official compiler, ADK Runner and subprocess skill executor using
  a **scripted model**, in 11 skill calls. It recovered from an omitted action, retained an intentionally
  wrong edit, failed its assertion, handed off to the verifier, repaired in the second coder round,
  reverified, audited fresh evidence and submitted a nonempty synthetic patch exactly once.
- Runtime versions: adk-submission 0.2.13, adk-eval-core 0.1.0, google-adk 1.36.1,
  google-genai 2.11.0 and swegemma 0.2.11. The compiler's actual LiteLlm configuration disables
  thinking on all three stages. This checks request configuration, not Gemma's response behavior.
- CPU CI repeats the regressions, generates the ZIP, and checks that rebuilding its committed
  manifest and notebook produces no tracked changes. It publishes the ZIP as an Actions artifact.
  The official-runtime smoke is a separate local command because
  it requires the competition wheelhouse and dependencies.

Run from the repository root:

```bash
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p 'test_submission16*.py' -v
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission16.py
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission16_notebook.py
# Install the pinned competition wheels and their CPU dependencies before this command:
python Co_Kaggle/g6/tests_on_tasks/tools/verify_submission16_runtime.py --output-dir /tmp/submission16-runtime
```

The smoke command writes events.json, smoke.json and synthetic.patch. These are mechanics evidence,
not task results. The Submission 16 notebook embeds the same helper and supplies the real Gemma model registry;
`evaluate` refuses to start the competition tasks unless that real-model smoke passes.

The official loop enforces stage ordering and at most two rounds. It does **not** force an LLM stage
to yield early. Per-stage call/time targets remain advisory; the shared harness budget still applies.
Assertions run after edits and are bound to the patch hash, but their semantic quality depends on
the model. Verifier action restrictions are reliability controls, not a security boundary around
arbitrary assertion code. A finalizer may submit available unverified work honestly.

The next acceptance step is the same first ten tasks with the released 4-minute / 40-call / 48-turn
configuration and compaction disabled. The target of two newly resolved tasks is **not yet measured**.
