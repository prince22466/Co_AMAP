# Submission 14 CPU validation

These are synthetic engineering checks, **not task-level Gemma testing logs**. No inference calls or GPU runs were made. The 2/10 target remains unmeasured.

- Workbench regression suite: 8 tests passed. Covers fresh src-layout imports, failing-before/passing-after assertions, stale evidence rejection, atomic syntax rejection, repeated failure blocking, protected/traversal/ambiguous edits, new-file auditing, caught assertion failures, no assertions, premature exit, timeout, patch mutation, empty/invalid patch audit and reproducible archives.
- Official runtime check: passed with the exact primary competition package versions recorded in `runtime_smoke.json`. Compiles the release ZIP, runs SequentialAgent/LoopAgent using scripted model responses, exercises the real swegemma subprocess executor and ADK run_skill_script wrapper, verifies the second-round state handoff and exactly one finalizer submission.
- Local evaluation regression suite: 4 tests passed, including a real official-harness empty-patch failure followed by a saved-patch pass on a synthetic src-layout package, without any registered inference model. Also checks denominator/empty-patch/baseline-pass handling and known infrastructure error classification.
- Notebook checks: all code cells compile, diagnostics and helper definitions execute, nested workflow metadata resolves, skill source is hashed, first-10 selection and inherited timer are configured, compaction is disabled. Kaggle execution and vLLM startup have not been run here.
- Separate skill forward test: strict/default string normalization exercise in a temporary repository; six test methods failed on baseline and passed after a production-only edit. CLI execution used a different current directory and checkout PYTHONPATH. No model API calls. This is usability evidence only, not benchmark accuracy.

The preserved E4B baseline ZIP is byte-identical to main's previous submission_14.zip. Submission 13 and all existing diagnostics/notebooks remain unchanged.
