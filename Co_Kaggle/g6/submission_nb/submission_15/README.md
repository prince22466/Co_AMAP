# Submission 15 — bounded 12B repair workflow

Replaces the previous E4B submission 15 with **gemma-4-12b-it-qat-w4a16-ct**, the same base model as submission 13. All three LLM stages use that model.

The goal is at least **2 newly resolved tasks out of the same first 10**. This is an acceptance target, **not a measured score**. The release has CPU regression/runtime validation, not a Gemma benchmark run.

## Changes tied to the observed failures

| Observed problem | Change |
| --- | --- |
| Optional helpers and repair scripts were often skipped | A real SequentialAgent runs a LoopAgent containing coder then verifier, at most twice, followed by finalizer. |
| Skills assumed `/workspace` existed inside subprocess Python | One workbench discovers the actual checkout from sandbox PYTHONPATH; verified through ADK's real temporary-directory wrapper. |
| Missing script arguments and names wasted tool calls | One script, six actions, concrete call syntax in each stage prompt; no workspace/budget fields or prerequisite load_skill call. |
| Failed edits were retried; malformed Python reached submission | Exact-match edits, parse-before-write, protected test/configuration paths, repeated-failure rejection, bounded workbench calls. Native edit/write tools are removed. |
| Tool transport success was mistaken for passing verification | Explicit outcomes; fresh subprocess assertions, source import provenance, caught assertion failure detection, timeout cleanup and patch-bound evidence. |
| Long deliberation and repeated context consumed time | Short role prompts; include_contents: none with explicit state handoffs; capped output/thinking; four-minute task timer. |
| Local tests imported installed packages or lacked dependencies | v8 adds src/ imports to the local subprocess environment, checks offline test dependencies and runs empty-patch CPU baselines before inference. |
| Empty patches and broken environments obscured progress | v8 retains raw results and separately reports newly resolved tasks using matched baselines; infrastructure errors retain the original denominator. |

The controller **enforces stage order, at most two loop rounds, and finalizer-only access to submit_patch**. The workbench enforces its edit checks and a 32-action limit (audit remains available). Stage-specific call counts and semantic coverage are instructions, not hard guarantees. The verifier can still choose weak assertions or skip a check; finalization captures an unverified patch honestly rather than losing all work. Arbitrary check code runs within the competition sandbox; the workbench is a reliability tool, not a security boundary.

Cross-layer changes use a short public API → forwarding → consumer trace, followed by a behavioral check. The large advisory graph scripts and optional agent consultations are removed. This is a fixed workflow graph, not a claim of a mandatory code-graph analysis on every task.

## Budgets

- Session: 4 minutes; 40 native/harness tool calls; 48 model turns; 30-second command timeout.
- Workbench: 32 actions per task, plus audits; individual assertion checks at most 15 seconds.
- Coder: 4096 output / 1024 thinking tokens; verifier: 2048 / 512; finalizer: 768 / 128.
- v8 disables event compaction to avoid uncapped summarizer requests. The official harness may configure compaction differently; the package cannot override host App configuration.

Four minutes is the agent timer, not total GPU allocation: model loading, sandbox setup, final verification and notebook overhead are additional. Two rounds do not guarantee that a model will return control before the global deadline.

## Test and package

From `Co_Kaggle/g6/tests_on_tasks/tools`:

```bash
python test_submission14.py -v
python build_submission14.py
python verify_submission14_runtime.py
python test_submission14_eval.py -v
python build_testing_v8.py
```

The first two commands need Python and PyYAML. Runtime/evaluation checks use the competition wheelhouse versions: adk-submission 0.2.13, adk-eval-core 0.1.0, google-adk 1.36.1, google-genai 2.11.0, swegemma 0.2.11, plus their dependencies and pytest. No inference model is loaded by these checks. Runtime results are in `../../tests_on_tasks/reviews/submission_14/` and are explicitly labeled synthetic.

Upload `../artifacts/submission_14.zip`. Use `../../tests_on_tasks/gemma-4-developer-agent-testing_v8.ipynb`:

1. Attach the revised ZIP, competition data, official runtime/test wheels, and the same 12B weights as submission 13. Replace the old E4B Kaggle dataset contents.
2. Run default `RUN_MODE='preflight'` with Accelerator=None. Keep `TASK_LIMIT=10` for the acceptance set. Missing dependency wheels or broken fixtures must be fixed in the environment; v8 does not rewrite tests to make them pass.
3. Set `RUN_MODE='evaluate'` on the intended GPU, retaining the same task order. Preflight runs before vLLM starts. The notebook inherits the four-minute limit instead of silently disabling it.
4. Read `acceptance.json` alongside the raw diagnostics: a new success must have a nonempty patch, pass verification, and fail the empty-patch baseline. A 2-task smoke run cannot satisfy the 2/10 target.
5. For old saved patches, select `RUN_MODE='replay'` and point `REPLAY_PATCH_DIR` at their `patches` directory. Both `<id>.patch` and v7's `<id>_returned.patch` names work. No inference is used.

v8's subprocess path/dependency fixes are local evaluation changes. Official competition Docker results can differ. The notebook preserves official test selection and grading, uses no gold solution patches, and never passes preflight test output to the agent.
