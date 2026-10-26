# Submission 20: validate before publishing a candidate

E4B, one agent, no separate finalizer. This is an architectural candidate for the
[17–19 review](../../tests_on_tasks/diagnostics/submission_17_18_19_review.md), not a claim
that real task performance has improved. **Real E4B validation is still pending.**

## Architecture

The model has native `read_file`, `get_status`, and `submit_patch`, plus one skill:
`run_skill_script(skill_name="candidate", file_path="scripts/gate.py", args=...)`.
There are no native editing, writing, or shell tools that can bypass the normal gate.

1. The skill clones the baseline into a candidate directory outside the live repository.
2. Edits affect only that candidate. Invalid Python, protected paths, and removed
   top-level definitions are rejected and rolled back. Editing does **not** require a
   test/probe argument. Two identical failed edits are rejected on the third attempt;
   at most 12 edit attempts are accepted for processing.
3. `check` executes the same model-written behavioral assertion on disposable baseline
   and candidate copies. It requires an assertion failure on baseline and success on
   candidate. Python changes require a local module import and execution of at least
   one changed line. Printed failures, missing dependencies, checkout mutation, and
   setup errors do not qualify. An optional discovered regression test must pass on both.
4. `publish` reruns those checks against the current bytes. Only then does it copy the
   candidate into the live workspace. A failed new candidate leaves the last published
   patch intact. Unexpected live edits block publication instead of being overwritten.
5. The same agent calls native `submit_patch`. Early submission sees the last published
   snapshot (initially empty), not an unverified staged edit.

This is a **publication gate**, not a host callback intercepting `submit_patch`.
The pinned competition compiler provides no submission-defined host callback registry.
A skill executes inside the permitted sandbox; nothing patches the harness or evaluator.

## What is enforced, and what remains uncertain

| Enforced by code | Still depends on the model/runtime |
|---|---|
| Candidate isolation and automatic structural checks after edits | Correct API, complete feature, and meaningful task-specific probe |
| Rollback of rejected edits; bounded edit attempts and identical retries | Model can still waste turns reading or miscalling the ADK wrapper |
| Baseline assertion failure, candidate success, local import, changed-line execution | Coverage of one changed line does not prove all changes or edge cases |
| Fresh checks at publication; no raw native mutation/shell tool | Agent must invoke the skill correctly, publish, and submit before global limits |
| Baseline/environment and real-model pilot promotion checks | Environment repair and actual E4B inference require the testing environment |

A Python probe runs arbitrary repository/test code inside the existing competition
sandbox. Disposable copies and before/after file checks reduce accidental side effects;
**this is not an adversarial security boundary**. A malicious probe could reach other
sandbox paths or falsify output. Do not claim unbypassable security or semantic proof.
Process termination during a multi-file copy is also not a whole-directory atomic commit.

## Avoiding the previous failures

- S17: one fixed skill name/script, a no-argument status operation, simple edit fields,
  native-style `filepath`/`old_string`/`new_string` aliases, and no mandatory assertion
  to edit. There is no separate verifier/finalizer to starve.
- **Tradeoff:** mutation once again depends on skill dispatch. Native edits cannot stay
  exposed while also preventing unchecked live edits in this runtime. Scripted wrapper
  recovery is tested; real E4B reliability is not yet demonstrated. The real smoke gate
  must pass before a larger run.
- S18/S19: audit is part of mutation/publication rather than an optional model action.
  `assert True`, unrelated already-passing tests, and printed `FAILED` cannot satisfy the
  fail-to-pass requirement. A syntactically clean but wrong solution can still pass a weak
  model-written probe; hidden evaluator tests remain the correctness authority.
- No retry threshold, task ID, expected benchmark patch, or archived test output is
  embedded as a task-specific solution in the agent. Diagnostic fixtures stay outside ZIP.

## Deliberate limits

Up to 20 changed files, 1 MB per file (100 KB for full writes), 12 edit attempts, five seconds per probe
or selected regression test, and unchanged 40-tool/48-turn/four-minute declared budget.
These are conservative initial limits, not tuned performance claims. Large-file rewrites,
intentional definition deletion, slow tests, import-time-only changes, and refactors with
no behavioral fail-to-pass probe may be rejected. For docs-only changes, an assertion on
actual document content is supported without a Python module import.

The gate does not install dependencies, modify tests/configuration, or use hidden tests.
It does not add graph/embedding tools in this revision. Testing a missing API should use
an assertion such as `assert hasattr(module, "required_name")` before calling it, so an
expected missing API is distinguished from an import/setup failure.

## Build and validate

From repository root:

```bash
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission20.py -v
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission20_pilot.py -v
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission20.py
```

The CPU suite needs PyYAML and pytest. With the official runtime installed, additionally run:

```bash
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission20_runtime.py -v
```

Pinned runtime: adk-submission 0.2.13, adk-eval-core 0.1.0, swegemma 0.2.11,
google-adk 1.36.1, google-genai 2.11.0. Compilation uses the freshly extracted ZIP, so
local Python caches cannot invalidate the submission. CI uploads the ZIP as an artifact.

## Real E4B pilot before another large run

Use an existing E4B inference server and the official task/snapshot/test environment:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/run_submission20_pilot.py \
  --tasks /path/to/tasks.jsonl \
  --snapshots /path/to/snapshots \
  --wheels /path/to/wheels \
  --model-url http://localhost:8000/v1 \
  --output /path/to/submission20-pilot
```

This standalone runner does not edit the testing notebook or submission records.
It first evaluates empty-patch baselines for five fixed development tasks. Expected
behavioral failures are allowed; missing dependencies, fixture recursion, and collection
errors stop the run before any model calls. Collection errors can also represent a missing
requested API: inspect them; the conservative gate requires resolving that ambiguity,
not changing assertions or silently dropping a task.

After baselines, the runner requires a real-model synthetic smoke with an independent
oracle, then runs all five tasks. It saves package hashes, raw results, test output,
patches, traces and `readiness.json`. Promotion requires all five resolved with nonempty
patches and no agent error. This strict initial gate can stay closed; it does not claim
the candidate is ready merely because CPU tests pass. It never launches a larger run.
The fixed tasks are a development gate, not an unbiased benchmark/generalization estimate.
