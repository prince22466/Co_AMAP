# Submission 19 — direct edits with structural review

E4B (`gemma-4-e4b-it-qat-w4a16-ct`), one agent, thinking disabled, declared budget
4 minutes / 40 counted tool calls / 48 model turns. Keeps Submission 18's native read,
edit, write, command and submit tools. No evaluator/notebook/submission-record changes.

Submission 18 produced patches but introduced invalid syntax, erased a class, modified
test runners, guessed nonexistent test paths and called whitespace checks verification.
This candidate targets those failures with a bounded read-only patch audit and revised
repair/test instructions. It has not been evaluated with real E4B inference.

## What changes

- Edits still use the official native tools. No assertion/module argument blocks editing.
- A small `patch-audit` skill has one script and takes **no arguments**. It locates the
  checkout, checks changed Python syntax, reports removed top-level classes/functions,
  flags known protected test/configuration paths and checks diff whitespace. It includes
  untracked files and reports an empty patch. It never executes repository code or edits it.
- Removed definitions return `review_required`: legitimate API removals are possible.
  Known protected paths or syntax errors return `blocked`. These are audit findings, not
  interception of the native edit_file or submit_patch tools. The model must act on them.
- The audit always reports `behavior_verified: false`. A clean audit does not mean the
  task is implemented. The agent must discover an existing focused test or assert actual
  task output, verify checkout import provenance, and recheck after its final edit.
- Revised instructions explicitly reject git diff --check as behavioral verification,
  invented test paths, broad pytest as the first check, test-runner/environment edits,
  repeated exact-match guessing and unconnected new API parameters/middleware.
- The optional CPU preflight checks a CLEAN task checkout, imports the selected module
  with root/src and root on PYTHONPATH, rejects installed-package origins, and runs one
  real selected baseline test. It diagnoses missing dependencies/fixtures; it does not
  install them, rewrite tests, or remove failed tasks from the denominator.

## Limits

The same skill dispatcher that E4B struggled with in Submission 17 is used only for a
fixed audit call, never for source navigation/edits. It may still be called incorrectly;
the prompt specifies one correction and a native-command fallback. This risk needs a
real-model smoke, not another claim based on scripted calls.

No host callback forces an audit, relevant behavioral test, submission, or early stopping.
No automatic rollback or automatic semantic validation is claimed. Native tools remain
available and can bypass these instructions. The audit is a reliability aid, not a security
boundary. Legitimate task-specific configuration changes need manual review. Graph tools
and a separate verifier agent are not added. More correct patches are a hypothesis to test.

## Build and test

From the repository root:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission19.py
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission19.py -v
# With the competition runtime installed:
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission19_runtime.py -v
python Co_Kaggle/g6/tests_on_tasks/tools/smoke_submission19.py --output-dir /tmp/submission19-smoke
```

CPU tests require PyYAML and pytest. Official-runtime tests use adk-submission 0.2.13,
adk-eval-core 0.1.0, google-adk 1.36.1, google-genai 2.11.0 and swegemma 0.2.11 plus
runtime dependencies. The CLI smoke builds/extracts the exact ZIP and uses a scripted
model: failing behavior -> deliberately broken syntax -> audit blocker -> repair ->
passing behavior -> clean structural audit -> nonempty submission. It is NOT inference.

CI publishes `submission-19-e4b-unscored`. Extract the artifact and upload the INNER
`submission_19.zip`. The committed manifest records all five packaged files and ZIP hash.
Regression excerpts from Submission 18, tests and smoke oracles stay outside the agent ZIP.

## Before another GPU benchmark

On each representative clean task checkout with the intended interpreter/dependencies:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/preflight_submission19.py \
  --workspace /path/to/clean/checkout --module actual_package \
  --test tests/actual_discovered_test.py --python /path/to/environment/bin/python
```

Choose an existing baseline test, not a guessed filename. If it fails, classify the
failure and repair the local test environment separately; do not assume the agent patch
caused it or drop the task. One passing test is not a complete environment certification.
The utility runs imports/tests, which may have side effects; it does not edit their configuration.

For a real E4B smoke, make build_submission19.py and smoke_submission19.py importable in
Kaggle after starting the model registry/server, then run:

```python
from smoke_submission19 import run_smoke
report = await run_smoke('/kaggle/working/submission19_smoke',
                         models=models, submission_dir=AGENT_DIR)
assert report['passed'], report
```

`AGENT_DIR` is the extracted Submission 19 ZIP. On failure inspect events.json and stop
before a 15/30-task run. A passing smoke exercises mechanics only. For benchmark comparison
record the actual budget and compaction settings; use the declared four-minute limit,
40 calls / 48 turns, no compaction and the same task IDs. Existing notebooks remain untouched.
