# Submission 22 — native edits plus bounded repair and check evidence

Derived from Submission 21 (Gemma 4 E4B). A single agent keeps native
`read_file`, `edit_file`, `write_file`, `run_command`, `get_status`
and `submit_patch`. It does **not** reproduce Submission 20's mandatory
skill-only candidate workspace (which resulted in zero patches on its 15-task
run).

## Evidence and objectives

Submission 21 solved **2/15 selected S1 tasks** (rich_4077 and rich_3905);
11/15 returned nonempty patches. The same candidate resolved **0/15** in the
older first-15 run, though it returned 15/15 patches. The task populations are
different: the 2/15 figure is *not* evidence of improvement on matched tasks.

Observed S1 issues motivating S22:
- repeated failed exact edits on `requests_7315` and `rich_3052`;
- `rich_4075` and `rich_3934` reported successful edits followed by empty submissions;
- `rich_3454` finished without a native submission, requiring a fallback;
- `rich_3676` and `rich_3052` submitted fixes after running existing tests,
  but both returned unresolved.

### Priority 1: deterministic edit recovery

The optional `patch-audit/scripts/recover.py` skill accepts an **exact,
uniquely matching** replacement on the live workspace, returns nearby numbered
context when the anchor does not match, and suppresses the second identical
failed anchor on the same file contents. It checks Python syntax and removed
top-level definitions before changing files, refuses test/evaluator paths,
and rolls back edits that fail `git diff --check`. No fuzzy modification or
whole-file replacement is attempted.

Crucially, **this is an optional fallback, not an intercept** for native
`edit_file`. The model can still ignore the helper or repeat native failures.
The prompt gives bounded next actions while native tools remain available.

### Priority 2: protect final patch preservation

The prompt requires inspecting `git status --short`,
`git diff HEAD --stat`, `git diff HEAD` and `git diff --check` before
submission, and requires checking the returned `patch_size`. The optional
read-only audit reports an empty patch, changed paths, syntax errors, removed
definitions, protected test/evaluator paths and a content digest.
Task-authorized `pyproject.toml` changes are **advisory**, not automatically
blocked, because real tasks may require configuration edits.

This does **not** make `submit_patch` an enforced gated operation. A native
submission can bypass the audit. Patch capture by the competition evaluator
must still be verified from actual trace and returned patch.

### Priority 3: task-specific checks

The prompt requests a meaningful baseline check before editing where feasible
and a re-run of **the same** check after the final edit. The optional
`scripts/verify.py` runs a nontrivial Python assertion in the local checkout,
checks the requested module's import provenance, rejects empty patches and
trivial always-true assertions, and reports checkout mutation. It explicitly
returns `baseline_verified=false` and `behavior_verified=false`: a passing
candidate assertion alone cannot prove the issue was fixed or the probe was
relevant. A human must examine the actual behavior and evaluator results.

If dependency or fixture setup fails, the agent must mark the behavior
`UNVERIFIED`, never edit tests to make them pass, and submit only legitimate
source changes. There is no claim that missing `inline_snapshot` or recursive
`httpbin` fixture problems are solved.

## Scope, rules, caveats

Only the competition's permitted native tools and an ADK skill are used.
No network, hidden grader access, replacement test suite, alternate model,
model mixing or new inference stage is required. S22 uses E4B, a 40-call
budget, 48 turns and the same sampling configuration as S21. Agents are
not given gold patches or hidden benchmark tests.

The native tool signatures do not support intercepting calls in this YAML
submission, so tool-usage discipline and final pre-submit checks are advisory
unless a helper is *actually invoked*. This is a deliberately smaller,
testable architecture rather than an unsupported mandatory gate.

## Build and CPU tests

From repository root:

```bash
python -m pip install 'PyYAML>=6,<7'
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission22.py -v
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission22.py
```

Output: `Co_Kaggle/g6/submission_nb/artifacts/submission_22.zip` and
`submission_22.manifest.json` (also emitted by the PR's GitHub Actions job).

CPU tests exercise real temporary Git checkouts for exact/ambiguous anchors,
repeated failed edits, rollback, test-protection, patch emptiness/digests,
legitimate config changes, local imports, nontrivial assertions, unverified
candidate checks, and mutation detection. They do **not** establish that E4B
will use the helpers, improve tool-call behavior or solve more tasks.

## Real-model decision gate (before a broad GPU run)

1. Run one controlled S1 pilot with the **same IDs, model, runtime and
   evaluation environment** used for S21.
2. Preserve the returned diffs and interaction traces. Count successful
   source edits, identical failed edit retries, empty returned patches,
   baseline/post-edit checks, and resolution.
3. Inspect any task-specific dependencies and evaluator fixtures independently;
   do not classify `resolved=False` alone as a model error.
4. Promote only after code-level review and a matched-task win, not on
   patch-production count or passing unrelated tests alone.

There is **no real E4B inference or measured score for S22 yet**.
