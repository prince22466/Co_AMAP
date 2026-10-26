# Submission 19 verification scope

The candidate retains E4B and direct native editing. It adds structural review without
making edit validity depend on the model constructing an assertion in the same tool call.

| Submission 18 finding | Submission 19 verification |
| --- | --- |
| requests_7433 left an extra `):` | Replays the exact archived replacement against a valid minimal enclosing function; the audit blocks its invalid syntax. |
| fastapi_15280 inserted a decorator in constructor parameters | Minimal structural regression must return blocked. |
| fastapi_15588 erased ServerSentEvent | Valid-syntax class-erasure regression must report the removed definition and review_required. |
| fastapi_15661 edited test runners/config | Protected-path regressions flag pytest.ini, scripts/test.sh, scripts/coverage.sh, tests and pyproject.toml. |
| requests_7427 repeated unsuccessful exact replacements | Archived example retained; revised prompt requires rereading after one failure and abandoning guesses after two. This behavior is not hard-enforced or proven by CPU tests. |
| Guessed tests/test_models.py | CPU preflight rejects nonexistent test paths before test execution. Agent test discovery remains model-driven. |
| Possible installed-package import / environment errors | Preflight rejects installed json as checkout evidence; separately verifies a local src-layout package and its discovered pytest test. No claim that Kaggle dependencies are repaired. |
| Whitespace check described as verification | Audit never returns behavioral PASS; prompt requires a separate output-sensitive check. Actual task relevance is not automatically judged. |

`failures.json` contains five source-trace excerpts. A test verifies their SHA-256 and
exact arguments/results against submission-18-zip_First15Tasks.zip. Excerpts never enter
the candidate ZIP or model input. They are observed agent failures, not gold solutions.

Executed locally:

- 15 CPU regressions passed (including actual archived replacement replay and provenance).
- 2 official-runtime tests passed: compiler/native-tools/thinking configuration and packaged
  Runner + sandbox skill execution with failure, audit, repair, behavioral recheck and submission.
- Packaged scripted smoke passed with 9 counted calls and 12 scripted model calls, detecting
  the broken candidate and independently checking the final output.
- GitHub CPU CI runs portable regressions and verifies reproducible ZIP/manifest generation.
  The official-runtime tests require the competition wheelhouse and are run separately.

No real E4B smoke or task benchmark has been run. Audits detect structural errors when called;
they do not intercept native edits/submissions. Correctness, test relevance, repair decisions,
retry discipline and checking before submission still depend on model behavior. No guarantee
that Submission 18's 0/15 score is improved is made.
