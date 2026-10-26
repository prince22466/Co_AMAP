# GPT-6 independent first-15 task diagnostics (development only)

**Status: exploratory candidate patches, NOT solved/graded results.**

The task set is the first 15 entries in the public Gemma 4 Developer Agent Competition `tasks.jsonl`. Each task starts from the exact `base_commit` in [manifest.json](manifest.json). The source was inspected at that commit. These patches were **independently drafted from task statements and source**, without using the provided reference `patch` or `test_patch` as solver inputs.

## Scope

| # | Task | Problem, in brief | Independent candidate |
|---:|---|---|---|
| 1 | fastapi_15661 | Automate FastAPI release preparation | Not yet drafted |
| 2 | fastapi_15588 | Prevent invalid newline/null SSE fields | [Candidate](candidates/fastapi_15588.patch) |
| 3 | fastapi_15589 | Only accept hyphen-style header alias by default | [Candidate](candidates/fastapi_15589.patch) |
| 4 | fastapi_15030 | Implement Server-Sent Events end to end | Not yet drafted |
| 5 | fastapi_14962 | Pydantic-core JSON response serialization | [Candidate](candidates/fastapi_14962.patch) |
| 6 | fastapi_15280 | Introduce `@app.vibe()` | Not yet drafted |
| 7 | fastapi_14964 | Deprecate `UJSONResponse` and `ORJSONResponse` | [Candidate](candidates/fastapi_14964.patch) |
| 8 | fastapi_15023 | Update yield/streaming-response documentation | Not yet drafted |
| 9 | fastapi_14986 | Avoid OpenAPI schema mutation; safely escape Swagger HTML | [Candidate](candidates/fastapi_14986.patch) |
| 10 | fastapi_14978 | Add JSON `strict_content_type` checking | Not yet drafted |
| 11 | requests_7505 | Handle runtime protocols with `__getattr__` proxies | [Candidate](candidates/requests_7505.patch) |
| 12 | requests_7502 | Recognize proxied file-like multipart uploads | [Candidate](candidates/requests_7502.patch) |
| 13 | requests_7433 | Detect streams hidden behind `__getattr__` | [Candidate](candidates/requests_7433.patch) |
| 14 | requests_7427 | Respect domain boundaries in `no_proxy` | [Candidate](candidates/requests_7427.patch) |
| 15 | requests_7315 | Preserve leading double slashes in URL paths | [Candidate](candidates/requests_7315.patch) |

## Validation and constraints

- `verify_patch_structure.py` fetches each exact public upstream base commit, applies the candidate patch, and runs Python AST syntax validation. **This is a structural check only.**
- Official Kaggle task results, behavioral test outcomes and tool-call traces are **not available from this offline interaction**. No task is marked resolved on the basis of producing a patch.
- Repository inspection in this draft used the GitHub connector to retrieve the pinned source. This is **not** an official SWE-gemma `read_file` tool call; therefore it is **not a matched-tool GPT-6 vs E4B benchmark**. True apples-to-apples comparison must run inside the competition harness and use only its tools.
- Published reference patches and `test_patch` exist in the public dataset. They were deliberately **not used to generate these candidates**. Keep any separate gold inspection in an independent evaluator.
- Existing S18–S20 0/15 results are not being re-scored by this PR.

## Next experiment

1. Run official baseline (unpatched) tests, then validate candidates under pinned docker and wheel versions.
2. Obtain official `resolved`, patch and tool-call outcomes for GPT-6 tool-equivalent runs and Gemma S21, using identical task IDs and evaluator limits.
3. Compare first divergence and classify: understanding, navigation, tool calls, editing, semantic implementation, verification, submission and environment.
4. Do not promote patches to a competition agent or claim any leaderboard improvement without real-model evidence.

## Preliminary focused public-test replay

The [first completed replay](https://github.com/prince22466/Co_AMAP/actions/runs/40411339643) normalized missing final newlines in published test diffs, applied the public verification test *only in the evaluator*, and ran identical focused tests on the base commit and patched checkout (Python 3.11/GitHub Actions, not the official Kaggle Docker image).

| Task | Base | Candidate | Interpretation |
|---|---:|---:|---|
| requests_7427 | exit 1 | exit 0 | **Focused public test passes** after independent patch |
| requests_7315 | exit 1 | exit 0 | **Focused public test passes** after independent patch |
| requests_7505 | exit 0 | exit 0 | Inconclusive: reproducer passes before patch |
| requests_7502 | exit 0 | exit 0 | Inconclusive: reproducer passes before patch |
| requests_7433 | timeout | not run | Blocked: 120-second baseline fixture/test timeout |

These are **not official resolved outcomes**; no full regression suite was run, and the task-specific public test may be insufficient. Do not count the 2 focused test passes as an official 2/15 score. A follow-up workflow adds 45-second per-test timeouts to keep fixture problems bounded.

