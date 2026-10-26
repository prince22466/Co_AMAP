# GPT-6 independent first-15 task diagnostics (development only)

**Status: exploratory candidate patches, NOT solved/graded results.**

The task set is the first 15 entries in the public Gemma 4 Developer Agent Competition `tasks.jsonl`. Each task starts from the exact `base_commit` in [manifest.json](manifest.json). The source was inspected at that commit. Initial patches were independently drafted from task statements and base-commit source. Later revisions were deliberately **guided by the published development `test_patch` and test results**. No reference solution `patch` was used as code input. Consequently these are not blind GPT-6 benchmark solutions.

## Scope

| # | Task | Problem, in brief | Independent candidate |
|---:|---|---|---|
| 1 | fastapi_15661 | Automate FastAPI release preparation | [Candidate](candidates/fastapi_15661.patch) |
| 2 | fastapi_15588 | Prevent invalid newline/null SSE fields | [Candidate](candidates/fastapi_15588.patch) |
| 3 | fastapi_15589 | Only accept hyphen-style header alias by default | [Candidate](candidates/fastapi_15589.patch) |
| 4 | fastapi_15030 | Implement Server-Sent Events end to end | [Candidate](candidates/fastapi_15030.patch) |
| 5 | fastapi_14962 | Pydantic-core JSON response serialization | [Candidate](candidates/fastapi_14962.patch) |
| 6 | fastapi_15280 | Introduce `@app.vibe()` | [Candidate](candidates/fastapi_15280.patch) |
| 7 | fastapi_14964 | Deprecate `UJSONResponse` and `ORJSONResponse` | [Candidate](candidates/fastapi_14964.patch) |
| 8 | fastapi_15023 | Update yield/streaming-response documentation | [Candidate](candidates/fastapi_15023.patch) |
| 9 | fastapi_14986 | Avoid OpenAPI schema mutation; safely escape Swagger HTML | [Candidate](candidates/fastapi_14986.patch) |
| 10 | fastapi_14978 | Add JSON `strict_content_type` checking | [Candidate](candidates/fastapi_14978.patch) |
| 11 | requests_7505 | Handle runtime protocols with `__getattr__` proxies | [Candidate](candidates/requests_7505.patch) |
| 12 | requests_7502 | Recognize proxied file-like multipart uploads | [Candidate](candidates/requests_7502.patch) |
| 13 | requests_7433 | Detect streams hidden behind `__getattr__` | [Candidate](candidates/requests_7433.patch) |
| 14 | requests_7427 | Respect domain boundaries in `no_proxy` | [Candidate](candidates/requests_7427.patch) |
| 15 | requests_7315 | Preserve leading double slashes in URL paths | [Candidate](candidates/requests_7315.patch) |

## Validation and constraints

- `verify_patch_structure.py` fetches each exact public upstream base commit, applies the candidate patch, and runs Python AST syntax validation. **This is a structural check only.**
- Focused public development-test outcomes are available below. Official Kaggle `resolved` results and comparable GPT-6 tool traces are **not available**. `verify_official_phase2.py` is an opt-in attempt to invoke the SWE-gemma Phase 2 harness, but has not been run in Kaggle's Docker/snapshot/wheelhouse environment.
- Repository inspection in this draft used the GitHub connector to retrieve the pinned source. This is **not** an official SWE-gemma `read_file` tool call; therefore it is **not a matched-tool GPT-6 vs E4B benchmark**. True apples-to-apples comparison must run inside the competition harness and use only its tools.
- Public `test_patch` tests were inspected after initial draft attempts to diagnose and repair candidates. Published gold `patch` solutions were not used. This is test-guided coding, not a clean blind comparison. Do not package development-specific fixes in the Gemma competition submission.
- Existing S18–S20 0/15 results are not being re-scored by this PR.

## Next experiment

1. Run official baseline (unpatched) tests, then validate candidates under pinned docker and wheel versions.
2. Obtain official `resolved`, patch and tool-call outcomes for GPT-6 tool-equivalent runs and Gemma S21, using identical task IDs and evaluator limits.
3. Compare first divergence and classify: understanding, navigation, tool calls, editing, semantic implementation, verification, submission and environment.
4. Do not promote patches to a competition agent or claim any leaderboard improvement without real-model evidence.

## Development test evidence

- [GitHub Actions run 40415216582](https://github.com/prince22466/Co_AMAP/actions/runs/40415216582): **13 of 15** public focused checks passed baseline-fail -> patched-pass: all five Requests and eight FastAPI. Remaining FastAPI: `fastapi_15030` SSE keepalive/OpenAPI, and `fastapi_14962` custom-response tutorial/fast path.
- Additional changes to both remaining candidates are under CI review. Status must be updated only from completed job logs.
- The focused development checks run on **generic GitHub Actions Python 3.13**, with manually installed public dependencies; they are **not identical to Kaggle's pinned Docker/wheelhouse verifier**.
- A public test may pass without the entire repository regression suite passing. A successful focused test must not be represented as an official `resolved=True` score.

### Competition Phase 2 reproduction

Run the opt-in [Phase 2 verifier](verify_official_phase2.py) **inside Kaggle** with the official installed SWE-gemma runtime, Docker image, task snapshots, and offline wheels, for example:

```bash
python Co_Kaggle/g6/tests_on_tasks/diagnostics/gpt6_first15/verify_official_phase2.py \
  --tasks /path/to/tasks.jsonl \
  --snapshots /path/to/snapshots \
  --wheels /path/to/wheels \
  --submission-dir /path/to/sample_submission \
  --output /kaggle/working/gpt6_first15_phase2
```

This script is **unexecuted against the actual SWE-gemma package versions** and may require API adaptation. The path arguments must point to official existing resources. Do not report it as verified unless it executes successfully and returns individual results.

The comparison to E4B can only be called matched-tool if both models use the official nine-tool SWE-gemma harness under the same resource limits. This repository work used GitHub content operations and external GitHub Actions and therefore is **not** that comparison.
