# Submission 21 — first-15 validation root-cause review

**Date:** 2026-10-27  
**Model:** `gemma-4-e4b-it-qat-w4a16-ct`  
**Scope:** 15 fixed SWE-gemma development tasks, archived real-model evaluation.  
**Headline:** **0/15 resolved, 15/15 nonempty patches.** Not a leaderboard score.

## Evidence and comparison method

Primary: `submission-21-zip_First15Tasks.zip`, particularly `raw_results/<task>.json`,
`test_outputs/<task>_returned.log`, `patches/<task>_returned.patch`, and
`selected_tasks.json`. The per-task source is not the abbreviated console output.

**No-patch control:** `submission-20-zip_First15Tasks.zip`. Automated archive comparison confirms
the **same 15 instance IDs, repos, base commits, and problem statements**, and
**0/15 nonempty control patches**. Only **9/15 control tasks have test output**:
the other six have exit code `-1` and no useful validation logs.
Thus S20 is a *no-returned-patch control*, not a full baseline rerun; control
verification is missing for six tasks. Some S20 skill operations also failed,
but no mutation was returned.

Historical comparison: `submission-19-zip_First15Tasks.zip`. It helps confirm
recurring fixture/dependency error signatures for cases missing an S20 verifier
output. S19 had candidate patches, so **S19 is not an unmodified baseline**.

Commands: the companion script
`tests_on_tasks/tools/inspect_submission21_archive.py`
extracts every task's exact reported test exit, key failure messages, files
changed and test-diff comparisons. It also asserts snapshot identity and
zero patches in the S20 control. The GitHub Actions job runs this analysis
**without using any GPU or model inference**.

**Classification rule:** "Environment-blocked" means the observed patched run
could not prove functionality because collection/fixture setup failed; it does
**not** mean the patch is correct. "Implementation/API" indicates concrete
missing required entry points. "Behavioral" indicates collected tests ran
and reported behavioral assertions. "Syntax" indicates code fails at import
due to a Python parsing error. Each task has one *primary* classification;
multiple causes can coexist.

## All 15 task diagnoses

| Task | S20 no-patch control | S21 patched verifier | Primary classification | Evidence and interpretation |
|---|---|---|---|---|
| `fastapi_15661` | Exit 2; missing `scripts.prepare_release` | Exit 2; same import error | **Implementation/API** | S21 edits only `scripts/add_latest_release_date.py`, not the required `scripts.prepare_release` module. A missing required module persists from the control; release-preparation API is not implemented. |
| `fastapi_15588` | 6 failed / 18 passed | 6 failed / 18 passed | **Behavioral** | SSE `event`/`id` must reject `\n`, `\r`, and `\r\n` with a matching field-specific message. S21 adds validators but produces messages like "SSE field cannot contain newlines" instead of the expected "SSE 'event' must be a single line"; CR handling is incomplete. Observable change in failure mode, **not a solved case**. |
| `fastapi_15589` | No control test output (`-1`) | Exit 2; `IndentationError` | **Syntax** | Submitted `fastapi/openapi/utils.py` has `unexpected indent` at line 164. Patch inserted new parameter logic mid-expression and left stray old lines. This is an S21-submitted syntax defect. |
| `fastapi_15030` | Exit 2; 6 collection errors | Exit 2; same 6 errors | **Implementation/API**, plus environment | `EventSourceResponse` is not importable from `fastapi.responses`, and several tests also lack `inline_snapshot`. S21 edits `fastapi/__init__.py` and `fastapi/responses.py`, but still does not expose the required symbol. |
| `fastapi_14962` | Exit 2; 2 collection errors | Exit 2; same 2 errors | **Environment-blocked** | Both reports fail importing `inline_snapshot` in custom-response tutorial tests. The S21 `fastapi/responses.py` patch cannot be behaviorally assessed by these results. S19 has the same dependency error. |
| `fastapi_15280` | No control test output (`-1`) | Exit 2; `SyntaxError` | **Syntax** | `fastapi/applications.py:102`: unterminated string literal. The 10,683-byte S21 patch inserts `def vibe(...)` inside existing class/docstring structure, breaking module parsing. |
| `fastapi_14964` | No control test output (`-1`) | Exit 2; 2 collection errors | **Environment-blocked** | `inline_snapshot` absent from custom-response tutorial collection; same signature in S19. S21 deprecation-related response changes remain unverified. |
| `fastapi_15023` | Exit 2; 1 collection error | Exit 2; same 1 error | **Environment-blocked** | `inline_snapshot` unavailable for streaming-data tutorial test. The SSE/streaming response changes were never behaviorally exercised by the returned verifier. |
| `fastapi_14986` | 5 failed / 2 passed | 4 failed / 3 passed | **Behavioral**, mixed improvement/regression | Two previous failures disappear (`test_configured_servers_not_mutated` and `test_multiple_different_root_paths_do_not_accumulate`), but **`test_legitimate_root_path_still_appears` newly fails**. S21 replaces `self.servers.insert(0, {"url": root_path})` with `pass`, suppressing valid root-path output. Swagger UI escaping also still fails. Net −1 failing test is **not** a complete improvement. |
| `fastapi_14978` | Exit 2; 3 collection errors | Exit 2; same 3 errors | **Implementation/API**, plus environment | `APIRouter.__init__()` still rejects `strict_content_type=`; two related tests fail during collection. A third test is blocked by missing `inline_snapshot`. S21 adds middleware but misses the router API contract. |
| `requests_7505` | No control test output (`-1`) | Exit 1; 197 setup errors | **Environment-blocked** | Recursive `httpbin`/`httpbin_secure` fixtures in `tests/conftest.py`, as also seen in S19. S21 changes `src/requests/models.py`; its intended file-like-body behavior remains unverified by this run. |
| `requests_7502` | Exit 1; 196 setup errors | Exit 1; same 196 errors | **Environment-blocked** | Recursive `httpbin` fixture; both control and S21 return 142 passed, 1 skipped, 1 xfailed, 196 errors. S21 `src/requests/_types.py` changes have no credible post-patch functional pass evidence. |
| `requests_7433` | No control test output (`-1`) | Exit 1; 195 setup errors | **Environment-blocked** | Recursive `httpbin` fixtures also observed in S19. Changes to `src/requests/sessions.py` may be incomplete, but archived tests cannot decide under these setup errors. |
| `requests_7427` | 7 failed / 209 passed / 13 skipped | Same 7 failed / 209 passed / 13 skipped | **Behavioral** | Identical utility failures: empty netrc defaults, multipart content-type parsing, encoding, and `NO_PROXY` boundary matching. S21 changes only part of proxy-bypass matching in `src/requests/utils.py`. |
| `requests_7315` | No control test output (`-1`) | Exit 1; 1 failed | **Behavioral** | `HTTPAdapter.request_url` returns `/v:h` rather than required `//v:h`. S21 edits URL path handling in `src/requests/models.py`, while failing behavior is in `requests.adapters.HTTPAdapter.request_url`; the change misses the consumer. |

## Cause counts (primary, disjoint)

| Primary classification | Number | Tasks |
|---|---:|---|
| Environment-blocked, patch otherwise **unverified** | **6** | `fastapi_14962`, `fastapi_14964`, `fastapi_15023`, `requests_7505`, `requests_7502`, `requests_7433` |
| Submitted Python syntax/indentation defect | **2** | `fastapi_15589`, `fastapi_15280` |
| Missing required task API / wrong entry point | **3** | `fastapi_15661`, `fastapi_15030`, `fastapi_14978` |
| Observable behavioral test failures | **4** | `fastapi_15588`, `fastapi_14986`, `requests_7427`, `requests_7315` |
| **Total** | **15** | **0 resolved** |

**Overlapping environmental signals:** `fastapi_15030` and `fastapi_14978`
also show missing `inline_snapshot` alongside explicit unmet API contracts.
Thus **8/15** tasks show an environment blocker somewhere in test output,
but only **6/15** have environment as the primary *diagnosable* cause.
The remaining code defects are independently visible.

**Control coverage:** 9 tasks are compared to S20 no-patch test results;
the other 6 lack S20 verifier output. Repeated S19 signatures strengthen
but do not establish an unmodified baseline for those six.

## What changed relative to unpatched behavior?

- **Same failure set despite patch:** `fastapi_15661`, `fastapi_15030`,
  `fastapi_14962`, `fastapi_15023`, `fastapi_14978`, `requests_7502`,
  `requests_7427`: S20 and S21 have the same failures/errors (although
  the underlying source and failure detail may differ).
- **Partial progress with regression:** `fastapi_14986` 5 → 4 failures,
  with two eliminated and one new root-path assertion failure;
  `fastapi_15588` remains at six failures, but rejects some malformed
  SSE values with the wrong exception text.
- **No control test output:** `fastapi_15589`, `fastapi_15280`,
  `fastapi_14964`, `requests_7315`, `requests_7433`, `requests_7505`.
  The S21 code and verifier results still demonstrate two syntax defects,
  one functional failure and three recurring fixture/dependency blocks.

## Architecture vs model capability

**Agent-system/test-infrastructure issues**

1. FastAPI task fixtures require `inline_snapshot`; Requests tests use
   self-referential `httpbin` fixture declarations in the archived
   evaluator environment. This defeats behavioral validation even with a
   correctly targeted patch. Repair/verify the **external** sandbox
   dependency/fixture setup; do not edit task tests or the submitted
   library to manufacture success.
2. S21's read-only syntax audit is **optional** (only two audit calls
   appeared in the console, both for `fastapi_14962`). Nothing reliably
   prevented the two syntax-broken patches from submission.
3. S21 prompt guidance about final checks and retry budgets did not
   consistently produce focused passing tests. The harness can still
   accept a nonempty patch without demonstrated behavior.

**Model/agent planning and tool-use failures**

1. E4B inserted code into the wrong syntactic location (`fastapi_15280`)
   and left malformed indentation (`fastapi_15589`).
2. Some changes targeted adjacent code but missed the required public
   consumer (`requests_7315`, `fastapi_15661`, `fastapi_14978`).
3. The regression in `fastapi_14986` shows lack of a competing-invariant
   test: stop accumulating attacker-supplied root paths **without**
   hiding legitimate root paths.

These categories overlap: deterministic runtime checks can contain
model mistakes without claiming the model suddenly reasons correctly.

## Required next steps **before** another GPU-based 15-task evaluation

1. **Repair baseline fixtures externally and rerun baseline-only tests:**
   `fastapi_14962`, `fastapi_14964`, `fastapi_15023`,
   `requests_7505`, `requests_7502`, `requests_7433`.
   Also retest `fastapi_14978` and `fastapi_15030` for the separate
   unmet APIs after the dependency blocker is removed.
   Record per-task baseline exit code, failed node IDs and exact output.
2. **Run a post-edit parser/import check over changed Python files** in
   a read-only, deterministic stage. Specifically replay `fastapi_15589`
   and `fastapi_15280`. Preserve S21 native editing; do not recreate
   S20's mandatory skill-edit wrapper.
3. **Independent, behavior-sensitive tests** for `fastapi_14986`,
   `fastapi_15588`, `requests_7427` and `requests_7315`; tests should
   exercise both the requested behavior and unaffected invariants.
4. Run a small real-E4B regression on representative tasks **after**
   verifying environment and validation mechanics; report resolved
   cases, not patch production alone. The diagnostics used a **disabled
   per-task session time limit**, so reproduce strict deployment limits.

## Limitations

The archive stores patched-result test outputs and does not contain a
complete set of pristine baseline results. S20 supplies a verified
zero-returned-patch comparator only where its verifier actually ran.
No missing dependencies were installed and no official evaluator or
GPU inference was rerun for this analysis. A collection error alone
cannot adjudicate semantic correctness, and identical failure counts
cannot guarantee byte-for-byte equivalent tracebacks.

**Conclusion:** S21 repairs patch *production*, not patch *correctness*.
The evidence shows six environment-masked evaluations and nine cases
with an independent submitted syntax, unmet API, or behavioral issue.
Fix evaluator observability first, then enforce cheap post-edit syntax
checks while keeping native editing available to Gemma E4B.
