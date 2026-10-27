# Gemma 4 public-development quick task sets: validated 15 and 30

**Source dataset:** [`tasks.jsonl`](https://github.com/prince22466/Co_AMAP/blob/main/Co_Kaggle/g6/competition_material/competition_data/tasks.jsonl), 129 public tasks (67 FastAPI / 48 Rich / 13 Requests / 1 HTTPX).

**Evidence:** [57-candidate CPU screening](https://github.com/prince22466/Co_AMAP/actions/runs/40563466553); [machine-readable per-task results](screening_evidence_20261027.json).

The **15-task set is an exact ordered subset of the 30-task set**: same first 15 task IDs. All selected tasks were **red on the base commit, green with the public reference patch** on their task-specific changed tests in a Python 3.12 Ubuntu GitHub Actions run. The test runner also rejected package shadowing by verifying the imported library path points into the checkout.

## Quick 15: iterate here first

**Composition:** 12 Rich, 2 Requests, 1 FastAPI. Mixed local edit sizes: one-line repairs, mid-sized changes, and multi-file edits. The issue domains include CLI streaming/output, HTTP request handling, dependency router invariants, rendering and formatting, and error presentation.

| # | Instance ID | Repo | Task focus | Gold patch chars | Test file(s) |
|---:|---|---|---|---:|---|
| 1 | `fastapi_14258` | fastapi |  Show a clear error on attempt to include router into itself | 489 | `test_router_circular_import.py` |
| 2 | `requests_7315` | requests | Preserve leading slashes in request path_url | 433 | `test_adapters.py` |
| 3 | `requests_7205` | requests | Fix empty netrc entry usage | 411 | `test_utils.py` |
| 4 | `rich_4077` | rich | proxy isatty | 231 | `test_file_proxy.py` |
| 5 | `rich_4076` | rich | preserve newlines | 714 | `test_ansi.py` |
| 6 | `rich_4075` | rich | empty with end | 1,413 | `test_console.py` |
| 7 | `rich_3718` | rich | fix(panel): fix title missing panel background | 1,131 | `test_panel.py` |
| 8 | `rich_3934` | rich | empty live | 1,446 | `test_live.py` |
| 9 | `rich_3882` | rich | Fix raw markup printed on prompt errors sometimes | 459 | `test_prompt.py` |
| 10 | `rich_3905` | rich | Update progress.py | 540 | `test_progress.py` |
| 11 | `rich_3518` | rich | fix(table): highlight columns added by add_row | 494 | `test_table.py` |
| 12 | `rich_3454` | rich | Fix [BUG] @ breaks highlighting in hyperlink #3327 | 569 | `test_highlighter.py` |
| 13 | `rich_3938` | rich | fix background style with soft wrap | 2,420 | `test_segment.py, test_text.py` |
| 14 | `rich_3676` | rich | Notes | 2,874 | `test_traceback.py` |
| 15 | `rich_3052` | rich | Case insensitive prompt choices | 5,149 | `test_prompt.py` |

## Extended 30: expand only after 15-task regression

Adds three Requests tasks and twelve Rich tasks to cover URL normalization, proxy/Content-Type parsing, terminal graphemes, Unicode width, segments, console capture, markup escaping, and state handling.

| # | Instance ID | Repo | Task focus | Gold patch chars | Test file(s) |
|---:|---|---|---|---:|---|
| 16 | `requests_7309` | requests | Fix malformed value parsing for Content-Type | 1,194 | `test_utils.py` |
| 17 | `requests_7427` | requests | Port bpo-39057 to Requests | 685 | `test_utils.py` |
| 18 | `requests_6644` | requests | Trim excess leading path separators | 435 | `test_adapters.py` |
| 19 | `rich_3953` | rich | Fix ZWJ and edge cases | 1,988 | `test_cells.py` |
| 20 | `rich_3944` | rich | fix fonts | 394 | `test_cells.py` |
| 21 | `rich_3935` | rich | fix for padding width | 1,880 | `test_columns.py, test_table.py` |
| 22 | `rich_3777` | rich | TTY_INTERACTIVE env var | 1,358 | `test_console.py` |
| 23 | `rich_3535` | rich | Regex error | 1,001 | `test_cells.py` |
| 24 | `rich_3521` | rich | Segment.split_cells fix | 1,953 | `test_segment.py` |
| 25 | `rich_3471` | rich | Fixed exception in append_tokens | 350 | `test_text.py` |
| 26 | `rich_3470` | rich | Fix record and capture | 378 | `test_console.py` |
| 27 | `rich_3278` | rich | Strip problematic private escape sequences | 176 | `test_ansi.py` |
| 28 | `rich_2943` | rich | Clear hashed cache when clearing meta. | 354 | `test_style.py` |
| 29 | `rich_3180` | rich | Fix double-width characters disappearing when wrapping | 6,427 | `test_cells.py, test_text.py` |
| 30 | `rich_3063` | rich | Fix escaping trailing backslash | 549 | `test_markup.py` |

**30-task composition:** 24 Rich, 5 Requests, 1 FastAPI. All 30 passed the local red→green check.

The 30-set intentionally includes both `requests_6644` (trim excess separators) and `requests_7315` (preserve leading separators). These use **different pinned historical base commits and tests**, and are not interchangeable: the intended behavior changed with context. Agents must solve each supplied task rather than applying a universal rule.

## Inclusion/exclusion criteria

1. Tasks must come verbatim from the public `tasks.jsonl` with unique `instance_id` and pinned `base_commit`.
2. Favor small or medium focused source changes, dedicated tests, independent modules, and diverse editing surfaces. Include several multi-file/nontrivial cases so the agent is not evaluated only on one-liners.
3. Verify `test_patch` applies cleanly on the unmodified base and the same tests **fail** before the gold `patch` but **pass** afterward.
4. Verify imports resolve to the historical checked-out source, especially the Requests `src/` layout. Do not accept wheel-shadowed passes.
5. Exclude tasks with test dependency/fixture collection errors, gold-patch verification failures, and baseline hangs in this screen. Exclude `rich_3468`, for which community testing reported an empty-patch pass.
6. Be deliberately Rich-heavy: FastAPI has widely reported public-environment dependency problems, and only `fastapi_14258` of the screened simple FastAPI cases passed the exact local check. These selections prioritize **usable quick-regression signal**, not representativeness of the hidden benchmark.

**Measured screening:** 57 candidates → **37 red→green**, **17 gold failures**, **3 base timeouts**. From the 37 we selected 30. Those checks required no GPU inference. Some gold failures can be version-specific, and should not be construed as intrinsically invalid public tasks.

## Run with exactly these task IDs

The files are [`quick_15_ids.json`](quick_15_ids.json) and [`quick_30_ids.json`](quick_30_ids.json), ordered arrays of IDs. The [materializer](../tools/materialize_quick_task_set.py) produces intact JSONL records without changing patches, tests, descriptions, or commits:

```bash
DATA=Co_Kaggle/g6/competition_material/competition_data/tasks.jsonl
SETS=Co_Kaggle/g6/tests_on_tasks/quick_sets
SCRIPT=Co_Kaggle/g6/tests_on_tasks/tools/materialize_quick_task_set.py

python "$SCRIPT" "$DATA" "$SETS/quick_15_ids.json" /tmp/g6_quick15.jsonl
python "$SCRIPT" "$DATA" "$SETS/quick_30_ids.json" /tmp/g6_quick30.jsonl
```

Use `/tmp/g6_quick15.jsonl` (then `/tmp/g6_quick30.jsonl`) as the input `tasks` file in the existing evaluation notebook/Evaluator while keeping the **same model, submission, maximum tool calls, timeout policy and harness versions**. Each task's benchmark `patch` and `test_patch` remains in the evaluator dataset and must not be revealed to the agent during inference.

Collect task-level `resolved`, patch size, test exit code, tool calls, time, test stdout and agent traces. Report the number **resolved**, not merely nonempty patches.

## Interpret the result correctly

This is a **local CPU test-file smoke**, not official SWE-gemma hermetic verification or a leaderboard score. It does not prove each gold patch will pass all official validation tests, nor that the original public dataset is fully sound. Dependencies were installed via pip in GitHub Actions (not an organizer-pinned wheelhouse); terminal formatting tests can differ across dependency versions. The source task patches were unmodified except addition of their missing terminal newline before invoking `git apply`.

For competition-parity conclusions, validate the same selection with the **actual pinned SWE-gemma runtime and gold patches**, record baseline and patched outcomes, and remove any task that fails the official gold check. The hidden test set uses other tasks/repositories, so this Rich-heavy panel is intended for **iteration and regression detection**, not a performance estimator of hidden leaderboard score.
