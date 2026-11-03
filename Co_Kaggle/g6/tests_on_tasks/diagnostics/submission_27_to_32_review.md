# Submissions 27–32: real E4B evaluation and lessons learned

**Scope:** the same **15 selected S1 tasks** tested with
`gemma-4-e4b-it-qat-w4a16-ct`. This is a retrospective **of recorded
diagnostic runs**, not a prediction of Kaggle public/private score.
It distinguishes **actual task resolution**, **patch production**, **tool
behavior**, and **evidence limits**. No candidate is declared better merely
because its README, unit tests, or intended architecture looks promising.

**Bottom line:** **S30 was the best observed score (4/15), but unsafe**:
it destructively overwrote a FastAPI source file. **S29 was the best
conservative reference (3/15)**: it recovered two wins lost in S28 with
bounded patches, although it still suffered frequent edit failures.
**S31's mandatory graph/multi-agent pipeline dropped to 1/15**; **S32's
guarded-recovery experiment recovered to only 2/15 without confirmed
successful guarded recovery**. None reached the stated >=5/15 target.

## 1. Evaluation protocol and caveats

All six printouts report the same 15 task IDs and the same model family.
They show `Per-task session time limit: DISABLED` and effective
`max_tool_calls: 40`, `max_turns: 48`. The packaged configurations generally
specify four minutes per task, **but that limit was not enforced by these
particular diagnostic runs**. These results therefore do **not** establish
four-minute Kaggle performance. In particular, several scored wins took
more than four minutes.

Each version has **one observed S1 run** in this review, not matched repeated
trials. These are exploratory comparisons with random/model variance; a
change in resolved count is not proof that one design change caused it.
The 15 selected tasks are a development screen, not an unbiased estimate of
the complete competition benchmark. Some earlier reviews also flag task
provenance ambiguities: verify the actual issue and grader before interpreting
a proposed source patch as correct.

The summary rows cover all 15 tasks. Timestamped *detailed* tool events
are incomplete for **S31 (13/15 shown)** and **S32 (14/15 shown)**, so their
event counts are **visible lower bounds**. Printed traces can elide
arguments; same-path repeat counts do not prove that exact parameters
were repeated. "Observed tokens" sums per-request token usage, including
repeated context; it is **not** unique generated tokens, GPU billing,
or a direct model-quality measure.

### Reproducible data sources

| Version | Original printout | Submission configuration |
| --- | --- | --- |
| S27 | [S27 printout](submission-27-zip_s1TasksPrintouts.txt) | [S27 prompt](../../submission_nb/submission_27/instruction.md) |
| S28 | [S28 printout](submission-28-zip_s1TasksPrintouts.txt) | [S28 prompt](../../submission_nb/submission_28/instruction.md) |
| S29 | [S29 printout](submission-29-zip_s1TasksPrintouts.txt) | [S29 prompt](../../submission_nb/submission_29/instruction.md) |
| S30 | [S30 printout](submission-30-zip_s1TasksPrintouts.txt) | [S30 prompt](../../submission_nb/submission_30/instruction.md) |
| S31 | [S31 printout](submission-31-zip_s1TasksPrintouts.txt) | [S31 pipeline](../../submission_nb/submission_31/agent.yaml) |
| S32 | [S32 printout](submission-32-zip_s1TasksPrintouts.txt) | [S32 prompt](../../submission_nb/submission_32/instruction.md) |

All per-task scores below come from each printout's
`instance_id / resolved / patch_chars / duration_seconds` table.
Detailed symptoms come from its timestamped tool events and from the
checked-in agent configurations; they are not inferred from patch size.

## 2. Overall results

| Version | Approach under test | Resolved | Empty patches | Largest patch (chars) | Total task time (min) | Observed tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **S27** | In-prompt evidence ledger + repetition/progress rules, native tools | **2/15** | 3 | 1,411 | 35.0 | 3.31M |
| **S28** | Short command-first locating/editing workflow; shell fallback | **1/15** | 1 | **1,123,099** | 27.0 | 2.44M |
| **S29** | Evidence-first native repair; restrict shell mutation; validate behavior | **3/15** | 3 | 2,893 | 29.6 | 2.65M |
| **S30** | Same single agent plus *optional* code-graph tools | **4/15** | 1 | **187,143** | 27.1 | 2.25M |
| **S31** | Mandatory graph navigator → implementer → independent verifier | **1/15** | 4 | 2,456 | 34.0 | 2.38M |
| **S32** | Single native agent; remove graph/write_file; optional guarded line edit | **2/15** | 1 | 1,837 | **45.9** | **3.91M** |

`patch_chars` is the reported final patch length, **not** changed-line
count or code quality. The S28 FastAPI diff reports **20,487 changed lines**;
the S30 FastAPI failure was a `write_file` overwrite with a **25-line
replacement**, accounting for its 187,143-character patch. A nonempty or
small patch is **not** proof that the requested behavior was implemented.

**Score trajectory:** 2 → 1 → 3 → **4** → 1 → 2 (out of 15).
There is **no monotonic improvement**, and all versions remain below
the 5/15 engineering target.

## 3. Retention, gains, and regressions

Only **five distinct tasks** were solved by *any* of these six versions.
Ten of the 15 have **never** been resolved in this recorded series.
This is the clearest reason to measure correctness and retain wins rather
than optimize for the number of edits or nonempty patches.

| Task ID | S27 | S28 | S29 | S30 | S31 | S32 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `rich_3905` | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| `rich_3454` | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| `rich_3518` | — | — | ✓ | — | — | — |
| `rich_4077` | — | — | — | ✓ | — | — |
| `rich_3882` | — | — | — | ✓ | — | — |
| **Other 10 tasks** | — | — | — | — | — | — |

| Transition | New wins | Previously resolved cases lost | Net result |
| --- | --- | --- | --- |
| S27 → S28 | none | `rich_3454` | **−1** |
| S28 → S29 | `rich_3518`, `rich_3454` | none | **+2** |
| S29 → S30 | `rich_4077`, `rich_3882` | `rich_3518` | **+1** |
| S30 → S31 | none | `rich_4077`, `rich_3882`, `rich_3905` | **−3** |
| S31 → S32 | `rich_3905` | none | **+1** |

S32 remains **−2 relative to S30** and also fails to retain S29's
`rich_3518` success. Only `rich_3454` was solved by five of six
versions; `rich_3905` was also solved by five of six. Neither stability
nor the correctness of the other three wins carries across designs.

## 4. What the individual experiments actually taught us

### S27 — Evidence & Progress Manager: *no confirmed accuracy gain*

- **Attempt:** a mentally maintained four-slot evidence ledger
  (requirements, location, hypothesis, failures), repetition detector and
  progress controller, keeping the six native tools.
- **Result:** **2/15**, matching the S26 baseline reported in
  [S27's README](../../submission_nb/submission_27/README.md).
  The same two wins `rich_3905` and `rich_3454` persisted.
- **Failure evidence:** the `fastapi_14258` trace has **26 visible
  `read_file (fastapi/routing.py)` calls**, even though the prompt says
  not to reread unchanged source. The trace also contains many failed
  edit operations.
- **Verdict:** retaining native tools was valuable compared with the
  earlier S25 wrapper failure, but **a prompt-only ledger did not
  demonstrably enforce progress or improve resolution**. It is not
  durable external memory or an automatic repetition interceptor.

### S28 — Command-first workflow + shell fallback: *severe regression*

- **Attempt:** replace S27's mental state protocol with shorter
  `rg -n` / `sed -n` source localization, earlier edits, and a shell
  editing fallback after failed anchors.
- **Result:** **1/15**, lost `rich_3454`. It had fewer visible source
  reads (60 vs S27's 153) and lower total run time (27 vs 35 minutes),
  but that did **not** improve correctness.
- **Severe failure:** for `fastapi_14258`, repeated failed edits
  culminated in an unrestricted `sed -i` insertion. The reported
  FastAPI diff had **20,487 changed lines**, and the final patch measured
  **1,123,099 characters**. The task was not resolved.
- **Verdict:** focused source viewing is useful as a technique, but
  **unrestricted shell mutation and early-edit pressure were dangerous
  in this E4B run**. Fast patch production is not successful repair.
  Do not reintroduce this fallback without actual enforced constraints.

### S29 — Conservative native edit and behavior checks: *positive recovery*

- **Attempt:** prioritize the issue's actual behavioral contract, use
  native `edit_file` with fresh short unique anchors, forbid `sed -i` /
  scripted source writes, limit retries and inspect the real diff.
- **Result:** **3/15**, gained `rich_3518` and restored
  `rich_3454` over S28. Maximum final patch fell from >1.1M to
  **2,893 characters**; no comparable destructive FastAPI overwrite
  appears in this run.
- **Remaining weakness:** three empty patches and **86 visible
  `FileEditError` events** across the printed details. Instructions
  did not stop all anchor loops or establish reliable behavioral checks.
- **Verdict:** **best observed conservative reference** among S27–S32.
  The combination of a native interface, issue-oriented location,
  conservative edits and checking has the strongest *supportive
  observation* without a catastrophic overwrite. This is a
  **version-level result**, not proof that any one prompt sentence caused
  the gains.

### S30 — Optional code graph: *highest score, but NOT a graph success*

- **Attempt:** augment S29's single agent with the native
  `search_similar_code`, `get_code_neighbors`,
  `get_code_subgraph` tools, invoked conditionally.
- **Result:** **4/15** (best observed), gaining `rich_4077` and
  `rich_3882` but **losing `rich_3518`**. Visible native edit
  errors dropped to **46** from S29's 86, subject to timestamped-event
  counting limitations.
- **Critical causality check:** **zero graph-tool calls** appear in
  the S30 diagnostic trace. Therefore **there is no evidence that the
  graph produced the improved score**; S30 also changed its prompt
  and edit/localization instructions.
- **Severe failure:** after repeated errors, the FastAPI agent invoked
  `write_file (fastapi/routing.py, 25 lines)`, overwrote the
  existing source and submitted a **187,143-character** unsuccessful patch.
  Merely telling the model not to misuse `write_file` did not stop it.
- **Verdict:** **best raw resolution, not safe to promote as-is**.
  Preserve the useful native workflow, but prevent destructive
  existing-file overwrites. Graph presence without graph use is an
  **unexercised feature**, not a demonstrated improvement.

### S31 — Forced graph navigator + three-agent pipeline: *major regression*

- **Attempt:** sequential graph-only navigator → bounded native
  implementer → independent verifier, remove `write_file` from
  all stages, force graph usage and independent final checks.
- **Result:** **1/15**, losing three of S30's four wins. Empty
  patches rose to **4/15**. Maximum patch size was only **2,456
  characters**, indicating the massive overwrite failure did not recur.
- **Trace evidence:** **91 visible graph-tool calls** but poor task
  resolution; **34 occurrences** of `Tool 'read_file' not found` in
  the navigator's limited tool context (occurrences are not necessarily
  unique charged calls). `rich_3905` was falsely described as fixed
  even though the submitted patch was **empty**.
- **Failure mechanism:** tool-context mismatch, irrelevant/repeated graph
  navigation, stage re-entry and verifier claims without a verified
  behavioral patch. A verifier restricted to its own stage still depends
  on what the model actually verifies.
- **Verdict:** removing `write_file` reduced **one safety hazard**;
  forcing retrieval and multi-agent handoffs **made this run much worse**.
  This does **not** show that code graphs are inherently useless; it
  shows that **mandatory** graph-first coordination failed here.

### S32 — Guarded one-line recovery: *safety improvements, weak adoption,
accuracy regression vs S30*

- **Attempt:** return to one native agent with no graph tools and no
  `write_file`, allowing **one optional guarded single-line
  `run_command` repair** after two native anchor failures. A separate
  control ZIP was created but **not run** in these diagnostics.
- **Result:** **2/15**, `rich_3905` and `rich_3454`. The maximum
  patch was **1,837 characters** and only one patch was empty.
  This is **not** an accuracy improvement over S29 or S30.
- **Trace evidence:** **63 visible `FileEditError` events**, **37
  visible `pytest` commands** (20 bare, non-targeted `pytest` calls),
  two unauthorized `sed -i` invocations in `rich_3882`,
  and **three `submit_patch` calls** in `rich_4077`.
  A Python heredoc appeared in `rich_3938` but **no
  `S32_GUARDED_LINE_EDIT_OK` success marker** appears in the printout;
  there is no confirmed successful guarded recovery.
- **Semantic failure:** `rich_3518` edited handling of `Column`
  headers in `Table.__init__`, while S29's successful repair concerned
  highlighting of **dynamically added columns in `Table.add_row`**.
  A valid nonempty patch on the wrong behavior fails.
- **Time/resource regression:** total recorded time rose to **45.9
  minutes** and observed tokens to **3.91M**, the highest here. **Six
  tasks exceeded four minutes**, including the successful
  `rich_3454` (~307 s), making timed-score transfer especially uncertain.
- **Verdict:** excluding `write_file` avoided the *particular*
  giant-overwrite behavior, but the guarded recovery **was not shown to
  help** and generic shell access still permitted unsafe commands.
  Since the matched `submission_32_control.zip` was not evaluated,
  this run cannot isolate the recovery template's causal effect.

## 5. Useful / unproven / harmful mechanisms

| Mechanism or practice | Classification from this evidence | What to retain or avoid |
| --- | --- | --- |
| Native `read_file`, `edit_file`, `run_command`, `submit_patch` | **Useful baseline** | Retain simple tools that E4B actually invokes; avoid S25-style mandatory wrappers. |
| Ground fixes in requested behavior and correct code owner | **Promising, insufficiently reliable** | Favor S29/S30's issue-oriented localization; measure wrong-target changes directly. |
| Fresh, short, unique native edit anchors | **Promising, not fully solved** | Errors improved in S30, but failures persisted in S29/S32; keep checking actual source and diff. |
| Removing ability to `write_file` over existing code | **Effective for one safety class** | No repeated S30-style giant overwrite in S31/S32; **cost:** excludes legitimate new-file repairs. |
| Mental evidence ledger / repetition detector without external enforcement (S27) | **No demonstrated accuracy benefit** | Do not confuse prompt rules with deterministic state tracking. |
| Shorter command-first source workflow (S28) | **Mixed:** fewer reads, worse score | Source slicing can help; early edit pressure/shell fallback cannot be declared a success. |
| Unrestricted shell mutation / source overwrite (S28, S30, S32) | **Harmful** | S28 and S30 catastrophic diffs; S32 still issued `sed -i` despite prohibition. |
| Optional code graph only declared in YAML (S30) | **Unproven** | Zero actual calls, so do not credit graph for S30 wins. |
| Mandatory graph + sequential multi-agent pipeline (S31) | **Harmful in observed run** | 91 graph calls, context/tool errors, lost wins; avoid mandatory graph stage without real-model evidence. |
| Independent verifier stage (S31) | **Not reliably effective** | A verifier can claim success on empty patches; test the requested behavior and inspect actual diff. |
| Optional guarded one-line fallback (S32) | **No verified benefit** | No success marker, no control run, editing errors and unsafe commands persisted; do not promote on syntax unit tests. |
| Unfocused repeated `pytest`, repeated submit, same-path reads | **Wasteful, sometimes misleading** | Replace with one issue-specific check and explicit stop criterion; measure actual behavior and retention. |
| Post-run gates, packager/unit tests | **Useful engineering hygiene, not task-score evidence** | Preserve them to catch invalid archives and diagnostics; only real E4B task resolution measures repair accuracy. |

A **safety improvement is not necessarily an accuracy improvement**.
S31/S32 show smaller patches but poor task resolution; S30 shows the highest
resolution while still having a destructive failure. Evaluate both
independent dimensions.

## 6. Recommendations for the next experiment

1. **Keep two distinct reference points:** S30 for highest observed
   accuracy (4/15) and S29 for a conservative source-editing baseline
   (3/15). Do not copy S30's unconstrained existing-file overwrite.
2. **Choose ONE hypothesis per next version.** Focus first on
   *behavior-first localization and relevant validation*, not another
   mandatory agent stage, graph detour or prompt-only editing fallback.
   For example: identify failing input, responsible API and expected
   result; apply a small patch; run a focused fail-to-pass check; preserve
   a confirmed good patch rather than continue editing.
3. **Track the five historically resolved tasks**, especially the
   three unstable wins `rich_3518`, `rich_4077`, `rich_3882`.
   Compare the actual source site and behavioral assertion that passed
   in the successful version; do **not** hardcode benchmark fixes or
   use reference patches in the agent submission.
4. **Require exact completion and safety evidence:** successful native
   edit, bounded `git diff --check` / `git diff --numstat`, at least
   one issue-specific behavioral check, no uncontrolled `sed -i` /
   `write_file` overwrite, and one final `submit_patch`. Treat missing
   fixtures as BLOCKED, not PASS.
5. **Use matched real-E4B repeats:** same 15 task IDs, model, seeds,
   tooling, snapshots, and session budget. Run at least two repeats,
   including **the actual four-minute limit**, and then check other
   held-out tasks. Never infer Kaggle-score gains from unlimited-time
   diagnostics or from synthetic tool/ZIP tests alone.

**Promotion gate proposal:** >=5/15 genuine resolutions in *each* matched
run, no destructive diff, and measured retention of prior wins; also
record per-task time, failed edits, wrong-target patches, issue-specific
test evidence, and final exact changed files. A version that has more
nonempty patches but solves fewer tasks should be rejected.

## 7. Trace-count definitions (for audit)

- **Resolution**, **patch length**, **duration** and **observed tokens**
  are taken from the last 15-row summary table of each printout.
- For **visible tool-event counts**, count timestamped entries with an
  actual tool name or `❌ ... FileEditError`; do **not** count the
  duplicated decorative "Latest Detail" cards or narrative mentions.
  By this parser, visible edit-error events are approximately:
  S27 **68**, S28 **81**, S29 **86**, S30 **46**, S31 **56**, S32
  **63**. Other historical review scripts use different definitions,
  yielding different totals; these are **not** identical to unique
  failed-argument pairs, charged tool counts, or all errors in ATIF.
- Visible source-read counts: S27 **153**, S28 **60**, S29 **93**,
  S30 **78**, S31 **111**, S32 **147**. S31/S32 undercount if the
  detailed trace omitted tasks; do not use these alone as efficiency
  denominators.
- S30 **0** observed graph calls; S31 **91** observed graph calls.
  S31 "read_file not found" **34 text occurrences**. S32 guarded
  recovery **0 success markers** in the printout.
- The repo's raw `.zip` diagnostics contain additional structured
  evidence for forensic audits. Read final patches, tool arguments and
  behavioral tests there before claiming more specific causality.

**Practical conclusion:** stop optimizing whether the agent *produces a
patch*. Optimize whether it **understands the requested behavior,
changes the right implementation, verifies an actual regression, and
preserves the resulting correct fix**. The S27–S32 record supports this
priority much more strongly than any new elaborate agent graph.
