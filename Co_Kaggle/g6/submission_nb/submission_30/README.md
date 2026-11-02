# Submission 30: selective native code-graph localization (E4B)

## Status and target

**Unscored candidate — performance gate NOT RUN.**
S29 resolved **3/15** on the fixed S1 set: rich_3905, rich_3518, rich_3454.
S30 is **not an obvious improvement** until it resolves **at least 5/15**
on the *same* official E4B/S1 evaluation and retains all three S29 wins.
One run is directional; require two complete matched repeats for promotion.

## Controlled change from S29

- Single Gemma 4 E4B QAT agent, same sampling, budget, native editing tools.
- ADD only the competition's three native read-only code intelligence tools:
  `search_similar_code`, `get_code_neighbors`, `get_code_subgraph`.
- Conditional graph routing in `instruction.md`; no subagents, skill
  wrappers, custom workbench, LoRA, hidden-test access or new scoring logic.
- Small 0–2 graph calls normally; exact symbol/source search first;
  similarity accepts EXISTING SYMBOL names, not natural-language prompts.
- Keep graph results as discovery hints, verify actual source and behavior
  before editing. If graph returns nothing, fall back to shell source search.
- Improve exact-anchor editing: shortest unique code source span, fresh
  verification after mismatch, no repeated guesses or shell mutation.
- Focused behavioral check + actual diff inspection before submission.

## S29 failure modes addressed (hypotheses, NOT measured outcomes)

| Task class | S29 symptom | S30 intended intervention |
| --- | --- | --- |
| rich_4077 | Edited console.py instead of file_proxy.py | Owner discovery by exact class/symbol + targeted graph suggestions |
| rich_4076 | Edited progress.py instead of ansi.py | Trace actual decoding behavior before edit |
| rich_3934 / rich_3938 | Wrong/incomplete multi-module changes | Neighbors or induced subgraph ONLY when source shows cross-module behavior |
| fastapi_14258 / requests_7205 / requests_7315 | Numerous edit_file failures | Shorter unique code anchors, read-once retry, no runaway edit loop |
| rich_3052 / rich_3676 | Patch exists but verification insufficient | One observable issue-specific test rather than unrelated green pytest |

Graph tools are advisory. Their presence does not enforce any verification or
retry limits at runtime. Do not claim causal improvement without real runs.

## Reproducible package and evaluation

From `Co_Kaggle/g6/tests_on_tasks/tools`:

```bash
python build_submission30.py
python -m unittest test_submission30.py
python gate_submission30.py \
  ../diagnostics/submission-29-zip_s1TasksPrintouts.txt \
  /path/to/submission-30-zip_s1TasksPrintouts.txt --stage s1
```

Upload `submission_nb/artifacts/submission_30.zip` only after verifying
it matches the committed `submission_30/` config. Do not assume the
presence of an artifact establishes model compatibility or solved tasks.

Official-runtime runner (requires existing E4B inference endpoint and
snapshots/wheels in the Kaggle sandbox):

```bash
python run_submission30_eval.py --tasks /path/to/tasks.jsonl \
  --snapshots /path/to/snapshots --wheels /path/to/wheels \
  --stage pilot --repeats 2 --time-limit 4 \
  --output /path/to/new-pilot-output
python run_submission30_eval.py --tasks /path/to/tasks.jsonl \
  --snapshots /path/to/snapshots --wheels /path/to/wheels \
  --stage s1 --repeats 2 --time-limit 4 \
  --output /path/to/new-s1-output
```

Pilot is predeclared seven tasks: rich_4077, rich_4076, rich_3934,
rich_3905, rich_3518, rich_3454, rich_3938. Require >=5/7 and retention
of the three S29 wins in BOTH real-model pilot repeats, with no unsafe edits.
S1 requires >=5/15 in EACH repeat, all three prior wins, no catastrophic
or repeated failed edits, and manual review of task-specific behavior,
actual patch diff, full tool arguments and runtime manifests.

The printed trace gate is post-run triage only, not an in-agent
interceptor. Use actual ATIF and official resolved booleans for final
assessment. No benchmark tests or grading fixtures are modified.
