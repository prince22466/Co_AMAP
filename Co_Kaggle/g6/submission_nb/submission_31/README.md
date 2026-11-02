# Submission 31 — graph-first navigation, safe native editing and independent behavior check

**Status: experimental / performance NOT RUN.** S29 resolved 3/15, S30
resolved 4/15 on the selected S1 set. The full trace for S30 showed **zero
calls** to the graph tools. S30 also replaced fastapi/routing.py with a
25-line file after 19 unsuccessful edits, producing a 187,143-character
patch; and it lost S29's rich_3518 success.

## New architecture (small, declarative ADK only)

A single `SequentialAgent` runs three E4B agents sequentially:

1. **Graph navigator:** can call only the three native competition graph
   tools; creates a short state handoff `navigation_evidence`. For an
   issue that names a usable symbol it is prompted to call graph tools,
   rather than leaving them unused as optional tools. Missing graph assets
   or no named symbol yield an explicit no-evidence handoff.
2. **Implementer:** native `read_file`, `edit_file`, `run_command`,
   `get_status`. `write_file` is REMOVED from every stage, and
   implementer has no submit tool. It must inspect CURRENT source and
   use exact short anchors. This structurally prevents **write_file**
   replacement of an existing source file, not unsafe shell writes
   through `run_command`. Shell-edit prohibition remains instruction-only.
3. **Independent verifier:** native `run_command`, `read_file`,
   `get_status`, `submit_patch`. This agent has no edit/write access,
   checks the changed diff and one issue-specific behavior before
   submitting. Because stock declarative ADK has no per-call interceptor,
   tests and safety threshold compliance are still prompt guided;
   post-run trace validation plus actual ATIF/diff review is required.

All stages share the S30 E4B model, sampling defaults, **40 calls,
48 turns, four-minute per-task limit**. The configuration does not
ship a bespoke tool interface, LoRA, code-generation skill, oracle
answers or benchmark task-specific instructions. No LoopAgent/repeated
review stages: unlike S24, this is one bounded pass.

## Why this addresses observed failures

- **P0 destructive overwrites:** no `write_file` capability in any stage.
  Unrestricted `run_command` remains the residual risk; final diff and
  post-run rejection monitor it. This is scoped to S1 existing-file fixes
  and may miss hidden tasks requiring new files.
- **P1 graph activation:** a dedicated read-only navigator must consider
  graph retrieval before the editor. Tool availability/execution is measured
  explicitly; simply declaring tools is not counted as using them.
  `search_similar_code` only accepts an existing symbol name, and graph
  nodes do NOT certify the correct source or behavior.
- **P2 behavioral proof:** editing agent cannot prematurely submit.
  Separate verifier checks actual diff and requested input/output using
  allowed native tools, then alone calls `submit_patch`.

## Promotion gate: >=5/15, retained wins, safety, observed graph usage

Predeclared S1 tasks remain unchanged. Require **two complete real E4B S1
repeats** with >=5/15 resolved each, including all three S29 wins
(`rich_3905`, `rich_3518`, `rich_3454`) and at least one new S30
win (`rich_4077` or `rich_3882`). Ideally retain all five across
the runs. Also reject destructive patches, repeated edit loops, absent
graph-tool calls on the S1 run and missing observed verification activity.
Match inference settings and evaluation budget across candidates. An
initial 7-task pilot is a cost-saving screen, NOT an alternative S1 gate.

## Build / validate

```bash
cd Co_Kaggle/g6/tests_on_tasks/tools
python build_submission31.py
python -m unittest test_submission31.py
python gate_submission31.py \
  ../diagnostics/submission-29-zip_s1TasksPrintouts.txt \
  ../diagnostics/submission-30-zip_s1TasksPrintouts.txt \
  /path/to/submission-31-zip_s1TasksPrintouts.txt --stage s1
```

Official-runner (requires installed harness, snapshots and served E4B):

```bash
python run_submission31_eval.py --tasks /path/to/tasks.jsonl \
  --snapshots /path/to/snapshots --wheels /path/to/wheels \
  --stage pilot --repeats 2 --time-limit 4 --output /path/to/new-pilot
python run_submission31_eval.py --tasks /path/to/tasks.jsonl \
  --snapshots /path/to/snapshots --wheels /path/to/wheels \
  --stage s1 --repeats 2 --time-limit 4 --output /path/to/new-s1
```

The official runner never changes reference tests or grading. A real
reported success requires official `resolved` booleans; synthetic unit
tests only verify packaging/gates, not agent performance. Keep this PR
as a **draft** until full evaluation succeeds.
