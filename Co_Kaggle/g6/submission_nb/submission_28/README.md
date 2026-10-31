# Submission 28 — command-first native E4B repair experiment

**Status:** candidate, unscored; requires official real-E4B validation. Target is a
substantial improvement over the observed S26/S27 **2/15** S1 resolved rate.
The target is a hypothesis, not a measured result.

## Evidence driving this change

S27 held S26's two solved cases (`rich_3905`, `rich_3454`) but did not add
solved tasks. It created 12 nonempty patches against S26's 10. Printed S27
traces show `fastapi_14258` reading `fastapi/routing.py` 26 times and
`rich_4076` repeatedly attempting edits; those are *same-path* counts, not
proof of identical arguments because text traces elide arguments/line ranges.

The S27 in-prompt evidence ledger and progress-controller decision tree are
apparently not reliably followed by E4B. S28 removes that multi-part state
protocol and supplies a short, sequential workflow with concrete bounded
`rg` + `sed` examples. It addresses the common failure of `read_file`
returning the first capped source slice over and over; a numbered `sed`
command can reach the actual symbol/anchor.

## Controlled change

- Identical E4B QAT model, generation config, six native tools and evaluation
  limits to S27. Only the packaged **instruction** is different.
- Retain native editing/read/command/submit tools. No mandatory skills,
  custom dispatch, wrappers, helper agents, LoRA or prohibited file types.
- Focused source selection -> exact small edit anchor -> relevant test ->
  actual diff -> single explicit submission.
- On repeated failed anchors change to a guarded shell edit with a uniqueness
  check, never a blind file overwrite.
- Prompt constraints are **advisory**. They do not enforce behavior at the
  tool-dispatch boundary or guarantee that E4B follows them. No improvement
  is claimed without matched real-model task evaluation.

## Reproduce the package and check it

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission28.py
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission28.py -v
```

Upload `Co_Kaggle/g6/submission_nb/artifacts/submission_28.zip` as
`submission.zip`. The checked-in archive includes only `agent.yaml`,
`eval_config.yaml` and `instruction.md`.

## Real-model pilot / release gate

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/run_submission28_eval.py \
  --tasks /kaggle/input/competitions/gemma-4-developer-agent/tasks.jsonl \
  --snapshots /kaggle/input/competitions/gemma-4-developer-agent/snapshots \
  --wheels /kaggle/input/competitions/gemma-4-developer-agent/wheels \
  --served-model gemma-4-e4b-it-qat-w4a16-ct \
  --stage pilot --repeats 2 --time-limit 4 \
  --output /kaggle/working/s28-pilot
```

Use *the same* snapshot, model server, allowed tools, 40-call ceiling, baseline
preflight and evaluator for matched S27 and S28. Predeclare these five IDs:
`rich_4077`, `rich_3905`, `rich_3518`, `rich_3454`, `rich_3882`.
For EACH repeat require >=3/5 solved, retain `rich_3905` and `rich_3454`,
<=1 empty patch, no agent errors. Inspect each model-emitted tool action,
real edit, relevant assertion and submitted patch, and do not promote on a
failed pilot. The legacy S25 gate additionally required retaining
`rich_4077`; this task was not solved in S26/S27, so record it as a target,
not a falsely claimed retention.

If pilot passes, run matched full S1 twice and compare resolution, empties,
source-read repetitions (proxy), edit failures, 40-call exhaustions, tokens
and latency. Compare printed traces with:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/compare_submission28.py \
  Co_Kaggle/g6/tests_on_tasks/diagnostics/submission-27-zip_s1TasksPrintouts.txt.txt \
  /path/to/s28-s1-printouts.txt
```

Use raw ATIF/tool calls, not text proxies, before attributing gains to fewer
identical calls. Do not change grading tests or promote merely because CPU
packaging tests pass.
