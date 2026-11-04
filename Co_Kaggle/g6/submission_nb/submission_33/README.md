# Submission 33 — behavior-owner localization before native editing

**Status: candidate, NOT tested on real Gemma E4B.** A valid ZIP is not proof
of resolution, and the 15-task diagnostic results are not four-minute scores.

## Why this design

The same fixed S1 task set yielded S27 **2/15**, S28 **1/15**, S29
**3/15**, S30 **4/15**, S31 **1/15**, and S32 **2/15** (one diagnostic
run each). See
[the source-backed review](../../tests_on_tasks/diagnostics/submission_27_to_32_review.md).

S29 retained correct small native edits, including a dynamically executed
behavior in a table update method. S30 had the highest raw score, but its
new wins were not graph-tool wins (zero observed graph calls), and its
`write_file` overwrite destroyed an existing FastAPI file (187,143-character
patch). Mandatory graph/multi-agent stages in S31 regressed; S32's
guarded shell fallback had no observed success marker and did not recover
the S29/S30 unstable wins. In S32 a small nonempty edit modified the
wrong phase of a dynamic operation. This is a *code-ownership/behavior*
failure, not simply an editor failure.

## Exactly what S33 changes

**One hypothesis:** a compact, explicit behavior-owner checkpoint
(operation, expected result, control case, time of behavior, exact producer)
plus targeted post-edit check is more useful for E4B than lengthy edit
fallback instructions or mandatory navigation stages.

- Single `gemma-4-e4b-it-qat-w4a16-ct` agent, no LoRA or extra model.
- Five native tools: `run_command`, `read_file`, `edit_file`,
  `get_status`, `submit_patch`; no graph tools, write_file, helper,
  custom workbench or shell editing recovery.
- Same temperature, top-p, max output/thinking and evaluation limits as S32.
  No new-file creation support in this experiment; tasks requiring a new
  file may necessarily fail.
- Small exact anchors, one read-and-retry on failed match, narrow testing of
  the *reported operation*, diff inspection and a single final submit.
- No task-specific solutions, reference patches or test/grade modifications
  are packaged.

The native-tool list **enforces absence of write_file and graph**.
Everything else (correct localization, no shell source writes, retry limits,
behavioral test discipline, one submit) is **prompt guidance only**.
Because `run_command` is an unrestricted shell, a separate trace and patch
audit is still mandatory.

## Build / local checks

From `Co_Kaggle/g6/tests_on_tasks/tools`:

```bash
python build_submission33.py
python -m unittest -v test_submission33.py
```

Produces `../../submission_nb/artifacts/submission_33.zip` and a SHA-256
manifest. Only `agent.yaml`, `eval_config.yaml` and `instruction.md`
enter the ZIP; the README, builder and diagnostics are outside the package.
These tests verify configuration, archives and invariants, **not** model
reasoning or runtime compilation.

## Real-E4B evaluation (not run here)

Use the existing official runner, with a preflight and then **two matched
runs**, not a selected lucky run. Use the same 15 IDs and sandbox for
candidate and baselines:

```bash
python run_submission24_eval.py --tasks /path/tasks.jsonl \
  --snapshots /path/snapshots --wheels /path/wheels \
  --archive ../../submission_nb/artifacts/submission_33.zip \
  --stage s1 --repeats 2 --time-limit 0 --baseline-only \
  --output /path/s33_preflight

python run_submission24_eval.py --tasks /path/tasks.jsonl \
  --snapshots /path/snapshots --wheels /path/wheels \
  --archive ../../submission_nb/artifacts/submission_33.zip \
  --stage s1 --repeats 2 --time-limit 0 --output /path/s33_s1_unlimited
```

`--time-limit 0` matches the historical *unlimited* diagnostics; then run
a **separate** `--time-limit 4` study to approximate configured task limits.
The baseline-only preflight must be ready before the model comparison.
The local runner needs the exact E4B served model, official wheels, task
snapshots and a supported sandbox; it cannot run offline in this PR.

For printed diagnostics, run:

```bash
python gate_submission33.py ../diagnostics/submission-29-zip_s1TasksPrintouts.txt \
 ../diagnostics/submission-30-zip_s1TasksPrintouts.txt /path/s33Printouts.txt
```

Promotion criteria for **each** of two matched S1 runs: at least **5/15**
resolved, retain S29's `rich_3905`, `rich_3518`, `rich_3454`, and
at least one S30-only win (`rich_4077` or `rich_3882`); no catastrophic
or unrelated patch, repeated unsafe shell mutation, or unverified outcome.
Also inspect all ATIF tool arguments, actual resulting source diffs and
issue-specific checks. A printed-log gate is only triage, not approval.
Compare held-out tasks before submitting to Kaggle.
