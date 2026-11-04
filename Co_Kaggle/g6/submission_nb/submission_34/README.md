# Submission 34 — compact-context prompt ablation (E4B)

**Status: candidate, not tested on the actual Gemma model.** The ZIP and CPU
checks do not prove issue resolution. Do not submit to Kaggle as an improvement
until the real-model promotion criteria are met.

## Hypothesis

S33 resolved 2/15 fixed S1 tasks, had 86 printed FileEditError events,
three empty patches, and zero observed `git diff` shell commands.
The previous long prompt did not prevent edit loops or enforce verification.
S29 resolved 3/15 with conservative patches (largest 2,893 chars).

**S34 changes only the S29 instruction text.** Its single E4B agent, six native
tools (including `write_file` for genuinely new files), model, generation
parameters, and evaluation budgets are *byte-identical* to S29. This allows
a meaningful prompt-level ablation without conflating tool changes, agent
structure, model choice, or temperature/thinking changes.

The new instruction is ~45% of S29's length. It asks for 20–60-line reads,
12-result searches, a three-fact working summary (expected behavior, current
code site, last failure), one edit recovery attempt, a focused test, and diff
inspection. This is **model-facing context discipline, not true host-enforced
context compaction**. S34 deliberately does not ship a skill: S25's mandatory
skill-script workbench caused severe outer-tool dispatch failures. Skills
should be tested *separately* only after the simpler native-agent baseline.

## Safety and limitations

- `write_file` stays exposed for exact S29 parity; the warning against
  overwriting existing source is **not enforced**. Inspect actual diffs.
- Retry limits, bounded reads, verification and one-time submission are all
  **prompt guidance**, not runtime constraints. We expect violations and
  measure them, rather than treating the prompt as enforcement.
- The effective E4B context window is not known from this experiment. The
  harness may compact events automatically. Reducing output length does not
  ensure that the model can retain the necessary evidence.
- The historical S1 diagnostics were run with the time limit disabled.
  A separate four-minute run is mandatory for submission-like timing.
- Do not copy benchmark reference patches, tests, or solutions into the
  agent prompt. The public-task IDs appear only in offline evaluation code.

## Build and CPU checks

From `Co_Kaggle/g6/tests_on_tasks/tools/`:

```bash
python build_submission34.py
python -m unittest -v test_submission34.py test_analyze_submission34.py
```

The checked-in `submission_nb/artifacts/submission_34.zip` must match the
deterministic builder byte for byte. Its only root entries are `agent.yaml`,
`eval_config.yaml`, and `instruction.md`. The separate
`submission_34.manifest.json` is generated locally by the builder.

## Real-model experiment

Use the existing official harness, the fixed S1 15 tasks, and the same
model server, snapshots and wheels for S29 and S34. Run a five-task pilot
first (`rich_4077`, `rich_3905`, `rich_3518`, `rich_3454`,
`rich_3882`); do not proceed to a large run if S34 loses native-tool
dispatch or has widespread empty patches.

Suggested S1 run, from `tests_on_tasks/tools/` (replace local paths):

```bash
python run_submission24_eval.py --tasks /path/tasks.jsonl \
  --snapshots /path/snapshots --wheels /path/wheels \
  --archive ../../submission_nb/artifacts/submission_29.zip \
  --stage s1 --repeats 2 --time-limit 0 --output /path/s29-matched
python run_submission24_eval.py --tasks /path/tasks.jsonl \
  --snapshots /path/snapshots --wheels /path/wheels \
  --archive ../../submission_nb/artifacts/submission_34.zip \
  --stage s1 --repeats 2 --time-limit 0 --output /path/s34-matched
```

Also repeat with `--time-limit 4`. Never mix unlimited-time and timed
scores when claiming progress. For an individual pair of complete S1
printed logs:

```bash
python analyze_submission34.py \
 ../diagnostics/submission-29-zip_s1TasksPrintouts.txt \
 /path/submission-34-zip_s1TasksPrintouts.txt
```

The analyzer reports resolved IDs, lost and gained tasks, read and edit
events, malformed edit calls, first-edit success proxy, duplicate submits,
diff-check commands, and total reported input/output tokens. Printed-token
totals include repeated context; they are not unique tokens or a measure
of the working context size. Review the full ATIF traces and actual patches.

**Predeclared promotion target, for each of two matched S1 repeats:**
at least 5/15 solved, retain all three S29 wins
(`rich_3905`, `rich_3518`, `rich_3454`), no unrelated or destructive
patch, and no systematic tool-call regression. Also inspect performance
under the real four-minute budget. A single log cannot authorize promotion.

If this fails, the next step is a separately tested optional skill or
host-supported context isolation mechanism, *not* adding more workflow
instructions to this submission.
