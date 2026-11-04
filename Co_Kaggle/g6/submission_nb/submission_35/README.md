# Submission 35 — E4B native-tool recovery with optional skill

**Status: unscored candidate.** S35 uses the **exact same model and generation
settings as S34**, not 31B. Real-model results are required before promotion.

## Evidence and goal

S34's single recorded E4B S1 run resolved 2/15 selected tasks and showed
128 visible FileEditError events, 45 malformed native-edit calls and five
empty patches. It lost S29's successes on `rich_3518` and `rich_3454`.
Prompt shortening did not prevent repeated failed operations.

S35 targets three capabilities without switching models:
1. **Tool-call reliability:** named native-tool arguments, five native tools,
   no `write_file` (a known overwrite hazard). The unrestricted
   `run_command` shell remains available, so this is not a sandbox.
2. **Feedback interpretation:** distinguish malformed calls, exact-match
   failures, no-op edits, missing dependencies, and behavioral test failures.
3. **Failure recovery:** after two failed edits at a location, optionally
   consult an ADK instruction-only recovery skill and use a guarded,
   small `git apply --check` fallback. No custom script dispatcher.

The optional skill **cannot intercept a malformed native call or enforce retry
limits**. The model must choose to load and follow it. The harness's
declarative format does not offer submission-defined tool callbacks. The
offline diagnostic gate detects failures *after* the run, not during it.

## Exact S34 model parity

Both S35 variants retain these S34 settings:

| Setting | S34 | S35 primary and control |
| --- | --- | --- |
| Model | `gemma-4-e4b-it-qat-w4a16-ct` | **Identical** |
| Temperature / top-p | 0.0 / 0.95 | **Identical** |
| Max output tokens | 4096 | **Identical** |
| Thinking budget / thoughts | 512 / false | **Identical** |
| Time / tool calls / turns | 4 min / 40 / 48 | **Identical** |
| Conversation history | `include_contents: default` | **Identical** |

S35 intentionally differs from S34 in its prompt, native tool list and
optional skill. Therefore S34→S35 is **not** a single-variable ablation.
The additional no-skill S35 control isolates the skill's effect within S35.

## S35 experimental variants

| Archive | Native tools | Other capability |
| --- | --- | --- |
| `submission_35.zip` | `run_command`, `read_file`, `edit_file`, `get_status`, `submit_patch` | Optional `tool-recovery` skill |
| `submission_35_control.zip` | Exactly the same five | No skill |

The control uses exactly the same E4B model, generation, task prompt,
sampling and evaluation limits as the primary. The **only difference
between the S35 variants** is skill registration and its packaged
`SKILL.md`. Neither uses LoRA or additional LLM agents.

The skill contains an error taxonomy and one example of a bounded
`git apply --check` sequence. It registers no tools, scripts or wrappers.
It is an optional reference, not a mandatory workflow. The `write_file`
tool is absent in both archives, so new files would require another
permitted approach and should only be created when strictly required.

## Build and CPU checks

From `Co_Kaggle/g6/tests_on_tasks/tools/`:

```bash
python -m pip install 'pyyaml>=6,<7'
python build_submission35.py
python -m unittest -v \
  test_submission35.py test_gate_submission35.py test_run_submission35.py
```

The builder validates S35 model, sampling and budget against the
**actual S34 `agent.yaml` / `eval_config.yaml`**, checks for forbidden
extra tools or models, validates skill metadata, and builds two
deterministic ZIPs. CI verifies they exactly match the committed
artifacts. **CPU tests are not task-resolution evidence and do not
compile the archives through the live competition runtime.**

## Matched E4B evaluation

Start the same E4B vLLM model used for S34. The runner delegates
execution and phase-2 verification to the existing harness without
changing the evaluator's model. With tasks, snapshots and wheels
available, run two same-seed pilots (from `tests_on_tasks/tools/`):

```bash
python run_submission35_eval.py --variant control \
  --tasks /path/tasks.jsonl --snapshots /path/snapshots --wheels /path/wheels \
  --stage pilot --repeats 2 --time-limit 4 --output /path/s35-control-pilot
python run_submission35_eval.py --variant primary \
  --tasks /path/tasks.jsonl --snapshots /path/snapshots --wheels /path/wheels \
  --stage pilot --repeats 2 --time-limit 4 --output /path/s35-primary-pilot
```

Pilot: `rich_4077`, `rich_3518`, `rich_4076`,
`requests_7315` and `rich_3905`. Selected to mix known
previous successes with observed edit failures; no reference fixes or
grader modifications are included in the agent ZIP.

Compare each matching repeat's official JSONL files:

```bash
python gate_submission35.py --stage pilot \
  --control /path/s35-control-pilot/candidate_1 \
  --primary /path/s35-primary-pilot/candidate_1
```

The JSONL gate compares resolved tasks, empty/oversized patches,
protected-file changes and agent errors. **The safety-screen result
remains blocked without complete printed tool telemetry**, even with
valid JSONL, because the gate deliberately does not reconstruct full
native-tool arguments from ATIF. For a complete screen, supply
`--primary-printout` with a full, matching timestamped tool printout;
`--control-printout` provides supplementary control telemetry.
If these are unavailable, manually review the official JSONL `trace`
fields and actual patches, without interpreting a blocked screen as a
failed score.

Run `--stage s1 --repeats 2` only after the five-task pilot is useful.
Use matched `--time-limit 0` for historical unlimited-time comparisons
and separately matched `--time-limit 4` for competition-like timing.
Never compare unlimited-time wins directly with timed wins.

## Promotion criteria and caveats

For both matched 15-task repeats, aim for at least 5/15 resolved,
retain all three S29 historical wins (`rich_3905`, `rich_3518`,
`rich_3454`), beat the **same-seed E4B no-skill control**, have
at most two empty patches, no destructive patch and no systematic
repeated failed edits or malformed native calls. Inspect the final
behavior and raw ATIF call arguments; the analyzer never marks
`promotion_approved` automatically.

Skill registration is not proof of skill use. If no `load_skill`
call occurs, the skill did not demonstrably help. If the control
outperforms the primary, discard the skill rather than expanding
the instructions again. If both variants remain weak, isolate the
tool-surface reduction next, without changing model size.

## Explicit non-goals

- No external Python callbacks, model switch, LoRA, or model mixing.
- No mandatory multi-agent stages or custom skill-script editors.
- No grader modifications, benchmark reference fixes or claims of
  mechanically enforced context compaction or retry limits.
