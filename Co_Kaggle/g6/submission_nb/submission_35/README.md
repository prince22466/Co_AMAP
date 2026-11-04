# Submission 35 — 31B-native tool recovery with optional instruction skill

**Status: candidate, no real-model task tests yet.** Do not infer issue
resolution from package tests or a successful CI check.

## Motivation and measured baseline

S34 E4B resolved **2/15 S1 tasks**, with **128 visible FileEditError**
events, **45 missing-argument errors** and **5 empty patches**. It lost S29's
`rich_3518` and `rich_3454` successes despite shortening the prompt.
The same-path retry instruction was not enforced: `requests_7315`,
`rich_3934`, and `rich_4076` each showed dozens of edit failures.

The competition supports 31B, 12B, E4B and E2B independently. S10's
earlier **31B** public Kaggle score was 0.15 (different harness topology);
that score motivates a capacity experiment but does not predict S35.

## The system: reliable native interface + optional recovery knowledge

**Primary S35 uses a single 31B agent** with five native tools:
`run_command`, `read_file`, `edit_file`, `get_status`,
`submit_patch`. The model sees concrete tool argument schemas in a
bounded 2.6K-character core instruction. `write_file` is *not
registered*; this prevents accidental calls to that destructive tool,
although `run_command` is still powerful and therefore not a security
boundary.

An optional **instruction-only** `tool-recovery` ADK skill supplies an
error taxonomy and a bounded `git apply --check` fallback. It registers
no extra edit tool and has no script to dispatch. Native tools remain the
primary path. The intended escalation is:

1. Inspect narrow source and try a native exact `edit_file`.
2. On missing tool arguments, correct the required schema ONCE. On an
   unmatched/ambiguous old string, reread local lines and retry ONCE.
3. After two failures, if the optional skill is actually loaded, use its
   recovery instructions to choose a new verified site or **one** small
   `git apply --check` + `git apply` command. Never overwrite source
   with `cat > source`, `write_file`, `sed -i`, or Python file rewrites.
4. Verify the *requested* behavior and inspect the final diff before submit.

These are **prompt-level decisions** except for tool availability
(`write_file` is removed) and the host's existing tool validators.
The competition's declarative submission format does **not** permit
a competitor-defined interception callback for malformed outer tool calls.
In particular, the skill cannot catch errors before `load_skill`
has been called; failure/repetition limits are NOT mechanically enforced.
The offline diagnostics gate detects such failure *after* a run.

## Matched control: isolate the skill

The PR ships two archives:

| Archive | Model | Native tools | Instruction | Extra |
| --- | --- | --- | --- | --- |
| `submission_35.zip` | Gemma 4 31B QAT | Same 5 | Identical | Optional `tool-recovery` |
| `submission_35_control.zip` | Gemma 4 31B QAT | Same 5 | Identical | No skill |

The only difference is the skill registration and its `SKILL.md`.
Both share exact generation parameters (temperature 0, output 8192,
thinking budget 2048, thoughts hidden), 40 tools, 48 turns, 4 minutes
and the same source/task set. No LoRA or external model is used.
This pair can identify the effect of providing the *skill*; it cannot
isolate 31B-vs-E4B because historical E4B setups differ.

**Keep the models separate:** do not attempt to put 31B and E4B agents
inside one submission. The competition permits only one model variant
per archive.

## Build and CPU validation

From `Co_Kaggle/g6/tests_on_tasks/tools/`:

```bash
python -m pip install pyyaml
python build_submission35.py
python -m unittest -v \
  test_submission35.py test_gate_submission35.py test_run_submission35.py
```

CI verifies two deterministic archives byte-for-byte against the
committed ZIPs, exact allowed tool set, ablation parity, skill metadata,
runner model scoping, negative configuration tests, and safety-screen
logic. **No CI check invokes a real Gemma model or guarantees that
the model will load the skill.**

## Paired official-runner evaluation

The runner delegates task execution and phase-2 checking to the
existing stock `run_submission24_eval.py` pipeline. It temporarily
points that runner at the competition-approved 31B model, and restores
the model setting after evaluation. Start the **31B vLLM server first**,
make sure the official wheels/snapshots and tasks are present, then
run a paired 5-task pilot. From `tests_on_tasks/tools/`:

```bash
python run_submission35_eval.py --variant control \
  --tasks /path/tasks.jsonl --snapshots /path/snapshots --wheels /path/wheels \
  --stage pilot --repeats 2 --time-limit 4 --output /path/s35-control-pilot
python run_submission35_eval.py --variant primary \
  --tasks /path/tasks.jsonl --snapshots /path/snapshots --wheels /path/wheels \
  --stage pilot --repeats 2 --time-limit 4 --output /path/s35-primary-pilot
```

The pilot IDs are `rich_4077`, `rich_3518`, `rich_4076`,
`requests_7315`, and `rich_3905`. These include two historically
straightforward successes and three problematic edit/tool trajectories.
No grader tests or benchmark reference fixes are injected into the
agent. Check baseline environment readiness before drawing conclusions.

Compare **the same repeat** from each pair using official JSONL:

```bash
python gate_submission35.py --stage pilot \
  --control /path/s35-control-pilot/candidate_1 \
  --primary /path/s35-primary-pilot/candidate_1
```

For a fuller tool-failure diagnostic, pass `--primary-printout` and
`--control-printout` if notebook-style *complete* time-stamped
printouts are also available. The official runner stores full ATIF
traces within JSONL; the printed-event gate is only a proxy and fails
closed when detailed task events are missing. Review actual edit args
and the final patch in the stored traces and outputs.

If the pilot is promising, repeat both variants using
`--stage s1 --repeats 2`, first with `--time-limit 0` to match
historical unlimited-time diagnostics and *separately* with
`--time-limit 4` for competition-like timing. Do not mix those
windows when comparing results.

## Promotion requirements and stop rules

For **each of two matched S1 repetitions**, the primary should reach
at least 5/15, retain the three known S29 successful tasks, beat its
same-seed no-skill 31B control, have <=2 empty patches, no destructive
diff, and no systematic malformed-argument or repeated-edit streaks.
Review full ATIF and targeted behavior tests before promotion.

A successful skill activation alone does not establish recovery; the
agent must change to a more productive action and then produce a
verified, correct patch. If the skill is never loaded, its value remains
unmeasured. If the control outperforms the primary, remove the skill
rather than adding more instructions. If both 31B variants improve
similarly over historical E4B, that suggests model capacity may be
important, **but it is not a causal comparison** without an E4B run
with the same exact architecture.

## Explicit non-goals

- No custom Python callbacks or forbidden host tool registration.
- No custom `run_skill_script` editor (S25 failed severely here).
- No mandatory multi-agent graph/verification stages (S31 regressed).
- No pretrained reference fixes, evaluator tampering or fabricated wins.
- No claim of runtime-enforced repetition detection/context reset.
