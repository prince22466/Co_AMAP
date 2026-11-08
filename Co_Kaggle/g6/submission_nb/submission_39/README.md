# Submission 39 — stacked S37 + S38 (Gemma 4 E4B)

**S1 candidate; no measured S39 task score yet.**

## Single-variable hypothesis

Does adding S37's observed lower-token, model-guided error memory to S38's
competition-supported persistent skill improve task resolution or at least
trigger its previously unused skill? S37 solved **3/15** S1 tasks, S38
**2/15**. S38's S1 trace had **zero visible load_skill/run_skill_script**
calls. Merely registering a skill did not make E4B use it.

This is deliberately a **prompt-only delta versus S38**, not a new
agent graph, a different model, another tool, or an unapproved callback.

## Exact composition

The builder produces a valid ZIP with five root entries:

    agent.yaml
    eval_config.yaml
    instruction.md
    skills/state-guard/SKILL.md
    skills/state-guard/scripts/state_guard.py

S39's agent.yaml and eval_config.yaml are **byte-for-byte S38 primary**;
the reviewed skill SKILL.md and Python script are **byte-for-byte S38**.
The model is still gemma-4-e4b-it-qat-w4a16-ct, using six familiar native
tools with temperature 0.0, top_p 0.95, max_output_tokens 4096, thinking
budget 512, and the S36/S38 4-minute/40-tool/48-turn limits.

The **sole S38-to-S39 change** is:

    S39 instruction =
      S36 primary instruction (including parser-aware guidance)
      + S37 failure-memory instruction
      + S38 stateful-skill activation instruction

The original S37 text is retained **except for exactly two compatibility
edits**: (1) its header now allows S38's skill, and (2) "do not spend
tool calls writing a ledger" now prohibits only *manual native-tool*
ledgers and clarifies that `load_skill` is a setup call outside the
**two `run_skill_script` recovery actions** (one source read, one exact edit).
The script's automatic state persistence remains permitted.
The builder reconstructs and validates the original S37 Git blob after
undoing these two edits. S38's activation paragraph is unchanged.

The skill's deterministic duplicate detection still applies **only to
operations invoked through run_skill_script**. Native read_file and
edit_file are **not intercepted**. The model must recognize a failure and
activate the skill. Do NOT claim universal state enforcement or a
guaranteed reduction in E4B retry loops.

## Build and CPU verification

From Co_Kaggle/g6/tests_on_tasks/tools:

~~~bash
python -m pip install 'pyyaml>=6,<7' 'google-adk==1.36.1'
python build_submission39.py
python -m unittest -v test_submission36.py test_submission38.py test_submission39.py
~~~

The resulting files are:

- submission_nb/artifacts/submission_39.zip
- submission_nb/artifacts/submission_39_control_s38.zip
- accompanying SHA-256 manifests

The CI workflow separately compiles **both** extracted archives using the
pinned upstream restricted adk_submission compiler (revision
528792a91d061d601830237543103ecd0432eb7b). Successful CPU/package
checks do not prove E4B will actually use the stateful skill.

## Real S1 test protocol

Use identical E4B QAT model, task/snapshot data, evaluator wheels and
session limits for both variants. From tests_on_tasks/tools:

~~~bash
# Historical unlimited-time S1 comparison (15 selected tasks):
python run_submission24_eval.py --stage s1 --repeats 2 --time-limit 0 \
  --archive ../../submission_nb/artifacts/submission_39_control_s38.zip \
  --tasks /data/tasks.jsonl --snapshots /data/snapshots \
  --wheels /data/wheels --output /results/s39_control

python run_submission24_eval.py --stage s1 --repeats 2 --time-limit 0 \
  --archive ../../submission_nb/artifacts/submission_39.zip \
  --tasks /data/tasks.jsonl --snapshots /data/snapshots \
  --wheels /data/wheels --output /results/s39_stacked
~~~

Also run a separate contest-style comparison with --time-limit 4 in BOTH
arms. Previous S36/S37/S38 S1 diagnostics had disabled per-task time limits;
a win requiring more than four minutes cannot be counted as timed success.

### Primary measurement / stopping gate

Compare actual phase-two resolved flags and:
- Whether the agent used load_skill and run_skill_script after a failed
  native edit (if **zero**, stacking did not activate enforced memory)
- Successful/rejected/blocked_duplicate script outcomes and native edit
  failure loops; printed errors can hide argument details
- Task-wise retained wins on rich_3905, rich_3454, rich_3518, rich_4077
- Empty/unsafe patches, correct implementation site, target-specific tests
- End-to-end duration and observed tokens (not a substitute for resolution)

Five-task low-cost pilot: rich_3518, rich_3938, rich_3676, rich_4077,
rich_3454. If the skill is unused and accuracy does not improve, do not
keep appending longer instructions. Reassess whether a competition-permitted
*automatic* interception mechanism exists before another skill experiment.

**Do not promote S39 before actual E4B testing.** A single 15-task
run can vary stochastically even with temperature 0.
