# Submission 38 — competition-compatible stateful skill (E4B)

**Status: buildable S1 candidate, NOT scored.** S37 was faster than S36
but remained at 3/15 and showed long editing and rereading loops.
S38 tests whether a real on-disk per-task failure ledger helps.

## Supported mechanism

The competition compiler accepts ADK skills, including Python scripts
executed with the approved run_skill_script tool inside the persistent
task sandbox. It does NOT allow arbitrary participant ADK callback code.

The state-guard skill includes one Python script implementing:
- read: return up to 100 numbered lines; block identical read range on
  unchanged file content (SHA-256 signature)
- edit: perform a bounded, exact, unique edit; store failed anchors and
  reject repeat failed edits against unchanged file content; atomic write
- status: return a compact last-action summary

The state file lives under the workspace .git/s38_state_guard.json. It
stores hashed operation identifiers, not old/new source strings, and
does not appear in the submitted git diff. Fresh sandbox = fresh state.
The script rejects symlinked or escaping paths, ambiguous replacements
and oversized edits.

**Scope limit:** Enforcement applies ONLY when E4B actually chooses
run_skill_script for read/edit. Native read_file and edit_file are NOT
intercepted, and internal reasoning is not controlled. If the model does
not use the skill, S38 has no deterministic repetition prevention.

## S36 parity

S38 uses the exact S36 primary model, six native tools, sampler and
limits. The only model-facing changes are registering one skill in
agent.yaml and appending the brief stateful_recovery_addendum.md to
the exact S36 instruction (including its parser-aware addendum).
The packaged eval_config.yaml is byte-identical to S36.

- E4B QAT model: gemma-4-e4b-it-qat-w4a16-ct
- Temperature/top_p: 0.0 / 0.95
- Max output / thinking: 4096 / 512
- Budget: 40 tool calls, 48 turns, 4 minutes

The ZIP contains five files: agent.yaml, eval_config.yaml, instruction.md,
skills/state-guard/SKILL.md, and
skills/state-guard/scripts/state_guard.py. No external callbacks, no
custom evaluator and no reference fixes.

A matching submission_38_control_s36.zip contains S36 primary unchanged.
The comparison changes skill registration and one activation prompt
together, so the cause of any difference must be established by real
run_skill_script traces. S35 previously showed that optional skills
can be unused or unhelpful.

## Build and S1 test

From Co_Kaggle/g6/tests_on_tasks/tools:

~~~bash
python -m pip install 'pyyaml>=6,<7' 'google-adk==1.36.1'
python build_submission38.py
python -m unittest -v test_submission36.py test_submission38.py

python run_submission24_eval.py --stage s1 --repeats 2 --time-limit 0 \
  --archive ../../submission_nb/artifacts/submission_38_control_s36.zip \
  --tasks /data/tasks.jsonl --snapshots /data/snapshots \
  --wheels /data/wheels --output /results/s38_control

python run_submission24_eval.py --stage s1 --repeats 2 --time-limit 0 \
  --archive ../../submission_nb/artifacts/submission_38.zip \
  --tasks /data/tasks.jsonl --snapshots /data/snapshots \
  --wheels /data/wheels --output /results/s38_primary
~~~

Both ZIPs and SHA-256 manifests are also CI artifacts. A separate
four-minute contest-style comparison must use --time-limit 4 in BOTH
arms and fresh result folders; S36/S37 historical S1 traces were
unlimited-time runs.

## What counts as success

Use official resolved outcomes, final patches, and raw ATIF traces.
Measure skill load/run calls, accepted vs rejected guarded edits, actual
blocked_duplicate feedback, repeated native edit failures, repeated
source reads, time/tokens, and retained previous wins. In particular:
S36 solved rich_4077, rich_3905 and rich_3454 in the last S1 run;
S37 solved rich_3905, rich_3518, rich_3454.

If there are zero run_skill_script calls, there is **no evidence the
mechanism was exercised**, regardless of resolution outcomes. Do not
promote just because package tests pass or calls became faster. The
original native write_file overwrite risk also remains from S36.
