# Submission 32 — guarded, single-line edit recovery experiment

**Unscored experiment.** Generated ZIPs cannot prove real Gemma E4B accuracy.

The primary `submission_32.zip` is a single native-tool E4B agent that
edits via `edit_file` FIRST. After TWO failed exact anchors at the same
location, it can use a strictly scoped one-line atomic replacement using
Python's standard library through existing `run_command`. It checks:
existing `.py` file smaller than 200,000 bytes, exact original 1-indexed
line, replacement lacks newline and is shorter than 400 chars, resulting
file parses with `ast.parse`, then same-directory `os.replace`.

This targets the recurring *editing* failure mechanism. It neither
adds new tools nor claims host-enforced shell safety. The control ZIP
`submission_32_control.zip` differs ONLY in instructions: it has no
transactional fallback and expressly prohibits shell edits. Both use
the same agent.yaml and eval_config.yaml. Both exclude graph tools and
`write_file`. This is an existing-file S1 ablation, **not** a suitable
permanent tool policy for tasks that legitimately need new files.

## Build and tests

```bash
cd Co_Kaggle/g6/tests_on_tasks/tools
python build_submission32.py
python -m unittest -v test_submission32.py
python smoke_submission32.py  # requires official Kaggle runtime wheels
```

Build writes the primary and control ZIP plus JSON SHA-256 manifests.
The ZIP contains ONLY `agent.yaml`, `eval_config.yaml`, `instruction.md`;
the README/research and local tests are not included in the submission.
Unit tests validate reproducibility and simulate guarded fallback on
temporary Python files, including syntax, line mismatch and size rejection.
They are NOT substitutes for real E4B evaluation.

## Matched real-model evaluation

Use existing official runner, keeping each variant and output distinct:

```bash
python run_submission32_eval.py \
  --tasks /path/to/tasks.jsonl --snapshots /path/to/snapshots \
  --wheels /path/to/wheels --variant primary --stage pilot \
  --repeats 2 --time-limit 0 --output /path/to/s32-primary-pilot
python run_submission32_eval.py \
  --tasks /path/to/tasks.jsonl --snapshots /path/to/snapshots \
  --wheels /path/to/wheels --variant control --stage pilot \
  --repeats 2 --time-limit 0 --output /path/to/s32-control-pilot
```

If pilot doesn't regress seriously, repeat both with `--stage s1`.
Historical S29–S31 printouts disabled the session limit. For
comparability use `--time-limit 0`; separately repeat with
`--time-limit 4` to approximate the submitted eval_config budget.
Never compare unlimited results against timed runs as a measured gain.

The automatic stage runner checks resolved task IDs and repeats but does
NOT certify the shell fallback. Use `gate_submission32.py` on each new
printed S1 log, and inspect full ATIF tool arguments, exact actual diffs,
local assertion semantics and file-size guard for any shell edits.

**Target:** >=5/15 in each of 2 runs, S29 wins
(`rich_3905`, `rich_3518`, `rich_3454`) retained, >=1 of S30 gains
(`rich_4077`, `rich_3882`) retained, no destructive files, no
repeated anchor failures. Do NOT choose a lucky repeat.

See [research.md](research.md) for sources, alternative architectures
rejected, and limitations.
