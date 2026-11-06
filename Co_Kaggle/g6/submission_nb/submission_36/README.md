# Submission 36 — parser-aware native edit experiment

**Status: unscored research candidate.** S36 tests whether teaching Gemma 4's
actual tool-argument string delimiters reduces malformed native editing calls.
It does **not** fix the host-side vLLM parser, add a tool wrapper, or assume the
model will follow an example.

## One hypothesis, one variable

The exact S29 E4B agent is the control. The primary appends only
[parser_aware_addendum.md](parser_aware_addendum.md) to S29's instruction.
Both archives contain the same **byte-identical** S29 `agent.yaml` and
`eval_config.yaml`. The original S29 prompt is unchanged in the control.
There are no new skills, agents, graph tools, adapters, or runtime callbacks.

| Dimension | S36 control | S36 primary |
| --- | --- | --- |
| Model | `gemma-4-e4b-it-qat-w4a16-ct` | **same** |
| Native tools | run_command, read_file, edit_file, write_file, get_status, submit_patch | **same** |
| Temperature / top-p | 0.0 / 0.95 | **same** |
| Thinking budget / include_thoughts | 512 / false | **same** |
| Max output | 4096 tokens | **same** |
| Limit | 4 min / 40 tool calls / 48 turns | **same** |
| S29 instruction | unchanged | unchanged + delimiter example |

The additive note shows `<|"|>` delimiters for string-valued tool parameters.
The legacy vLLM Gemma 4 API parser can mishandle ordinary quote characters,
commas and brackets in arguments. Upstream evidence:
[vLLM #51284](https://github.com/vllm-project/vllm/issues/51284)
and [vLLM v0.19.1 parser](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/tool_parsers/gemma4_tool_parser.py).
These are *plausible mechanisms*, not proof that S29/S35 errors were caused by
that parser. The current Kaggle wheelhouse may differ. Literal special tokens
in prompts may themselves impair model behavior: this is why the matched
control is mandatory.

**Safety caveat:** `write_file` remains present for strict S29 parity and
is therefore an **existing-file overwrite risk**. The model's no-overwrite
instruction is NOT enforced. Do not promote a candidate with destructive
diffs. Shell commands also remain unrestricted.

## Build and CPU validation

From `Co_Kaggle/g6/tests_on_tasks/tools`:

```bash
python -m pip install 'pyyaml>=6,<7'
python build_submission36.py
python -m unittest -v test_submission36.py
```

Generates two deterministic ZIP archives in `submission_nb/artifacts`:

- `submission_36_control.zip` — exact S29 member contents (ZIP metadata may differ from the archived S29 ZIP)
- `submission_36.zip` — parser-aware primary

Each ZIP has exactly three root files: `agent.yaml`,
`eval_config.yaml`, and `instruction.md`. The builder also emits
SHA-256 manifests adjacent to each ZIP. CI builds and uploads both ZIPs;
the binaries are generated artifacts, not checked into Git. CPU checks verify
package/configuration determinism, **not model tool-call correctness**.

## Real E4B evaluation — mandatory before claiming gains

Use the same E4B model server, official wheels, snapshots, task JSONL,
compaction configuration, and Python environment for both variants. Capture
the actual vLLM version and wheelhouse hashes. Keep the evaluation scorer
unchanged.

The existing `run_submission24_eval.py` runner handles the official
baseline preflight, evaluation and raw ATIF/JSONL recording. First build
packages as above. For the known five-task pilot:

```bash
python run_submission24_eval.py --stage pilot --repeats 2 --time-limit 4 \
  --tasks /path/tasks.jsonl --snapshots /path/snapshots --wheels /path/wheels \
  --archive ../../submission_nb/artifacts/submission_36_control.zip \
  --output /path/s36_control_pilot

python run_submission24_eval.py --stage pilot --repeats 2 --time-limit 4 \
  --tasks /path/tasks.jsonl --snapshots /path/snapshots --wheels /path/wheels \
  --archive ../../submission_nb/artifacts/submission_36.zip \
  --output /path/s36_primary_pilot
```

If the baseline-only preflight reports `baseline_blocked`, **stop**:
investigate missing fixtures, dependency mismatches, wrong imports, or task
provenance. Do not edit graders or silently count an invalid baseline as an
agent failure. If the pilot establishes tool-call reliability, repeat the
same two commands using `--stage s1`, and use distinct, unused output dirs.
Also run separately with `--time-limit 0` to compare against historical
unlimited-time diagnostics; never combine timed and untimed scores.

For each full 15-task repeat, compare complete paired JSONL results:

```bash
python analyze_submission34.py \
  /path/s36_control_s1/candidate_1 /path/s36_primary_s1/candidate_1 \
  --format jsonl
```

For printed 15-task traces, use the same analyzer's `--format printout`
to report malformed edit calls, edit failures, first-edit validity, excess
submissions, duration, and retained resolved IDs. Printed diagnostics may
elide actual arguments, so inspect raw ATIF events / server traces as well.

## What to measure

- **First edit-call validity**, and valid parsed keys `filepath`,
  `old_string`, `new_string`. Separately count parser rejection,
  missing fields, and exact-anchor-not-found failures.
- **First effective edit rate**, distinct from a dispatched but rejected
  or no-op call. Record repeated unchanged operations after errors.
- **Task resolution and retention**: S29 resolved `rich_3905`,
  `rich_3518`, and `rich_3454` in one untimed diagnostic run.
- **Safety and time**: empty/oversized patches, unrelated modified files,
  destructive overwrites, repeated submit calls, and four-minute failures.

These are diagnostic measurements, not a claim that we can access raw model
tokens from every evaluation environment. If raw emissions are unavailable,
mark root-cause attribution **unconfirmed**.

**Promotion target:** two matched 15-task runs each with >=5/15 resolved,
retention of the three historical S29 wins, no destructive patch, and
demonstrably better tool-call validity. Then evaluate held-out tasks from
Rich, Requests and FastAPI. A single successful run is not sufficient.

## Following experiments — deliberately not mixed into S36

After the parser experiment, perform separate matched ablations for
`thinking_budget: 0` versus 512, then sampling temperature 0.0 versus
0.5. Each must keep the previous experiment's model, tool surface, task
set, and other generation options fixed. Do **not** vary these parameters
inside S36's parser experiment.
