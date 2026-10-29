# Submission 24 — E4B repair and independent review

**Unscored candidate.** Uses only `gemma-4-e4b-it-qat-w4a16-ct`, without LoRA.
Submission 23 and all earlier submissions are unchanged. CPU/runtime tests below
do not establish that E4B solves more tasks.

## What changes

S21 and S22 each solved 2/15 selected S1 tasks; their wins did not overlap. S22
made no optional skill calls and returned five empty patches. S24 replaces that
single long conversation with an ADK `SequentialAgent`: locate, a two-iteration
editor/reviewer `LoopAgent`, then finalization.

| Stage | Responsibility | Output passed to next stage |
|---|---|---|
| Locator | Search the relevant symbol/caller; read a short range; capture a baseline when cheap | `location`: contract, observed location, cause, check |
| Editor, twice | Native source editing; second round addresses reviewer evidence | `candidate`: actual edits and check results |
| Reviewer, twice | Inspect the actual diff, audit syntax/definitions, check requested behavior | `review`: READY / REPAIR / EMPTY / BLOCKED and evidence |
| Finalizer | Preserve and submit the existing patch, including uncertain attempts | `completion`: factual submission status |

All stages use `include_contents: none`. In the pinned ADK this keeps the current
stage's tool observations and the latest handoff, while dropping older raw tool
history. Task, harness `hints`, and short state handoffs are explicitly templated
into each working stage. Only the finalizer is given `submit_patch`; editing uses
native `edit_file`, `write_file`, and `run_command`, with no mandatory skill gateway.

Locator/reviewer receive 1,024 thinking tokens and 4,096 output tokens; editor
receives 1,536 and 6,144. `include_thoughts: false` omits raw reasoning from replay;
it does **not** disable reasoning with a positive budget in compiler 0.2.13.
Finalization uses no thinking. Concise prompts avoid Python-style fake tool-call
examples. These changes are hypotheses for improving E4B, not measured gains.

## What is enforced, and what still depends on the model

**Runtime/config enforced:** stage order, at most two loop iterations, native
editor tools, only-finalizer submit-tool registration, context mode, model alias,
generation limits, and global 40 charged tools / 48 LLM turns / 4 minutes.

**Prompt guidance:** per-stage call allocations (4 / 8 / 5 / 8 / 5 / 2), unique edit
anchors, stopping repeated reads, actually invoking the audit, relevant tests,
READY reuse and final diff inspection. A stage can still consume the global budget
before later stages run. `run_command` is a general shell tool; reviewer roles
are not an OS-enforced read-only boundary. There is no custom per-call interceptor,
hard semantic gate, guaranteed tool-format repair, or guaranteed finalizer reach.

The official harness does not supply a callback registry to submissions, so S24
does not declare unsupported callbacks or modify the evaluator. The one packaged
skill is S22's unchanged read-only structural audit, now assigned to the reviewer.
It cannot prove behavior or block native submission.

## Build and verify

From the repository root:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission24.py
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission24.py -v
```

The exact deliverable is `submission_nb/artifacts/submission_24.zip` with its
SHA-256 manifest. There are no task IDs, reference patches, evaluation tests,
model weights, or adapters inside the package.

For the official-runtime checks in a disposable Python 3.12 environment:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/install_submission24_runtime.py --wheel-dir /tmp/g6-wheels
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p 'test_submission24*.py' -v
```

The installer downloads and hash-checks the public official wheels. CI **executes**
the compiler, Runner, native tools and skill executor. Scripted tests cover syntax
failure -> reviewer feedback -> repair -> same-probe success -> one submission;
empty patches; a locator returning text instead of a call; context isolation;
hint propagation; reasoning configuration; and real `SessionTrace` serialization.

## Real E4B evaluation

Use existing `gemma-4-developer-agent-testing_v11.ipynb` with the S24 archive and
the same S1 indices: `[24,14,17,19,21,22,66,67,72,75,107,113,71,106,117]`.
Its graph-aware metadata reader already follows file-relative agent configs.
Retain its full diagnostic ZIP, including model requests, traces, patches and results.

Alternatively, after starting the normal competition vLLM server, use:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/run_submission24_eval.py \
  --tasks /kaggle/input/competitions/gemma-4-developer-agent/tasks.jsonl \
  --snapshots /kaggle/input/competitions/gemma-4-developer-agent/snapshots \
  --wheels /kaggle/input/competitions/gemma-4-developer-agent/wheels \
  --served-model gemma-4-e4b-it-qat-w4a16-ct \
  --stage pilot --output /kaggle/working/s24-pilot
```

Use the exact ID returned by `/v1/models` for `--served-model` (the notebook often
serves an absolute model path). The runner checks that it identifies E4B QAT. It
performs baseline-only verification before spending inference calls and stops on
environment-blocked baselines. It never changes benchmark tests or setup hooks.

The pilot has the four distinct S21/S22 wins plus `rich_3882`. Inspect traces before
expanding: aim for at least three solved pilot tasks, nonempty patches and no
repeated dispatch failures. Then run `--stage s1 --repeats 2` in a fresh output
directory. Record **both** runs. Use `--archive` with S21/S22 ZIPs for fresh matched
controls. Default `--time-limit 0` matches the historical S1 runs, whose notebook
disabled session timing; `--time-limit 4` exercises the scored-package limit.
Do not compare these modes as though their time budgets were identical.

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/compare_submission24.py \
  --s21 Co_Kaggle/g6/tests_on_tasks/diagnostics/submission-21-zip_s1Tasks.zip \
  --s22 Co_Kaggle/g6/tests_on_tasks/diagnostics/submission-22-zip_s1Tasks.zip \
  --s24 /kaggle/working/s24-s1/candidate_1 \
  --output /kaggle/working/s24-comparison.json
```

Predeclared practical target: **at least 6/15 solved, four net gains over each
baseline, at least three of the four prior wins retained, at most two empty
patches, and no agent errors**. This is a development target, not statistical
significance. Repeat it, inspect each diff, and then evaluate the additional S2
tasks before attributing gains to the architecture or expecting a higher hidden score.

See [validation and design evidence](../../tests_on_tasks/reviews/submission_24/validation.md).
