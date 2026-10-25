# Agent development workflow

The current [Submission 16 candidate](../submission_nb/submission_16/README.md) addresses the
missing-action, repeated-failure accounting and absent verification seen in Submission 15's
first-ten diagnostic run. Use [testing v9](gemma-4-developer-agent-testing_v9.ipynb): CPU preflight
first, then a real-Gemma repair smoke gate before the unchanged ten-task evaluation. Its 19 CPU
regressions and scripted official-runtime smoke pass locally; real-model success remains unmeasured.
See [the validation record](reviews/submission_16/validation.md) for scope and remaining limits.

`submission_1/` preserves the first main-coder/optional-analyzer configuration. `submission_2/` revises only its two instruction files. `submission_3/` adds an optional read-only patch reviewer. `submission_4/` uses a coder, repository locator and test verifier. `submission_5/` changes Submission 4's timeout and thinking settings. `submission_6/` copies Submission 2 and changes only both agents' model declarations to the newly allowed `gemma-4-e2b-it-qat-w4a16-ct`. Local validation, packaging, and diagnostics are in `tools/`; the companion notebooks run the competition evaluator. Reports and notebooks stay outside the packaged agent.

The submission records report Submission 1 at 0.10 and 0.06, Submission 2 at 0.10, and Submissions 3, 4 and 5 at 0.0. Submission 6 is unmeasured. See [the E2B candidate](submission_6/README.md), [the organizer model update](../competition_material/Update_new_models.md), [the submission records](../submission_records.md) and [the experiment record](prompt_experiments.md). Earlier unmeasured statements in historical candidate notes are superseded by these reported scores.

## Local commands

Run from the project root. These commands work in CMD as well as PowerShell:

```bat
python submission_nb\tools\submission_tools.py validate --submission-dir submission_nb\submission_1
python submission_nb\tools\submission_tools.py build --submission-dir submission_nb\submission_1 --output submission_nb\artifacts\submission_1.zip
python submission_nb\prepare_evaluation_notebook.py
python submission_nb\tools\submission_tools.py build --submission-dir submission_nb\submission_2 --output submission_nb\artifacts\submission_2.zip
python submission_nb\prepare_evaluation_notebook.py --submission-dir submission_nb\submission_2 --baseline-dir submission_nb\submission_1 --output submission_nb\evaluate_submission_2.ipynb
python submission_nb\tools\submission_tools.py build --submission-dir submission_nb\submission_3 --output submission_nb\artifacts\submission_3.zip
python submission_nb\prepare_evaluation_notebook.py --submission-dir submission_nb\submission_3 --reviewer-control --output submission_nb\evaluate_submission_3.ipynb
python submission_nb\tools\submission_tools.py build --submission-dir submission_nb\submission_4 --output submission_nb\artifacts\submission_4.zip
python submission_nb\prepare_evaluation_notebook.py --submission-dir submission_nb\submission_4 --verification-control --output submission_nb\evaluate_submission_4.ipynb
python submission_nb\tools\submission_tools.py build --submission-dir submission_nb\submission_5 --output submission_nb\artifacts\submission_5.zip
python submission_nb\prepare_evaluation_notebook.py --submission-dir submission_nb\submission_5 --runtime-baseline-dir submission_nb\submission_4 --output submission_nb\evaluate_submission_5.ipynb
python submission_nb\tools\submission_tools.py build --submission-dir submission_nb\submission_6 --output submission_nb\artifacts\submission_6.zip
python submission_nb\prepare_evaluation_notebook.py --submission-dir submission_nb\submission_6 --output submission_nb\evaluate_submission_6.ipynb
python -m unittest discover -s submission_nb\tools -p "test_*.py"
python -m unittest discover -s submission_nb -p "test_evaluation_workflow.py"
```

The tools use Python and PyYAML; notebook generation and workflow tests also need `nbformat`. `build` resolves `agent.yaml`, the optional `eval_config.yaml`, recursive includes, configured helpers, and referenced skills/adapters. Only those dependencies enter the ZIP, with `agent.yaml` at its root. The validator permits the organizer-listed 31B, 12B, E4B and E2B quantized variants, and requires one variant throughout each submission. Local validation checks the documented supported configuration; the installed competition compiler remains authoritative.

Regenerate the corresponding notebook after changing the submission or local tools. Its embedded files are a snapshot at generation time. Architecture mode can also use a mounted candidate directory for future submissions containing binary adapters; prompt-comparison mode fixes the unchanged model and rejects adapter experiments so they do not confound the prompt test.

## Run the comparison

Upload/open a companion notebook in a Kaggle environment with the competition data, selected Gemma model, wheelhouse, and adequate GPU memory. Inspect the parameter cell before running it. `evaluate_submission_6.ipynb` compares E2B coder-only and E2B coder-plus-analyzer variants; the full two-agent candidate is its default release. It uses the supplied updated starter's installation cell and selects E2B weights. To compare against Submission 2's 31B model, run separate model sessions using the same explicit `TASK_IDS`; each notebook run serves one variant of Gemma.

`evaluate_submission_5.ipynb` compares Submission 4, timeout-only and thinking-off-only controls, and Submission 5. All retain the same three agents, prompts and four-minute allowance; only timeout and thinking differ. Effective behavior needs confirmation in the installed runtime.

`evaluate_submission_4.ipynb` remains the helper comparison between `coder_only`, `optional_locator`, `optional_verifier` and `locator_verifier` under Submission 4's settings. Helpers search or execute focused checks; only the coder has editing and submission tools. Shell restrictions are prompt instructions.

`evaluate_submission_3.ipynb` compares `coder_only`, `optional_analyzer` and `optional_reviewer` under the same Submission 3 coder prompt, model, sampling and total budget. Its two-agent control removes only the reviewer and differs from the original Submission 2 coder prompt.

`evaluate_submission_2.ipynb` remains the prompt comparison between Submission 1 (`baseline`) and Submission 2 (`candidate`); both retain the two-agent configuration and identical runtime settings.

The original `evaluate_submission_1.ipynb` remains available for its architecture comparison:

- `coder_only`: identical main coder with the analyzer connection removed.
- `optional_analyzer`: the selected candidate. Delegation remains the coder's decision.

Both variants run the same selected task IDs. The default of two tasks checks installation and execution; it does not establish which variant performs better. Pin `TASK_IDS` for repeatable experiments, then expand to a development set spanning repositories. Keep a separate validation set when selecting prompts or adapters. The notebook's manifest records the selected tasks and configuration hashes.

Runs use fresh directories under `/kaggle/working/agent_runs/`, retaining configuration snapshots and results. The release variant is packaged to `/kaggle/working/submission.zip`. Local diagnostics and comparison variants are not included.

The downloaded Windows data currently contains two snapshots and graphs/embeddings for different commits; it has no task with all matching artifacts. The notebook performs a coverage check before inference. Obtain matching task data or use the full mounted competition dataset. Do not mix graphs from another commit with a snapshot.

The notebook initially uses the starter's subprocess sandbox. Compare in the scoring Docker environment before promotion. Its model context, GPU utilization, caching, and compaction settings follow the documented scoring values where the starter API supports them; this does not prove complete scoring parity.

## Inspect the public tasks before evaluation

The [task catalog](task_inventory/task_catalog.md) covers all 129 public tasks
using their descriptions and agent-visible metadata. Reference fixes and grading
tests are excluded. Each task has a manually reviewed technical-area category,
a change kind, a short summary, classification confidence and body-detail flags.
Categories are not difficulty estimates or claims about the hidden test set.

Open [task_inventory.ipynb](task_inventory/task_inventory.ipynb) for the executed
exploration, repository/category charts, description-quality checks and the full
filterable task table. [task_catalog.csv](task_inventory/task_catalog.csv) includes
the original problem descriptions for spreadsheet browsing. Edit
[task_inventory_labels.tsv](task_inventory_labels.tsv) to revise classifications,
then regenerate the outputs:

```bat
python submission_nb\tools\profile_tasks.py
python submission_nb\tools\test_profile_tasks.py
```

[test_sets.json](task_inventory/test_sets.json) defines nested 4-task smoke,
12-task pilot and 24-task category-coverage sets, plus 90 development and 39
public validation tasks. Shared repository/base-commit snapshots stay in one
split, and the two previously tested FastAPI tasks stay in development. Related
PRs at other commits can still overlap conceptually. The single HTTPX task is
in development, so the validation set cannot independently measure HTTPX.

Replace `TASK_IDS` and `TASK_TYPES` in the Kaggle opening controls with the
copyable assignments in
[kaggle_task_selection.py](task_inventory/kaggle_task_selection.py). Its default
is the 12-task pilot; choose `extended_24`, `development`, `validation` or
`all_public` explicitly when expanding. The smoke/pilot/extended sets overlap by
design and must not be counted as independent samples. All 129 task-type labels
are included, so later resource summaries can group by technical area.
The six evaluation notebooks include additional control variants: distinguish
those variants from the six final submitted candidate configurations when
estimating run counts and comparing results.

## Read the diagnostics

For the starter notebook, replace its evaluation cell with the complete contents
of [kaggle_evaluation_cell.py](kaggle_evaluation_cell.py), after the existing
data/model setup. Alternatively, copy the three code cells from
[kaggle_evaluation_diagnostics.ipynb](kaggle_evaluation_diagnostics.ipynb).
Regenerate these companions with `python submission_nb/prepare_diagnostic_cell.py`.
The evaluation notebooks for Submissions 1–6 include the same instrumentation.

Observer version `2.0-task-context` fixes the cross-task model-hook leak found in
`20261014T221140.821885Z_ib98kd08.zip`. When upgrading a kernel that ran the old
cell, restart its session once to clear retained legacy hooks, then run the
normal model/data setup and the updated cell. New manifests and observer status
files record the diagnostic version. Explicit session-timeout errors now populate
`stop_reason` with their provenance; elapsed duration alone never establishes it.

Version `2.1-iteration-records` additionally saves exact configuration/prompt
bytes under `configuration_snapshot/`, public task descriptions/commit IDs in
`selected_tasks.json`, returned verification output in `test_outputs/`, available
serving settings, and model finish reasons. Model responses received before a
consumer closes its generator are labelled `response_received`, rather than
being classified as a model interruption solely because of `GeneratorExit`.
Requests cancelled before a response still have unknown token usage.

Version `2.2-result-serialization` fixes saving completed evaluator results that
contain an arbitrary `SessionTrace` object. Results are converted from Pydantic's
Python mode with recursive conversion, credential redaction and cycle handling.
Returned patches and verification output are saved before the full result;
an artifact-saving error is recorded separately from the evaluator's outcome.
Agent observer context is scoped to each generator advance/close, so a consumer
closing the agent in another task does not leak context or label ordinary
consumer closure as an agent failure. Tool argument-validation responses with
an `error` field are recognized as errors. The archive auditor now checks patch
length against the actual returned patch and flags missing result artifacts.
These fixes are covered offline, including a Pydantic reproduction of the
`20261015T212051.926673Z_ugqjy2rs.zip` failure; real Kaggle execution is still
needed to confirm complete verification-log capture with the updated cell.

The starter replacement now places the evaluation controls in its opening lines
and **disables the per-task session timer by default**, including when the
submission YAML specifies one minute. It explicitly sets
`EvaluationBudget(time_minutes=None)`; passing only the flat
`max_time_minutes=None` would inherit the harness's 60-minute budget.
The opening `EVAL_OVERRIDES` block exposes time, tool-call, turn, total-token,
cost, command/verification timeout and tool-output limits. `INHERIT` preserves
the submission value or documented default. A positive number sets a limit;
`None` removes a supported budget. `max_turns=None` still uses the harness's
500-turn fallback. Other limits retain their existing values by default, so the
sample still allows 10 tool calls and 50 turns. Command and verification
timeouts are separate from the disabled session timer.

Task selection, cache and compaction settings also appear at the top.
`TASK_LIMIT=None` selects all tasks, `TASK_IDS` pins tasks, and `TASK_TYPES`
assigns labels before inspecting outcomes. A missing label stays `unclassified`.
Per-response output/thinking limits remain in the agent's sampling YAML, and
the model context limit remains in the earlier vLLM setup. These controls apply
to the standalone replacement and its companion notebook; the six submission
evaluation notebooks continue using their variant budgets.
The cell prints the effective limits and saves them, the structured budget and
harness limits in `run_manifest.json` before running.

Each run saves per-task outcomes, returned patches, complete serializable
results, original harness artifacts, a console transcript and filtered host
logs. The run manifest records configuration hashes, declared models, agent
sampling settings, effective evaluation parameters and runtime versions. The
updated `verification_timeout_seconds` is passed separately when supported by
the installed `EvalConfig`; older runtimes receive a recorded warning.

Temporary notebook observers wrap ADK `BaseAgent.run_async` and
`LiteLlm.generate_content_async`, pass through original results, close their
generators, and retire the task's observer after each task. Methods are restored
when they are still owned by the observer. A runtime retry wrapper can retain a
passive dispatcher, which selects the current task through a context variable;
it does not keep writing into the previous task's files. They capture nested helper
activity, native tool call IDs, visible answers, parent run IDs, model usage and
timing. They do not record thought text or request credentials. Capture is
limited to the current process and these entry points; absent helper usage is
reported as a gap. `LIVE_AGENT_OBSERVER = False` disables the hooks. These
observers do not enter submission packages. Integration references:
[ADK AgentTool](https://github.com/google/adk-python/blob/main/src/google/adk/tools/agent_tool.py),
[BaseAgent](https://github.com/google/adk-python/blob/main/src/google/adk/agents/base_agent.py),
[LiteLlm](https://github.com/google/adk-python/blob/main/src/google/adk/models/lite_llm.py).

The most useful files under `report/` are:

| File | Purpose |
|---|---|
| `task_summary.csv` / `.json` | Solved status, patch size, durations, token usage, stop reason, errors and coverage warnings |
| `requests.csv` | Individual observed model requests, agent attribution, sampling configuration and latency |
| `agent_usage.csv` | Request usage and model time by task and agent |
| `stage_usage.csv` / `agent_stage_usage.csv` | Usage by inferred stage, with unknown/mixed categories |
| `task_type_summary.csv` | Outcomes, median task time and usage grouped by user-assigned task type |
| `<task_id>_interactions.csv` / `.json` | Chronological agent starts, delegation requests/responses, tool exchanges and visible answers |
| `<task_id>_tools.csv` / `.json` | Tool arguments, results and matching call/result intervals |
| `<task_id>_measurement_health.json` | Usage coverage, live/ATIF reconciliation, pending tool calls and stdout coverage |
| `<task_id>_agent_runs.csv` / `.json` | Agent run/parent IDs and inclusive elapsed time; nested durations overlap |
| `<task_id>_interaction_log.txt` | Readable visible interaction previews when Rich widgets bypass stdout capture |
| `summary.json` | Outcome coverage and measurement definitions |

`observer/<task_id>/` retains append-only event/request logs and hook status.
Each completed variant produces a diagnostic ZIP outside its results folder;
the starter replacement writes under `/kaggle/working/diagnostic_runs/`.
Download that ZIP to preserve the complete testing record.

Audit a downloaded archive without loading models or extracting its files:

```bat
python submission_nb\tools\audit_evaluation.py submission_nb\YOUR_RUN.zip --output-dir submission_nb\diagnostics\YOUR_RUN
```

The audit checks raw/status/report request counts, task attribution, token sums,
ATIF reconciliation, snapshot hashes and outcome consistency. It flags missing
usage and pending tool outcomes without assigning invented values. It also
records whether real helper calls, nonempty patches, verification exits and
solved outcomes have been observed. No such coverage is inferred from an
agent merely being configured. Empty stdout files are expected when the runtime
uses Rich widgets; the derived interaction log and structured logs remain useful.

Before a longer architecture comparison, validate two additional Kaggle cases:
a helper actually being invoked with its response and request usage captured,
and a nonempty patch reaching verification with the returned test output saved.
Then use the intended submission directory, fixed task IDs and labels, matched
budgets, and the same instrumentation version across variants. The supplied
two-task, one-minute starter runs validate instrumentation and timeout reporting;
they do not establish an architecture or prompt ranking. Snapshot files cover
declared configuration/prompt dependencies, while mounted models and adapter
weight files remain external inputs.

Request usage comes from live observation when available, otherwise ATIF; the
two are never added together. Cached/reasoning counts retain provider semantics
and remain unknown if absent. Model generator active time measures time spent
awaiting the SDK/server, excluding consumer pauses. Streaming request wall time
can include pauses between chunks. Parallel/nested times overlap and must not
be summed into task wall time. Stage labels are inferred from roles/actions;
the code does not claim an exact split between code, thoughts and tool-argument
tokens. An absent stop reason remains unknown. Instrumentation adds some
overhead, so use the same observer settings across comparisons.

The observer and reporting paths have offline tests with the supplied ATIF
traces and simulated nested agents. Real Gemma execution, the installed Kaggle
wheelhouse and scoring Docker parity still require testing in Kaggle.

After downloading a results directory, run:

```bat
python submission_nb\tools\submission_tools.py report --results-dir path\to\results --output-dir submission_nb\diagnostics\run_01 --submission-dir submission_nb\submission_1
```

The report writes Markdown, JSON, and CSV. Preserve the original run manifest and artifacts alongside it. If reporting an older run, point `--submission-dir` at that run's saved configuration, so budget comparisons use the configuration actually evaluated.

| Observation | Evidence and next experiment |
|---|---|
| Setup or model error | Check exception and setup logs before tuning agent behavior. |
| No patch | Inspect searches, reads, elapsed time, and stop reason; improve localization or shorten exploration. |
| Tool or turn exhaustion | Inspect actual counted usage and repeated actions; simplify the loop before increasing budgets. |
| Time limit with a nonempty patch | Inspect verification separately. A timer stop does not itself establish failure or success. |
| Patch/application/test failure | Read the exact diff and verification output; improve implementation or targeted tests. |
| Context-length error | Inspect per-request input/output usage and compaction evidence; reduce large reads and outputs. |
| Many analyzer calls with no improvement | Compare against `coder_only`, restrict delegation, or revise the helper's instructions. |
| Reviewer overhead or unhelpful findings | Compare `optional_reviewer` against `optional_analyzer`; inspect findings and repairs rather than inferring benefit from call counts. |
| Locator/verifier overhead or inconclusive checks | Use Submission 4's four helper combinations; inspect actual shell commands, test outcomes and subsequent coder repairs. |
| Long reasoning or grading timeout | Use Submission 5's isolated runtime controls; inspect effective thinking settings and the shared command/grading timeout. |
| Missing trace | Preserve logs and result fields; enable the runtime's trace export before claiming detailed stage timings. |

Resolution rate uses the reported verification outcome. Missing outcomes remain unknown. Task duration can include setup and verification, whereas the agent timer excludes setup. Model-request counts do not establish the harness's enforced turn count. Aggregate run tokens do not measure context size; context pressure needs per-request usage. `get_status` and `submit_patch` are exempt from counted tool calls, while helper delegation is shown separately when identifiable.

Some metrics depend on the runtime's trace schema: requests, tool frequencies, successful edit timing, and per-request tokens. Missing measurements remain null with warnings. Failure categories and suggested experiments are diagnostics based on recorded evidence, not causal proof.

## Promote and extend

Start with the unchanged model, compare the two structures, and keep the variant that solves more tasks within the total inference allowance. Inspect per-repository results and the failure reports, then change one component and repeat the same tasks. Use adapter training only after collecting a stable baseline and evidence of the behavior to improve.

See [submission configuration and extension points](submission_1/README.md) and [architecture rationale](agent_system_selection.md). Before final submission, run the authoritative compiler, evaluation with matching data, Docker parity checks, and a full inference-time check. No new GPU benchmark has been run in this Windows workspace.
