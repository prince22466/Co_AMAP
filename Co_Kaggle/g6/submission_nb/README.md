# Agent development workflow

`submission_1/` preserves the first main-coder/optional-analyzer configuration. `submission_2/` revises only its two instruction files, retaining the same two-agent configuration and budgets. Local validation, packaging, and diagnostics are in `tools/`; the companion notebooks run the competition evaluator. Reports and notebooks stay outside the packaged agent.

The user reported Submission 1 scores of 0.10 and 0.06, without task-level logs. Submission 2 is an unmeasured follow-up. See [its changes](submission_2/README.md) and [the experiment record](prompt_experiments.md).

## Local commands

Run from the project root. These commands work in CMD as well as PowerShell:

```bat
python submission_nb\tools\submission_tools.py validate --submission-dir submission_nb\submission_1
python submission_nb\tools\submission_tools.py build --submission-dir submission_nb\submission_1 --output submission_nb\artifacts\submission_1.zip
python submission_nb\prepare_evaluation_notebook.py
python submission_nb\tools\submission_tools.py build --submission-dir submission_nb\submission_2 --output submission_nb\artifacts\submission_2.zip
python submission_nb\prepare_evaluation_notebook.py --submission-dir submission_nb\submission_2 --baseline-dir submission_nb\submission_1 --output submission_nb\evaluate_submission_2.ipynb
python -m unittest discover -s submission_nb\tools -p "test_*.py"
python -m unittest discover -s submission_nb -p "test_evaluation_workflow.py"
```

The tools use Python and PyYAML; notebook generation and workflow tests also need `nbformat`. `build` resolves `agent.yaml`, the optional `eval_config.yaml`, recursive includes, configured helpers, and referenced skills/adapters. Only those dependencies enter the ZIP, with `agent.yaml` at its root. This excludes this documentation, notebooks, diagnostics, and preserved reference weights. Local validation checks the documented supported configuration; the installed competition compiler remains authoritative.

Regenerate the corresponding notebook after changing the submission or local tools. Its embedded files are a snapshot at generation time. Architecture mode can also use a mounted candidate directory for future submissions containing binary adapters; prompt-comparison mode fixes the unchanged model and rejects adapter experiments so they do not confound the prompt test.

## Run the comparison

Upload/open a companion notebook in a Kaggle environment with the competition data, required Gemma model, wheelhouse, and adequate GPU memory. Inspect the parameter cell before running it. `evaluate_submission_2.ipynb` compares Submission 1 (`baseline`) with Submission 2 (`candidate`); both retain the two-agent configuration and identical runtime settings. This is the prompt comparison to run next.

The original `evaluate_submission_1.ipynb` remains available for its architecture comparison:

- `coder_only`: identical main coder with the analyzer connection removed.
- `optional_analyzer`: the selected candidate. Delegation remains the coder's decision.

Both variants run the same selected task IDs. The default of two tasks checks installation and execution; it does not establish which variant performs better. Pin `TASK_IDS` for repeatable experiments, then expand to a development set spanning repositories. Keep a separate validation set when selecting prompts or adapters. The notebook's manifest records the selected tasks and configuration hashes.

Runs use fresh directories under `/kaggle/working/agent_runs/`, retaining configuration snapshots and results. The release variant is packaged to `/kaggle/working/submission.zip`. Local diagnostics and comparison variants are not included.

The downloaded Windows data currently contains two snapshots and graphs/embeddings for different commits; it has no task with all matching artifacts. The notebook performs a coverage check before inference. Obtain matching task data or use the full mounted competition dataset. Do not mix graphs from another commit with a snapshot.

The notebook initially uses the starter's subprocess sandbox. Compare in the scoring Docker environment before promotion. Its model context, GPU utilization, caching, and compaction settings follow the documented scoring values where the starter API supports them; this does not prove complete scoring parity.

## Read the diagnostics

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
| Missing trace | Preserve logs and result fields; enable the runtime's trace export before claiming detailed stage timings. |

Resolution rate uses the reported verification outcome. Missing outcomes remain unknown. Task duration can include setup and verification, whereas the agent timer excludes setup. Model-request counts do not establish the harness's enforced turn count. Aggregate run tokens do not measure context size; context pressure needs per-request usage. `get_status` and `submit_patch` are exempt from counted tool calls, while helper delegation is shown separately when identifiable.

Some metrics depend on the runtime's trace schema: requests, tool frequencies, successful edit timing, and per-request tokens. Missing measurements remain null with warnings. Failure categories and suggested experiments are diagnostics based on recorded evidence, not causal proof.

## Promote and extend

Start with the unchanged model, compare the two structures, and keep the variant that solves more tasks within the total inference allowance. Inspect per-repository results and the failure reports, then change one component and repeat the same tasks. Use adapter training only after collecting a stable baseline and evidence of the behavior to improve.

See [submission configuration and extension points](submission_1/README.md) and [architecture rationale](agent_system_selection.md). Before final submission, run the authoritative compiler, evaluation with matching data, Docker parity checks, and a full inference-time check. No new GPU benchmark has been run in this Windows workspace.
