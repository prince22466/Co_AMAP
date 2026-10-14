# Agent development workflow

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
| Reviewer overhead or unhelpful findings | Compare `optional_reviewer` against `optional_analyzer`; inspect findings and repairs rather than inferring benefit from call counts. |
| Locator/verifier overhead or inconclusive checks | Use Submission 4's four helper combinations; inspect actual shell commands, test outcomes and subsequent coder repairs. |
| Long reasoning or grading timeout | Use Submission 5's isolated runtime controls; inspect effective thinking settings and the shared command/grading timeout. |
| Missing trace | Preserve logs and result fields; enable the runtime's trace export before claiming detailed stage timings. |

Resolution rate uses the reported verification outcome. Missing outcomes remain unknown. Task duration can include setup and verification, whereas the agent timer excludes setup. Model-request counts do not establish the harness's enforced turn count. Aggregate run tokens do not measure context size; context pressure needs per-request usage. `get_status` and `submit_patch` are exempt from counted tool calls, while helper delegation is shown separately when identifiable.

Some metrics depend on the runtime's trace schema: requests, tool frequencies, successful edit timing, and per-request tokens. Missing measurements remain null with warnings. Failure categories and suggested experiments are diagnostics based on recorded evidence, not causal proof.

## Promote and extend

Start with the unchanged model, compare the two structures, and keep the variant that solves more tasks within the total inference allowance. Inspect per-repository results and the failure reports, then change one component and repeat the same tasks. Use adapter training only after collecting a stable baseline and evidence of the behavior to improve.

See [submission configuration and extension points](submission_1/README.md) and [architecture rationale](agent_system_selection.md). Before final submission, run the authoritative compiler, evaluation with matching data, Docker parity checks, and a full inference-time check. No new GPU benchmark has been run in this Windows workspace.
