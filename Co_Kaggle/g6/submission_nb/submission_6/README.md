# Submission 6: Submission 2 with Gemma 4 E2B

Prepared on 2026-10-14 from Submission 2. The [organizer update supplied locally](../../competition_material/Update_new_models.md) permits four model variants and requires every agent/subagent within a submission to use the same variant. The smallest newly allowed variant is `gemma-4-e2b-it-qat-w4a16-ct`.

## Exact package changes

Only the `model:` value changes in two active files:

- `agent.yaml`: main coder changes from 31B to E2B.
- `sub_agents/code_analyzer.yaml`: optional read-only analyzer changes from 31B to E2B.

Every other packaged file is byte-identical to Submission 2. Both instruction files, tool access, helper history/output settings, sampling and task budgets are preserved. No adapters are included. The local validation/build tools now permit the four organizer-listed variants while rejecting mixed variants, including unreferenced standalone agents.

## Retained configuration

| Component | Setting |
| --- | --- |
| Agents | Main coder and optional read-only analyzer; both E2B. |
| Main tools | Nine harness tools plus the optional analyzer. |
| Analyzer tools | `read_file` and three graph/retrieval tools. |
| Task allowance | Four minutes, 40 counted calls, 80 turns. |
| Command/grading timeout setting | 60 seconds, preserved from Submission 2. |
| Coder generation | Temperature 0.2, top-p 0.95, output ceiling 16,384; thinking budget 4,096 and include_thoughts true. |
| Analyzer generation | Temperature 0.2, top-p 0.95, output ceiling 2,048; thinking budget 1,024 and include_thoughts true. |

Submission 5's timeout and thinking changes are not carried into this candidate. The earlier shared grading-timeout risk and uncertain numeric thinking-budget enforcement remain as documented in [the investigation](../zero_score_investigation.md). This release isolates the model substitution; it does not claim to resolve those runtime questions.

## Build and evaluate

From the project root:

```bat
python -B submission_nb\tools\submission_tools.py validate --submission-dir submission_nb\submission_6
python -B submission_nb\tools\submission_tools.py build --submission-dir submission_nb\submission_6 --output submission_nb\artifacts\submission_6.zip
python -B submission_nb\prepare_evaluation_notebook.py --submission-dir submission_nb\submission_6 --output submission_nb\evaluate_submission_6.ipynb
```

The companion notebook embeds the actual E2B candidate, sets the model input to `/kaggle/input/models/google/gemma-4/other/gemma-4-e2b-it-qat-w4a16-ct/2`, and takes its installation cell from the supplied updated starter. Attach the E2B model input and updated competition wheelhouse. Confirm the mounted model path/version in the configuration cell before running.

It compares `coder_only` and `optional_analyzer`, both E2B, with identical prompts/sampling/budgets. The selected release is `optional_analyzer`, matching Submission 6's two-agent structure. Every benchmark variant must use the same model served in that notebook run. Startup discovers one declared variant and registers only that alias, with new-model startup options from the updated starter.

To compare Submission 2's 31B against Submission 6's E2B, run separate model sessions on identical explicit `TASK_IDS`, preserve their manifests and compare outcomes and runtime. Do not use the prompt-comparison baseline flag to combine different models under one model server. The default two tasks are a smoke check, not a representative performance measurement.

The local ZIP has seven active files with `agent.yaml` at its root. Reports, notebooks, documentation and the release manifest stay outside the ZIP. After notebook evaluation, the selected release is packaged to `/kaggle/working/submission.zip`.

## Evidence status

Submission 2 previously scored 0.10. Submission 3, Submission 4 and Submission 5 are recorded at 0.0. No task-level scoring artifacts identify the cause. Smaller-model speed and memory savings are organizer-described opportunities; actual task resolution and efficiency for this unchanged prompt/configuration require measurement.

Submission 6 is unmeasured. Local checks cover all four allowed models, rejection of mixed variants, exact preservation of Submission 2 outside the two model declarations, notebook model selection/routing and package consistency. Runtime-routing checks use stubs and do not load Gemma weights. The official updated compiler/runtime, effective thinking behavior, GPU inference, grading and total time must be checked in Kaggle.
