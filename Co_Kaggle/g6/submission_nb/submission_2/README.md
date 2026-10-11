# Submission 2: prompt revision

This candidate retains Submission 1's two configured roles: the main coder and optional read-only analyzer. Both use the unchanged competition Gemma model, with no adapter. Only `prompts/system.md` and `sub_agents/analyzer_resources/instruction.md` differ among the packaged files. All YAML configurations, tool permissions, context settings, sampling values, and enforced budgets are byte-identical to Submission 1.

## Why revise the prompts

The user reported two Kaggle scores for Submission 1: **0.10 and 0.06**. No task-level results, logs, or traces were available. These observations establish a baseline to compare; they do not identify why tasks failed or demonstrate that an individual prompt instruction is responsible.

The revision targets several plausible failure modes:

| Change | Intended effect | Tradeoff to inspect |
|---|---|---|
| Continue using tools until implementation is complete and the patch is captured | Avoid ending with a plan, diagnosis, or premature final response | Does the agent actually produce and submit a patch? |
| Cover every requested behavior and relevant caller/example variant | Avoid a narrow fix that passes one test but misses other requirements | More source changes may need more verification. |
| Edit once source supports the cause and fix; pre-edit reproduction is optional | Avoid repeatedly confirming the same hypothesis | Early edits still need source evidence and relevant tests. |
| Usually one or two analyzer calls, a four-call upper behavioral target, and a 180-word answer | Reduce exploration and handoff overhead | Complex findings can lose detail; targets are not enforced quotas. |
| Recheck uncertain assumptions rather than repeating the helper's exploration | Reduce duplicate reads | Graph findings still need current-source support. |
| Use short targeted behavioral checks and shell timeouts when appropriate | Leave time for repair and submission | A slow valid test may need longer; a timeout is not a pass. |
| Diagnose failure output instead of assuming environment problems are unrelated | Avoid dismissing failures introduced by the patch | Avoid spending the remaining task fixing unrelated dependencies. |
| Combine bounded final review where practical | Reduce extra tool/model interactions | Review new files and truncated output deliberately. |

Word counts are recorded in the [experiment record](../prompt_experiments.md). They are not tokenizer measurements. The intended benefit is changed behavior guidance, not an established performance gain.

## Fixed configuration

- Main model: `gemma-4-31b-it-qat-w4a16-ct`; same model for the analyzer.
- Main tools: nine harness built-ins plus the optional `code_analyzer_agent` tool.
- Analyzer tools: `read_file`, `search_similar_code`, `get_code_neighbors`, `get_code_subgraph`.
- Task budget: four minutes, 40 counted calls, 80 turns, 60-second command cap.
- Main sampling: temperature 0.2, top-p 0.95, 16,384 output ceiling, 4,096 thinking budget.
- Analyzer sampling: temperature 0.2, top-p 0.95, 2,048 output ceiling, 1,024 thinking budget.
- Analyzer history: `include_contents: none`; coder sends a self-contained focused request.

## Package and compare

From the project root:

```bat
python submission_nb\tools\submission_tools.py validate --submission-dir submission_nb\submission_2
python submission_nb\tools\submission_tools.py build --submission-dir submission_nb\submission_2 --output submission_nb\artifacts\submission_2.zip
python submission_nb\prepare_evaluation_notebook.py --submission-dir submission_nb\submission_2 --baseline-dir submission_nb\submission_1 --output submission_nb\evaluate_submission_2.ipynb
```

The generated companion embeds both candidates and compares them on the same public tasks. Its default variants are `baseline` (Submission 1) and `candidate` (Submission 2); both retain the two-agent configuration. It checks that resolved configuration excluding instruction text is identical. Use fixed task IDs, expand beyond the two-task smoke default, and repeat/reverse execution order before judging latency differences. The selected release is `candidate`, pending your assessment of the reports.

Reports retain resolution, patches, verification outcomes, stop evidence, tool/delegation counts, and available timing/token usage. Missing measurements remain unknown. See [the development workflow](../README.md) and [experiment record](../prompt_experiments.md).

This is an unmeasured candidate prepared for evaluation. Local format, packaging, and notebook checks do not establish better coding-task performance. Runtime/compiler validation and GPU evaluation still need the competition environment.
