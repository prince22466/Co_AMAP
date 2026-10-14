# Finding suitable Gemma 4 architectures and prompts

Research date: 2026-10-11. Target: the competition's unchanged `gemma-4-31b-it-qat-w4a16-ct` model and its scoring harness. This is a proposed experiment plan, not a measured architecture or prompt ranking. No submission configurations or existing notebooks are changed by this document.

## Evidence and limits

Google documents Gemma 4's native system instructions, function-call format and thinking controls. Its guidance distinguishes ordinary conversation turns from an ongoing tool-call turn when retaining thought history. It also describes instructions for reducing reasoning length, without identifying a universally optimal prompt. These are integration requirements and useful experimental leads; they do not establish a winning software-engineering architecture. [Google prompt-formatting guidance](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4).

Google's function-calling examples pass tool schemas through the chat template and use the application's tool execution loop. Our prompts should use the competition tools and completion protocol. The runtime should handle Gemma's special tokens rather than copying raw token syntax into agent instructions. [Google function-calling guide](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4).

The vLLM recipe documents Gemma-specific tool and reasoning parsers. The local notebook generator already requests both parsers, automatic tool choice and a 32,768-token context. That configuration still needs runtime confirmation in Kaggle; current generic recipe defaults are not necessarily the scoring stack's pinned defaults. [vLLM Gemma recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html).

Google's general sampling recommendation is temperature 1.0, top-p 0.95 and top-k 64. Our current coder and helpers configure temperature 0.2 and top-p 0.95, with no explicit top-k. Treat the general recommendation as a separate sampling candidate, not proof that it improves the competition's quantized coding agent. [Google model card](https://ai.google.dev/gemma/docs/core/model_card_4).

Public competitor evidence is limited. [JAgent](https://github.com/rlawoals0529/gemma-4-developer-agent) reports a 0.05 public score for a single-agent baseline. [happyc0der's experiment log](https://github.com/happyc0der/gemma-swe-agent) reports configurations around 0.06-0.10, along with substantial differences between proxy and scoring environments. These are author-reported runs with different configurations, not paired evidence against our candidates. [Gosling-dude's project](https://github.com/Gosling-dude/gemma4-developer-agent) explicitly says its multi-specialist system had not been run with the real Gemma model at its documented status. Configuration complexity alone does not establish performance.

[mini-SWE-agent](https://mini-swe-agent.com/latest/) provides a useful minimal-loop reference. Its [default prompt](https://raw.githubusercontent.com/SWE-agent/mini-swe-agent/main/src/minisweagent/config/default.yaml) expects a bash-block protocol that differs from our harness. Adapt workflow ideas; preserve our native tool calls and `submit_patch()` protocol. Its general benchmark results do not establish a Gemma-specific ranking.

Our [submission records](../submission_records.md) report starter 0.03, Submission 1 0.06 and 0.10, and Submission 2 0.10. Submission 3 remains unmeasured. Aggregate scores do not identify the benefits of helper agents or individual prompt changes.

## Initial architecture comparison

Use one coder that owns all source edits, verification, repairs and patch submission. Compare helper availability with a shared coder prompt whose helper instructions are conditional on tool availability.

| Variant | Analyzer available | Reviewer available | Hypothesis to test |
| --- | --- | --- | --- |
| Coder only | No | No | Direct ownership and fewer handoffs improve completion under tight budgets. |
| Coder + analyzer | Yes | No | Focused source investigation improves localization enough to offset delegation overhead. |
| Coder + reviewer | No | Yes | An independent check of an existing patch catches incomplete or incorrect implementation. |
| Coder + analyzer + reviewer | Yes | Yes | Both roles help distinct failure modes without consuming too much implementation time. |

Helpers remain optional and read-only. Their inference and tool usage share the total task budget. The reviewer gets the issue, acceptance conditions, current diff, changed paths and actual test results. The analyzer gets one uncertainty plus source evidence already inspected. Both return concise evidence, recommendations and explicit uncertainty. No helper may terminate the coder's task or submit its patch.

The current Submission 3 notebook includes coder only, coder + analyzer and all three agents. Coder + reviewer is an additional proposed control, not currently implemented. This four-variant design isolates both helper effects and their interaction. Actual Submission 1/2/3 comparisons remain separate because their coder prompts differ.

A mandatory planner/coder/reviewer chain is a later candidate if traces show persistent planning errors or ineffective optional routing. Independent patch ensembles and parallel writers are lower priority under the current time envelope. These priorities are engineering hypotheses, not claims about inherent Gemma limitations.

## Prompt families

Use the same tool schemas, runtime, tasks, sampling, total budget and helper prompts when comparing coder prompt families. Change one behavior at a time when diagnosing an observed failure.

| Family | Proposed content | What would justify it |
| --- | --- | --- |
| Compact workflow | Objective, focused search, complete fix, targeted check, final diff, explicit submission | Fewer empty patches, repeated reads and premature completions. |
| Acceptance checklist | Extract requirements; associate each with relevant implementation and verification evidence; cover missing requirements before submission | Better resolution of tasks with multiple behaviors or exact compatibility/error contracts. |
| Efficient reasoning | Compact workflow plus instructions to stop reconsidering established evidence, choose the next decisive action, and reason more when a new contradiction appears | Less reasoning time/token usage without lower resolution. |

Example compact coder prompt seed, authored for this experiment:

> Implement the issue in /workspace and capture a complete implementation patch. Inspect the named source and relevant tests, then edit when the cause and fix are supported. Cover every requested behavior and preserve existing contracts. Run a focused check and use its actual result to repair the implementation when needed. Keep exploration and reasoning concise; re-read only to answer a new uncertainty. Use only available tools. Check remaining resources before expensive work, review the final changes, and call submit_patch last. Continue tool use while work remains; a proposed fix is not completion.

This seed needs the harness's necessary constraints on protected files, scratch files, supported tool arguments, new-file diff handling and helper routing. It is not a drop-in replacement for an existing submission.

Example acceptance-checklist addition:

> Before editing, identify the requested behaviors and relevant edge cases. Before submission, check each against the patch and observed verification. Preserve required signatures, exception types, exact messages and compatibility. Mark untested behavior as uncertain; do not infer full correctness from one passing command.

Example efficient-reasoning addition:

> Use the shortest reasoning needed to choose a justified next action. Do not repeatedly explain or recheck established findings. Move from source evidence to implementation once the fix is supported. Spend additional reasoning on new contradictions, ambiguous requirements or demonstrated test failures.

These are unmeasured candidate instructions. Do not reward reasoning brevity when it reduces correctness. Compare actual tokens and latency, not prompt word count alone. Plain instructions that repeat a token budget do not enforce it.

## Runtime probes before the benchmark

Use a separate smoke session to confirm tool calls are parsed and executed, numeric arguments keep their types, edit payloads are complete, and task submission produces a patch. Confirm the installed versions and effective chat template, thought-history behavior, compaction and caching match the intended scoring environment.

The YAML `thinking_budget` values are declared settings. Verify their effective translation and enforcement in the installed ADK/LiteLLM/vLLM path; neither successful YAML validation nor Google's ON/OFF thinking support proves a numeric generation cap is enforced. Likewise, verify actual top-k and sampling settings rather than assuming omitted values. Record output truncation, malformed calls, repeated actions and unsupported/dropped parameters.

Confirm that raw traces identify all invoked agents and contain request usage and start/end timing where available. A helper with no trace must remain unmeasured. Preserve raw evidence even when an exporter cannot support every requested breakdown.

## Staged evaluation

1. Instrument the common evaluator and reporting layer for task, request, agent, tool and stage measurements. Validate one smoke trace before long runs.
2. Preserve the historical candidates. Compare actual Submission 1, Submission 2 and Submission 3 on identical task IDs; separately compare architecture controls under one shared coder prompt.
3. Screen the four helper combinations on 12 fixed development tasks spanning repositories and issue types. At the current four-minute task cap, 48 task executions allow up to 3.2 hours of agent time, plus setup, verification and model startup. This is a planning calculation, not a measured runtime or run-limit guarantee.
4. Compare the three prompt families on the strongest one or two architectures. Expand finalists to approximately 24 development tasks and repeat promising comparisons; keep a separate validation set. A small development sample is a screening tool, not evidence of a universal winner.
5. Check a small crossing of the best prompt and architecture finalists to detect interactions; screening with one shared prompt can miss an architecture that needs different routing instructions.
6. Test sampling separately: current configuration versus the documented Google recommendation, if accepted by the competition runtime. Test thinking changes separately and measure their actual effect. Do not change prompts, sampling and architecture in the same diagnostic comparison.
7. Validate finalists on held-out tasks, preferably repeat them, and check scoring Docker parity and full inference time before promotion.

Classify task types from issue descriptions before examining results. Freeze task IDs, labels and configuration hashes. Include attempted-but-failed and unstarted tasks in coverage reporting. Gold patches and hidden verification content must not be supplied to the solving agents or used to select task-specific prompts. Reverse or rotate execution order when assessing timing; record cache state rather than treating warm and cold runs as equivalent.

## Measurements and decision criteria

Primary selection: tasks resolved within the same total inference allowance, with complete task coverage. Secondary comparisons: input/output/reasoning/cached tokens where supported, setup/agent/verification time, first successful edit, tool errors, context pressure, explicit submissions, empty patches and exhaustion causes.

Break usage down by coder/analyzer/reviewer and by localization/implementation/verification/review/repair/submission/other. Stage assignment may be inferred and must include provenance and unknown/mixed categories. Tokens embedded in edit tool calls are code output, not an additional disjoint category. Avoid double-counting nested delegation durations, repeated trace exports or provider token subcategories. Report trace coverage and whether totals reconcile; do not silently convert missing usage to zero.

For each task type and repository, show task counts, paired outcome changes, median and p90 usage/time where sample sizes support them, and solved/failed breakdowns. For helper impact, inspect findings and subsequent coder actions, including useful corrections, unsupported recommendations and time lost to delegation. Helper invocation alone does not establish benefit.

Retain configurations that solve more held-out tasks under the actual time envelope. When observed resolution is indistinguishable, prefer lower resource usage and simpler behavior, while acknowledging uncertainty from small samples. Task-type routing becomes a later candidate only when measured subgroup gains repeat on validation data and routing uses information available before solving.
