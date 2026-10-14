# Zero scores: possible causes and permitted diagnostics

Investigation date: 2026-10-13. The user reports Submission 3 and Submission 4 at 0.0, with no additional Kaggle log information. Submission 1 scored 0.06 and 0.10; Submission 2 scored 0.10. A successful scoring job with a zero score does not establish that individual tasks executed correctly. The cause is not identified without task-level results.

## What was checked locally

All four archives pass ZIP CRC checks, contain exactly the locally resolved dependency files, and match their preserved source after newline normalization. All agents declare the permitted model. Referenced files are present, and prompts contain no brace-delimited ADK state placeholders. There is no observed archive-layout, stale-source, missing-include or prompt-interpolation defect.

| Candidate | Configured agents | Coder instruction words | Public scores reported |
| --- | ---: | ---: | --- |
| Submission 1 | 2 | 768 | 0.06, 0.10 |
| Submission 2 | 2 | 653 | 0.10 |
| Submission 3 | 3 | 928 | 0.0 |
| Submission 4 | 3 | 526 | 0.0 |

All four declare four minutes, 40 counted tool calls, 80 turns and `timeout_seconds: 60`. Their coder sampling is temperature 0.2, top-p 0.95, max output 16,384 and thinking configuration 4,096/include_thoughts true. Word counts are not token measurements. Submission 4's shorter prompt also scored zero, so prompt length alone is an insufficient explanation.

Local archive identity:

| ZIP | SHA256 |
| --- | --- |
| `submission_1.zip` | `242c31b9c1c87f51ba124bd129ed05a2a7002b0f93808ade92e9a930c28fe4aa` |
| `submission_2.zip` | `1b392087e7923e464498c3d8a8e1c8452f6f7b5c1c62e05823a1cb0b7077a67c` |
| `submission_3.zip` | `76715657bf3ed383b6168842b53b4f9613f8eee15afdd6ecbef601ad807217c6` |
| `submission_4.zip` | `4ccf10158c57673700155fec5428bb77fde4cf426e94707747366c201386c46b` |

The exact scoring runtime and uploaded executions are unavailable. The local environment lacks `swegemma`, `adk_submission`, ADK and real-model inference setup. The checks above are not official compiler or execution tests.

## Evidence-backed configuration risks

The sources below are copies of organizer harness code published in another participant's repository. They provide concrete implementation evidence for that snapshot, not proof of the runtime used in the user's scoring jobs. Confirm the installed wheel versions and source in Kaggle before relying on the details.

### Command timeout also constrains grading

The inspected `EvalConfig` maps `timeout_seconds` into `harness.command_timeout_seconds`. The final verification function passes that same value to the grading pytest command. Our 60-second setting can therefore cut off a correct patch's grading run. Earlier notes describing it only as an agent command cap omitted this shared use. Submission 2 also used 60 seconds and scored above zero, so this is a real risk rather than an established explanation of the regression.

Sources: [config mapping](https://github.com/happyc0der/gemma-swe-agent/blob/main/harness/official/swegemma/config.py), [verification implementation, pytest timeout](https://github.com/happyc0der/gemma-swe-agent/blob/main/harness/official/swegemma/harness/verification.py).

An isolated test would compare an unchanged candidate at timeout 60 versus 300, preserving the four-minute agent allowance, tasks, sampling and tools. Keep the agent's own quick checks bounded independently, for example with shell `timeout 20s`. Increasing the command timeout alone does not increase the configured agent session allowance, but it permits longer individual commands and longer grading. Record actual full runtime.

### Numeric thinking budgets may not cap reasoning

The inspected compiler bridge reads `thinking_budget` to decide whether thinking is enabled. It forwards an `enable_thinking` toggle, and optionally a reasoning level, rather than the numeric budget. Neither 4,096 for the coder nor 1,024 for helpers is thereby guaranteed to be a hard reasoning cap. Additional helper inference may consume the shared four-minute allowance before useful edits or patch capture.

Source: [generation bridge](https://github.com/happyc0der/gemma-swe-agent/blob/main/harness/official/adk_submission/resolvers/generation.py).

Check actual outbound generation settings and usage in the user's runtime. Test disabled thinking or shorter generation ceilings separately from architecture and timeout changes. Avoid treating a prompt instruction or accepted YAML value as enforcement. A shorter ceiling can truncate a tool call, so inspect finish reasons and complete payloads.

## Other hypotheses, with the evidence needed

| Hypothesis | What would support it | Next comparison |
| --- | --- | --- |
| Delegation consumes implementation time | Helper requests dominate time/tokens, first edit is late, or the task stops with an empty patch | Same Submission 4 prompt with neither helper, each helper individually, and both. |
| Invalid or incomplete calls | Tool argument errors, truncation or repeated failed edits/searches | Inspect raw events and finish reasons; simplify the demonstrated failing action. |
| Context pressure or repeated exploration | Per-request inputs approach the context ceiling; reads repeat; explicit context errors | Compare focused outputs and available compaction evidence. |
| Patch quality or incomplete requirements | Nonempty, applicable patches fail required assertions/regressions | Inspect exact patches and official verification output. |
| Runtime/compiler/setup problem hidden in task results | Errors before the first useful tool call, missing patch artifacts, setup or compiler exceptions | Compile with the installed competition wheels and inspect raw task errors. |
| Sandbox/environment differs from expectation | Missing commands, wrong source location, test collection errors | Check a small smoke session in the scoring-like environment. |
| Evaluation changed or scores vary | Preserved Submission 2 no longer reproduces its earlier nonzero score | Rerun the identical Submission 2 ZIP as a control. |

Graph tools were retained in Submission 3 and removed in Submission 4. Their removal cannot explain both zeros on its own. Adding a second helper connection is a shared change, but availability does not establish invocation or overhead. Do not infer that three-agent systems cannot work from these scores.

## Which logs a submission can influence

There are distinct surfaces:

| Surface | Controlled from the submitted agent? | Meaning |
| --- | --- | --- |
| Text returned by `run_command` or a permitted skill script | Yes, when the agent invokes it | A diagnostic marker becomes tool output visible to the agent. |
| Harness transcript/trace files | Indirectly | Invoked commands and results can be recorded by the harness. |
| Kaggle's top-level scoring log | No guaranteed submission setting identified | Host display, verbosity and retained artifacts determine visibility. |
| Provider request usage and precise timing | Not reliably through a sandbox script | Capture these through the evaluator's events/tracing layer. |

The inspected runner creates per-task transcript files and records events. In quiet mode its display routes to the file console. Sandbox stdout therefore should not be equated with Kaggle host stdout. A model's final response may also be recorded privately rather than shown in the visible scoring log.

Source: [agent runner, transcript/display setup](https://github.com/happyc0der/gemma-swe-agent/blob/main/harness/official/swegemma/harness/agent_runner.py).

The official competition documentation permits configured harness tools and sandboxed skills. Its compiler resolves callbacks through organizer-controlled registries; it does not import arbitrary submission Python as host code. No supported submission field for forcing the host logger to debug was identified. An arbitrary `agent.py`, logging configuration file or unregistered callback cannot be assumed to execute. [Official tool/skill rules](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview), [compiler restrictions](https://github.com/happyc0der/gemma-swe-agent/blob/main/harness/official/adk_submission/compiler.py), [restricted agent schema](https://github.com/happyc0der/gemma-swe-agent/blob/main/harness/official/adk_submission/schema.py).

## Small compliant checkpoint proposal

Start with existing `run_command` tools; adding a skill adds another variable. Request short diagnostic markers in the next necessary shell command, covering:

- Coder start.
- Entry/return for each helper, with the actual helper's name.
- First source edit confirmed by Git state.
- Focused verification start and actual outcome.
- Pre-submission patch state and remaining budget via `get_status`.

For example, prefix an existing targeted check with:

```bash
printf '%s\n' 'G4DIAG agent=swe_coder_agent stage=verify'
timeout 20s python3 -m pytest -q tests/ACTUAL_EXISTING_TEST.py
```

`ACTUAL_EXISTING_TEST.py` is a placeholder in this proposal, not a runnable task target. The agent must select an actual relevant test. The final command's exit code remains the test/timeout outcome; do not append a successful print that masks a failed test.

A separate checkpoint can print a marker and bounded `git status --short`/`git diff --stat`; those report workspace evidence, not verification success. Existing `get_status` provides measured remaining time and calls. Label self-selected stages as agent-reported. Exact token counts must come from provider usage, not model self-report or character counts.

Use the existing sandbox interfaces and ordinary repository inspection. A permitted skill script may print concise diagnostics or keep scratch data under `/tmp`, but such files may disappear when the task sandbox is torn down. Keep diagnostics out of implementation patches and protected test/runner files. Do not read private grading data, access host logs through sandbox escape paths, alter grading behavior or send data outside the offline environment.

Marker instructions depend on model compliance. They cannot diagnose a compiler failure before the model starts, capture every request automatically, force private files to become participant-visible, or guarantee data survives an aborted job. Plain-text markers belong in tool output; they should not cause the agent to emit a final completion response before patch capture. Avoid introducing literal brace-delimited examples into ADK instruction text unless their templating behavior is handled correctly.

No diagnostic ZIP was generated or historical submission archive modified in this investigation. The marker proposal needs a development smoke test to confirm which output surface is visible before spending another scored submission.

## Recommended next work

1. In a user-controlled Kaggle evaluation notebook, use the installed official compiler and preserve its errors. Record runtime versions and effective model settings.
2. Run a small fixed task set with the actual Submission 2 and Submission 4 configurations. Preserve raw task results, nonempty patches, transcripts, traces and grading output. Print a compact task summary in that notebook, where logging is under the user's control.
3. Inspect whether failures are before editing, during capture, or after applying a nonempty patch. Confirm helpers appear in traces when invoked.
4. Compare the unchanged candidate at command/grading timeout 60 versus 300. Change thinking behavior in a separate comparison.
5. Use Submission 4's four architecture controls under the same prompt and total budget. Repeat promising results on held-out tasks.
6. Add checkpoint markers only after confirming their transcript/log visibility. If scored-job artifacts remain private, request an organizer-supported log/export mechanism through the competition's normal channels rather than trying to force access from the sandbox.

The existing evaluation notebooks already retain many artifacts, but they do not guarantee complete per-agent usage/timing attribution or expose scored-job private outputs. A diagnostic pass should establish that coverage before a long benchmark or another architecture revision.
