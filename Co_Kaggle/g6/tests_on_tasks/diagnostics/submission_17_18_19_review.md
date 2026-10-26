# Submission 17–19: what failed and what remains

Based on the real task traces and returned patches in [diagnostics](../diagnostics). Comparisons below use the same first 15 tasks. These runs used Gemma E4B; scripted development tests are not evidence of real-model success.

## Results

| Measure | Submission 17 | Submission 18 | Submission 19 |
|---|---:|---:|---:|
| Tasks solved | 0/15 | 0/15 | 0/15 |
| Nonempty returned patches | 0/15 | 14/15 | 12/15 |
| Tasks with agent execution errors | 10/15 | 0/15 | 3/15 |

A returned patch means code was produced, not that the task was solved.

## What improved, and what did not

| Problem | What changed | Current status |
|---|---|---|
| **Cannot reliably reach editing tools** | S17 repeatedly supplied incorrect skill arguments and script paths. S18 exposed native editing tools directly; S19 kept them. | **Major improvement.** S17's widespread skill-dispatch failure did not recur, but S19 still omitted required native-tool arguments nine times. |
| **Runs end without useful patches** | S18 removed the separate coder/verifier/finalizer structure and let one agent edit and submit. | **Improved, not solved.** S19 again had three execution failures, including budget exhaustion and missing submission. |
| **Broken Python or accidental deletion** | S19 added an audit for syntax, removed definitions, and protected paths. | **Detection works when used; enforcement is missing.** Only 7/15 tasks invoked it. An unaudited S19 patch still had an `IndentationError`. |
| **Wrong or superficial implementation** | S19 strengthened task instructions. | **Unresolved.** A documentation task still received only an import comment; an SSE task received an empty class with the wrong name. |
| **Weak verification** | S19 asked for discovered tests, local imports, and behavioral assertions. | **Unresolved.** It still invented test paths, ran unrelated passing tests, and submitted after a script printed `FAILED` but exited zero. |
| **Repeated failed actions** | S19 instructed the agent to stop guessing after two edit failures. | **Unresolved.** It made 45 failed edit calls and repeatedly reread the same code. |

Some specific S18 mistakes did not recur: S19 preserved `ServerSentEvent` in `fastapi_15588` and avoided the earlier syntax error in `requests_7433`. Neither task was resolved. This is partial improvement, not proof of a general fix.

## Agent-system architecture problems

- **Instructions are being used as controls.** “Audit after editing,” “stop retrying,” and “verify before submitting” are requests to the model. The system does not reliably enforce those transitions.
- **The audit is optional and advisory.** The agent can skip it or submit after a blocker. A clean syntax audit also cannot establish correct behavior.
- **Verification is not tied to the requested change.** A passing command may test unrelated code; exit code zero can hide a printed failure. Verification needs to cover the task and the final patch.
- **Recovery does not reliably change strategy.** Repeated reads and failed edits consume the budget without triggering a deterministic fallback or reserving enough capacity to finish.
- **Development tests overstate readiness if interpreted broadly.** Scripted tests showed that tools could detect selected faults. They did not show that E4B would choose the right tools, interpret results, or repair real tasks.

S17 also showed that adding more agent stages and skill wrappers can make execution less reliable. More stages alone do not create an effective repair loop.

## Likely model-capability problems

- **Task understanding:** choosing the wrong API, file, or implementation; replacing a real feature with a placeholder.
- **Code reasoning:** missing integration points, changing behavior incorrectly, and struggling with exact edits and escaping.
- **Tool use:** omitting mandatory arguments and confusing repository files with skill scripts.
- **Use of feedback:** repeating failed actions and treating weak checks as sufficient evidence.

These are observed behaviors of E4B **in this setup**, not proof that model size alone is responsible. Prompts, tool interfaces, context handling, and orchestration also affect them. A controlled comparison is needed to separate model limitations from system design.

## What remains to fix first

1. **Make the execution path reliable:** bounded retries, a concrete recovery strategy, and validation before submission using mechanisms the competition runtime actually supports. Do not describe prompt-only rules as enforced.
2. **Verify the actual task:** use a focused check that exposes the original problem and passes after the change; confirm imports use the checkout and rerun after the final edit.
3. **Separate environment failures from patch failures:** missing `inline_snapshot` and broken `httpbin` fixtures still affect evaluation. Establish a working baseline without changing tests to manufacture a pass.
4. **Require real-model evidence:** run a small representative E4B set before spending GPU time on a larger evaluation. Measure correct solutions and successful recovery, not just patch production.

**Conclusion:** S18 largely fixed S17's inability to execute edits. S19 added useful fault detection, but did not establish a reliable repair-and-verification loop. Correct task implementation remains the central unsolved problem.
