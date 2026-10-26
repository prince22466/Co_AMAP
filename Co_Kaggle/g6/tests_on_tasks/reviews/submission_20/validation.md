# Submission 20 validation scope

## Source evidence

The request targets the four priorities in
[the submission 17–19 review](../../diagnostics/submission_17_18_19_review.md).
`failures.json` contains six exact S19 tool-call excerpts, with archive paths and hashes.
Tests verify their provenance. The S18 syntax regression reuses the exact archived edit
from `../submission_19/failures.json` on a minimal valid enclosing definition.

## Regression mapping

| Observed failure | New verification |
|---|---|
| S17 skill wrapper error | Official ADK test injects missing script path, then corrects it and completes; no claim that real E4B will recover |
| S17 edit rejected for missing assertion/module | Candidate edit works without a behavioral probe or module; those are required only for publication |
| S17 finalizer starvation | Compiled agent has no child agents/finalizer; native submit remains in the same agent |
| S18 broken syntax/class deletion | Mutation rejects and restores prior candidate bytes; exact archived syntax-breaking replacement is replayed |
| S19 optional/skipped audit | Live diff remains empty before successful publication, including an early native submission |
| S19 ignored blocker | Failed publication cannot expose candidate to native submission |
| S19 stale or irrelevant passing test | Publication reruns current checks; baseline-pass probes and changed-code nonexecution are rejected |
| S19 printed FAILED with exit 0 | Probe output failure marker blocks publication |
| S19 repeated bad edits | Third identical failed edit is blocked; total edit attempts bounded; check/publish remain accessible |
| Missing inline_snapshot / recursive httpbin | Pilot baseline classification blocks model spending; no test rewrites or denominator filtering |
| Synthetic tests mistaken for real success | Pilot rejects scripted smoke as promotion evidence; all five actual tasks must resolve |

## What has not been demonstrated

No real E4B inference or full Kaggle evaluation was available in the authoring environment.
The model may still miscall the skill, supply a weak probe, implement the wrong API, run out
of turns, or omit submission. Candidate publication enforces its mechanical evidence
requirements; it does not prove complete task correctness or malicious-code containment.

The pilot runner's baseline classification/promotion policies are CPU-tested. Its full
five-task GPU flow is supplied for the user's official environment, not reported as run.

## Completed validation (2026-10-26)

- 26 CPU gate/package/provenance regressions: passed.
- 6 baseline/pilot promotion policy tests: passed.
- 3 pinned official ADK runtime tests: passed, including packaged skill execution,
  malformed-wrapper recovery, official protected-path coverage, early empty submission, rollback, rejected tautology,
  successful checked publication, and an independent final behavior oracle.
- These are development tests with synthetic fixtures/scripted model responses.
  Real-model smoke, the five-task pilot, and Kaggle score remain unmeasured.

## Follow-up review fixes

- Detect printed failure markers before output truncation, so noisy output cannot hide
  a previous FAILED message. A regression reproduces the original false verification.
- Compare the published live paths and content hash with the verified candidate;
  Git-ignored new targets now trigger rollback instead of a false published result.
- Cover the official runtime's static protected paths, including `.pytest.ini`,
  `_swegemma_stubs.py`, and uppercase test directories. Runtime tests compare the
  policies. Additional task-specific protected paths remain owned by the harness.
