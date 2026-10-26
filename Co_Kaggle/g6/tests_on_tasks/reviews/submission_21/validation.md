# Submission 21: recovery scope and acceptance gates

## Real E4B evidence motivating the change

The first-15 diagnostics show S19: 0 solved, 12 nonempty patches.
S20: 0 solved, 0 nonempty patches, 153 run_skill_script attempts,
37 missing required-wrapper-argument errors, 57 missing skill/script errors
and zero candidate check/publish operations (console-trace counts).
S20's gate was implemented but not reached by E4B.

## Regression mapping

| Previous issue | S21 mechanism | Enforced? |
|---|---|---|
| S17/S20 mandatory skill edit path | Restore S19 native edit_file, write_file, run_command | Yes, tool availability is configured |
| S18 syntax errors | S19 AST audit and prompt repair instruction | **No**, audit is optional |
| S18 accidentally deleted class | Compare AST top-level names | **No**, audit is advisory |
| S18 test runner/environment edits | Case-insensitive protected-path audit; harness scratch excluded transparently | **No**, native tools can bypass |
| S19 repeated failed edit attempts | Stop after two errors, reread and change strategy | **No**, prompt-only |
| S19 unrelated tests and zero-exit FAILED | Same focused baseline/final command, check output | **No**, prompt-only |
| S20 premature empty submission | Budget reserve and submit-last instruction | **No**, prompt-only |
| Missing inline_snapshot/httpbin | Preflight baseline collection gate before real E4B evaluation | Yes, in standalone pilot, not agent |
| Scripted tests mistaken for real progress | Real-E4B synthetic native-tool smoke before fixed five-task pilot | Yes, in standalone pilot only |
| Plain-text missing tool errors | Replay S20 error strings in pilot policy tests | Yes, in standalone pilot only |
| Four unchecked patches promoted after one resolved task | Require successful post-edit targeted command and submission evidence per task plus manual review | Yes, in pilot promotion only; semantic relevance needs review |

No claim of an unbypassable edit gate or solved-task improvement is made.

## Acceptance

1. CPU tests exercise native tool selection and fault-injected audit behavior.
2. The pinned official-runtime test can compile and run the packaged skill
   using a scripted model where the competition dependencies are available.
   No official-runtime execution is claimed merely from generic CI.
3. CI builds and uploads exact ZIP and manifest; it does not run E4B.
4. The pilot requires dependency-healthy baselines, then the pinned official
   runtime scripted smoke (no E4B inference), then real E4B synthetic
   native-edit/behavior/submit smoke BEFORE five task inferences.
5. Five nonempty patches, five post-edit focused checks followed by native
   submission, at least one resolved task, no recorded dispatch error, and traces
   are necessary but NOT sufficient for promotion.
6. A separate all-five manual-review JSON is required; the initial pilot
   always reports `ready_for_15_tasks: false`.
7. ADK-generated root-level `.adk_exec_*` files are deliberately excluded
   from audit and reported as ignored scratch; the audit does NOT protect them.
