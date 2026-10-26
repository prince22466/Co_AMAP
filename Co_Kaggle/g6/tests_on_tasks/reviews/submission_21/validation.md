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
| S18 test runner/environment edits | Case-insensitive protected-path audit | **No**, native tools can bypass |
| S19 repeated failed edit attempts | Stop after two errors, reread and change strategy | **No**, prompt-only |
| S19 unrelated tests and zero-exit FAILED | Same focused baseline/final command, check output | **No**, prompt-only |
| S20 premature empty submission | Budget reserve and submit-last instruction | **No**, prompt-only |
| Missing inline_snapshot/httpbin | Preflight baseline collection gate before real E4B evaluation | Yes, in standalone pilot, not agent |
| Scripted tests mistaken for real progress | Five-task pilot with real evaluator and nonempty-patch/trace/solved gates | Yes, in standalone pilot only |

No claim of an unbypassable edit gate or solved-task improvement is made.

## Acceptance

1. CPU tests exercise native tool selection and fault-injected audit behavior.
2. Optional pinned runtime test compiles and exercises the real packaged skill with
   a scripted model. It is **not** E4B task inference.
3. CI builds and uploads exact ZIP and manifest; it does not run E4B.
4. Real-model pilot on 5 fixed task IDs must show all five nonempty patches,
   at least one resolved task, no reported agent/tool-dispatch failure,
   and inspectable traces. Baselines with missing dependencies block pilot inference.
5. Inspect meaningful behavioral checks and final patches manually before 15-task promotion.
