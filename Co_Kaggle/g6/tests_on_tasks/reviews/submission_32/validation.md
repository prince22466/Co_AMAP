# Submission 32 validation and review ledger

## Evidence checked before PR opening

- Main config: a single LlmAgent, `gemma-4-e4b-it-qat-w4a16-ct`,
  `include_contents: default`, standard S30 generation settings.
- Declared native tools: exactly `run_command`, `read_file`,
  `edit_file`, `get_status`, `submit_patch`.
- **Not exposed:** `write_file`, code-graph tools, `run_skill_script`,
  custom wrappers, skills, adapters, LoRA or subagents.
- Evaluation limits remain identical to S30: 40 charged tool calls,
  48 turns, 4-minute session, 30-second command setting.
- Both ZIP variants were constructed from their current GitHub branch source;
  each archive has only three root members, with the same agent.yaml and
  eval_config.yaml; `instruction.md` is the single treatment difference.
- The permitted built-in `edit_file` already uses three-tier exact,
  whitespace-flexible and regex fallback matching (see harness guide);
  S32 therefore restricts positional recovery to one-line cases after
  two unsuccessful native edit attempts, not a second unlimited patcher.
- Representative guarded-template commands were executed in a local CPU
  Python/bash sandbox: exact-line substitution, CRLF, no final newline,
  wrong original line, invalid syntax, newline injection and symlink
  rejection behaved as expected. These were representative execution
  checks; the checked-in `test_submission32.py` exercises the actual
  prompt-extracted snippet when run in the repo.
- Static connector-side configuration assertions completed; code files
  and tool restrictions were inspected.

## Required tests still pending in the competition environment

```bash
cd Co_Kaggle/g6/tests_on_tasks/tools
python build_submission32.py
python -m unittest -v test_submission32.py test_submission32_gate.py
python smoke_submission32.py
```

The full Python unit suite and real `adk-submission` compiler smoke are
**not claimed to have run** here. Official Kaggle wheels, task snapshots
and a served E4B model are not available in the current container.

## Release gate (fail closed)

1. Run both variants on identical pilot task IDs, same served E4B and
   time-limit protocol, minimum two repeats each.
2. If pilot is promising, run both full 15-task S1 experiments, minimum two
   repeats each; avoid comparing different session limits.
3. Each S1 repeat needs >=5 actual official resolved, all S29 successes,
   >=1 of S30's two new successes, <=1 empty patch, no unexpectedly
   oversized patch or edit/repetition loop.
4. Any shell-source modification requires reviewing *full actual ATIF tool
   arguments*, source diff, and behavioral tests; visible
   `S32_GUARDED_LINE_EDIT_OK` is not proof of compliance. Reject fallback
   before 2 observed native failures or repeated fallback.
5. Review S2 and hidden-like tasks after passing S1; existing-file-only
   restriction may prevent legitimate new-file repairs.

## Design limitations

This is NOT a runtime instrumented/atomic tool-boundary interceptor:
`run_command` is generic shell, and an LLM can call a different command.
The only hard protection is removing `write_file`; guard conditions are
enforced only if the model executes the provided Python template unchanged.
The fallback is deliberately scoped to existing Python files in the
workspace. The model may never invoke it, invoke it incorrectly, or spend
more calls on it than it saves. This is why a native-only control is packaged.

Success of syntax preservation alone does not establish semantic correctness.
The original `rich_4075` task-provenance mismatch in earlier diagnostics
also means no task-specific hard-coded heuristics should enter the agent.
