---
name: repo-navigation
description: Evidence-driven repository repair. Use for issue localization, incomplete API changes, misleading checks, repeated tool errors, and final patch inspection under a small tool budget.
---

# Repository repair with evidence

Use this skill once per issue. The coder still owns changes and submission; existing helpers remain optional. Do not add an extra planning or review agent.

## 1. Establish the contract and workspace

In the first repository command, obtain `pwd -P` and a bounded `git status --short`. Pass that observed absolute workspace path to scripts: the skill executor may change its working directory to a temporary skill directory. Do not search for the repository outside the assigned workspace.

Write at most four compact acceptance bullets from the issue and source. Separate stated requirements from assumptions. For a new option, identify its public entry, forwarding layers, consumer and default/override behavior before editing. For validation, identify the stage that must reject input: construction, assignment, serialization or execution. A correct error at the wrong stage can break valid use. Load `references/contracts.md` only when these choices matter.

## 2. Inspect the decisive source, then edit

Search concrete symbols in likely source and test paths together. For a Python file, use `scripts/inspect_source.py` with `path` and optionally `symbol` to obtain definition ranges and bounded source. Follow the returned `next_line` when truncated. Do not repeatedly search the same file for one function at a time.

After roughly eight counted inspection calls, choose a supported change or one remaining decisive question. This is a checkpoint, not a hard stop. Before ten calls remain, stop broad exploration and implement/verify the best-supported fix. Use `get_status`; no timer means unknown, not zero.

Use small native edits. After a rejected call, read its missing fields, shorten the payload and correct it once. After the second rejection on the same operation, use an assertion-guarded `run_command` replacement. Load `references/tool_recovery.md` for the template. Never repeat an identical rejected call or rewrite a whole module blindly.

## 3. Verify the public behavior from the edited source

Prefer one existing focused test or a short assertion-based reproduction. Derive expectations from the issue and existing contracts before seeing the patched output. Check one requested case and one neighboring valid case; vary empty/boundary inputs or default/explicit options when relevant.

After an implementation edit, prefer `scripts/repair_gate.py` with `module`, `code`, stable requirement IDs and observed budgets; it invokes the isolated checker. Use `scripts/check.py` directly only when the gate is unavailable. It puts workspace `src/` and the root first, verifies the requested module's origin, runs in a fresh process, bounds time/output and reports completion and executed assertion counts. A caught assertion failure still yields `failed`; an uncalled check yields `no_assertions_executed`. Do not catch assertion failures, merely print a failure, or treat `exit_zero_without_assertions` as behavioral evidence. Execute runtime paths after moving imports; parsing alone cannot find runtime NameError. Load `references/verification.md` when a check is misleading or blocked.

## 4. Audit and capture

Run `scripts/audit_patch.py` once instead of separate status, syntax and whitespace commands. It reports protected edits, untracked files, syntax errors and suspicious scratch paths without modifying anything. Read the actual bounded diff separately; the audit does not judge semantic correctness. Remove only your own scratch files and undo only your own protected-file edits. Never change expected answers or runner settings to make a check pass. Capture intended implementation files with `submit_patch` as the final tool action.

## Invocation

Scripts are self-contained Python files; no installation, network, task IDs or stored solutions are used. Invoke through the harness's exposed `run_skill_script` schema. Use the standard ADK fields: `skill_name="repo-navigation"`, `file_path="scripts/inspect_source.py"` and `args={"workspace": "<observed workspace>", "path": "<source.py>", "symbol": "<Class.method>"}`. For `check.py`, replace `path`/`symbol` with `module`/`code`; for `audit_patch.py`, only `workspace` is needed. All scripts accept `--workspace`; inspect also accepts `--start`, `--lines`, and check accepts `--timeout` (1–20 seconds).

Load domain files through `load_skill_resource`, with `skill_name="repo-navigation"` and `file_path="references/<name>.md"`. The ZIP builder generates these ADK-compatible paths from the source `resources/` directory. If resources cannot be loaded, apply the checkpoints above and continue; do not guess alternative host paths or copy skill files into the patch.


## Submission 10 repair gate
Use scripts/repair_gate.py after a substantive patch instead of a separate check.py call. Pass requirements (1–4 stable comma-separated IDs), covers (only IDs with independently justified assertions actually tested), code (assertions), optional module, workspace and observed time_left/calls_left. It invokes the isolated checker, rejects vacuous assertions that do not inspect runtime values, stores state in repository Git metadata (outside the patch), invalidates passing evidence after edits, counts up to two actual patch-changing repair cycles and inconclusive alternatives to one, and stops repeated unchanged failures. Follow its next_action, but diagnose failures before editing. A gate error or exhausted budget means audit_then_submit. The gate never edits or submits code and does not prove hidden-test success. Invoke it with `run_skill_script`, `skill_name="repo-navigation"`, `file_path="scripts/repair_gate.py"` and `args` containing `workspace`, `requirements`, `covers`, `code`, optional `module`, and observed `time_left` / `calls_left`. Omit unknown budgets rather than inventing numbers; do not use `run_command` to execute the skill directly.