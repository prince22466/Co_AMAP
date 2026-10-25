---
name: repo-navigation
description: Inspect repository code, apply syntax-checked exact edits, run bounded local assertions and audit a patch. Use for repository repair and verification.
---

Call `run_skill_script` with `skill_name="repo-navigation"`,
`file_path="scripts/workbench.py"` and an `args` dictionary. The script locates the
real Git checkout from PYTHONPATH or the sandbox working directory, including subprocess mode.
Do not supply `/workspace` manually. Calling load_skill first is unnecessary.

Actions: `inspect` (query or path/symbol/start/end), `replace` (path/old/new),
`create` (path/new; new files only), `check` (module/code/timeout), `audit` (no extra fields).
Example args: `{"action":"inspect","query":"def render"}`.
Example check: `{"action":"check","module":"package","code":"import package\nassert package.answer() == 42"}`.
Checks run in a fresh process with src/ and repository imports first. At least one assertion
must execute, the named module must come from the checkout, and caught assertion failures fail.
Checks time out after at most 15 seconds. Audit never upgrades an error to a pass.

Use exact source text from inspect for edits. An ambiguous replacement or invalid Python leaves
files unchanged. Repeated identical failures are rejected until the patch or arguments change.
Never edit tests or test/dependency configuration. The script is a reliability aid, not a security
boundary: arbitrary check code runs inside the competition's sandbox.

For multi-layer options, inspect the public signature, forwarding call and consumer. Use a public
behavioral assertion to verify that chain. Structural similarity alone is not correctness evidence.
