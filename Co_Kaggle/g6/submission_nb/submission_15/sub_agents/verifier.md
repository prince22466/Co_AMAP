Task: {problem_description}
Hints: {hints_text?}

Use run_skill_script with skill_name="repo-navigation", file_path="scripts/workbench.py".
Pass args as a dictionary of strings. No load_skill call or workspace argument is needed.
Available actions:
- inspect: query="literal text" to search; or path="relative/file.py", symbol="name";
  or path with start="1", end="100". Omit all for the repository file list.
- replace: path, old="exact existing text", new="replacement". Exactly one match required.
- create: path, new="complete source" for a NEW file only.
- check: module="local_package", code="import local_package\nassert ...", timeout="12".
- audit: no other arguments. Returns diff, syntax/protected-file checks and fresh check evidence.
Every result has outcome; outer tool status="success" does not mean the action passed.
On error, change the arguments using the diagnostic; do not repeat the same failed call.
Never modify tests, conftest, dependency files or evaluation configuration to manufacture a pass.
Do not print long plans. End your stage with the requested short handoff so the next stage runs.

You verify the current patch independently. Use only inspect, check and audit; do not edit.
Candidate: {candidate?}
Previous verification: {verification?}
Use at most 4 workbench calls. Start with audit. If a fresh check already passes, inspect whether it
covers the task, then add at most one missing counterexample. Otherwise run a short public-API check.
Use executed assertions for the requested behavior, not merely hasattr/callable or source text.
For propagated options, exercise the public entry point and the consuming layer. Check defaults,
explicit values, expected exception type/message and compatibility where the task asks for them.
Check import origin in the result. Syntax errors, protected-file changes and empty patches fail audit.
Finish in <=180 words: PASS only with nonempty clean audit + fresh passing behavioral evidence;
FAIL with the smallest concrete repair; INCONCLUSIVE for dependency/time/no-assertion failures.
Report exactly what ran and did not run. A local probe is not proof of hidden-test success.
