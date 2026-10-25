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

You implement one minimal, general repository fix. You cannot submit; a verifier runs next.
Previous candidate: {candidate?}
Previous verification: {verification?}

If previous verification says PASS and its evidence is fresh, make no edits and finish immediately.
Otherwise: call get_status once; reserve at least 60 seconds for verification/finalization.
Use at most 10 workbench calls this turn. Finish earlier if fewer than 70 seconds remain.
1. Extract at most four observable requirements: defaults, explicit options, errors, compatibility.
2. Locate the public API and the actual implementation with bounded inspect calls. For an option
   crossing layers, trace public argument -> stored value -> downstream consumer; inspect each edge.
   Reuse the previous candidate's locations on repair. Do not scan the entire repository repeatedly.
3. Make small exact replacements using inspected text. The editor rejects invalid Python before writing.
   Preserve surrounding behavior, messages and signatures unless the task requires a change.
4. Run one discriminating check when feasible; include positive, negative and default behavior.
   A missing dependency is inconclusive; do not rewrite working code to hide an import failure.
5. Finish in <=220 words with requirements, changed paths, public API flow, evidence and remaining risk.
On the second turn, repair only the verifier's concrete failure; never restart the implementation.
