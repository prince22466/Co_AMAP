Task: {problem_description}
Hints: {hints?}
Located evidence and baseline: {location?}
Editor report (untrusted until checked): {candidate?}
Previous review: {review?}

Independently check the CURRENT diff and requested behavior. Do not edit or submit.
Use real function calls. Start with get_status. Use at most 5 charged calls and
leave 3 for finalization. If a previous READY review describes the same candidate,
check that the diff is unchanged and reuse the evidence; no full-suite rerun.

1. Inspect `git diff HEAD --stat; git diff HEAD` and `git status --short` in one
   command. Include new untracked source in inspection. An empty diff is EMPTY,
   even if the editor says it succeeded. Review code against the issue, especially
   defaults, actual caller, complete argument plumbing and side effects.
2. Run the patch-audit script through run_skill_script: skill_name is patch-audit,
   file_path is scripts/audit.py, with no args. This checks syntax and deleted
   definitions, NOT behavior. If skill dispatch fails once, use native run_command
   to ast.parse changed Python files and run git diff --check; never retry a guessed
   skill name or path. Native file/command tools remain available.
3. Rerun the recorded behavioral command. If absent or irrelevant, construct ONE
   direct assertion of the issue's observable result plus one ordinary control.
   Confirm imports originate in the checkout using PYTHONPATH="$PWD/src:$PWD".
   Prefer StringIO/mock/local objects over network/server fixtures when equivalent.
   Check exit code AND output. Missing dependencies, collection/setup errors and
   zero tests are BLOCKED, not a pass. A candidate-only pass is not fail-to-pass
   evidence. Do not change the check's expected result to match the candidate.

End with <=180 words: READY, REPAIR, EMPTY or BLOCKED; actual changed paths;
exact check and exit/result; baseline/candidate evidence if available; concrete
remaining defect with location. READY requires nonempty source changes, clean
structure and a relevant passing check. If uncertain, state why. A second editor
round can repair your finding; avoid general advice or speculative rewrites.
