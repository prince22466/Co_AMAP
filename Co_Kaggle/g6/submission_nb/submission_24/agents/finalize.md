Task: {problem_description}
Candidate: {candidate?}
Final independent review: {review?}

Finish the existing attempt. Only this stage has submit_patch. Use real tool
calls, not code blocks containing tool names. No new exploration or source edits.
Use at most 2 charged calls. Check get_status, then inspect git status --short,
git diff HEAD --stat and git diff --check together if budget remains. Include
untracked source; submit_patch captures it automatically. Do not commit, reset,
stash, remove source, or modify tests/runner configuration.
Call submit_patch exactly once, even when the review is BLOCKED or unresolved:
preserve useful work for the independent evaluator instead of discarding it.
Report its actual patch_size; never claim an empty patch or unverified behavior
was successful. End immediately after submission with a brief factual status.
