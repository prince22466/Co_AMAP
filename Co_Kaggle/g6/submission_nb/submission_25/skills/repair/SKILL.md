---
name: repair
description: Search and read repository code, edit observed line ranges with syntax checks, and execute focused behavioral probes or tests. Use for every repository operation during a repair.
---

Call run_skill_script with skill_name="repair", file_path="scripts/work.py".
Pass args as an object. Only op and the fields for that operation are needed.

| op | Fields | Result |
| --- | --- | --- |
| status | none | State, available source/test paths, operation help |
| search | query; optional path | Regex matches with line numbers and existing test paths |
| read | path; optional start, end | Numbered source; authorizes those lines for one edit |
| edit | path, start, end, text | Replace inclusive lines; reject stale read, invalid syntax, or a broken saved check |
| probe | code | Run Python assertions with repository imports; save the exact check |
| test | path; optional query | Run one existing pytest file/node, optionally filtered by -k |
| check | none | Rerun the saved probe/test against current source |
| diff | none | Current diff, source-change summary, and current check evidence |

For example: args={"op":"read","path":"formatter.py","start":1,"end":20}.
Then args={"op":"edit","path":"formatter.py","start":2,"end":2,
"text":"    return value.rstrip()\n"}. Use JSON newlines, not double-escaped
backslash-n between source lines. The script never guesses how to unescape code.

Each read is capped at 100 lines and a bounded JSON size. Only complete displayed
lines authorize an edit. Up to 12 edit attempts are processed; an identical failed edit cannot
be retried against unchanged source. After editing, use check to reuse a probe
or test. Passing checks are remembered (up to two) and rerun during later edits.
A rejected edit restores the previous file. A small recovery journal also
restores it on the next operation if an edit was interrupted.

PASS means a check executed successfully, not proof that the task is solved.
fail_to_pass additionally means the identical check failed on the clean starting
tree before passing on this patch. A docstring-only diff is reported explicitly.
Tests with zero executed cases and probes without an executed assertion linked
through imports/assignments to observed local Python activity are BLOCKED. Inspect failure output; never claim it passed.
Missing tests or imperfect probes do not prevent submitting a valid partial fix.
Checks that modify tracked files or create non-ignored files are blocked and
their file changes restored. Put temporary test output outside the checkout.

Use native submit_patch after checking the real diff. The script does not call
the model, use reference answers, install dependencies, or change the evaluator.
