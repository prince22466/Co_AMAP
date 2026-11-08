---
name: state-guard
description: Sandbox-persistent memory and guarded source reads/edits after repeated native tool failures. Use only as a targeted recovery path.
---

# Stateful retry recovery

Use this skill after a failed native `edit_file` anchor or when you find
yourself reading the same unchanged source repeatedly. It is not a mandatory
first step. Native tools remain preferred for ordinary work.

This skill contains one Python script:
`scripts/state_guard.py`. It executes inside the task sandbox and records
attempt fingerprints in `/workspace/.git/s38_state_guard.json` (or the
equivalent per-task subprocess workspace). That ledger is not included in the
final git patch; it is local to the task and disappears when its sandbox ends.

Call `run_skill_script` with these exact argument names:

```text
skill_name: "state-guard"
file_path: "scripts/state_guard.py"
args: {"action": "read", "file": "pkg/module.py", "start": 50, "end": 85}
```

To make one bounded EXACT replacement from freshly read source:

```text
skill_name: "state-guard"
file_path: "scripts/state_guard.py"
args: {"action": "edit", "file": "pkg/module.py", "old": "exact source lines", "new": "replacement code"}
```

Check the returned JSON:
- `ok` read: source lines are numbered; use an exact, unique substring
- `ok` edit: a bounded change was applied; run a targeted check and diff
- `rejected`: old anchor missing/ambiguous; the failure is recorded
- `blocked_duplicate`: an identical read or FAILED edit was already tried
  on unchanged source; **do not repeat** it; change range, source or anchor
- `error`: never assume the edit happened, inspect the failure

One guard invocation is ONE charged tool call. The script rejects unsafe
paths and nonunique edits and never changes files outside the repository.
The skill DOES NOT intercept native tools, and it cannot prevent E4B from
ignoring this recovery path. Do not use it for reference fixes, grading,
network access or speculative huge rewrites.
