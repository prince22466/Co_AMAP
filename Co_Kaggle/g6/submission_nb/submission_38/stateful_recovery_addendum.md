
S38 COMPETITION-COMPATIBLE STATEFUL RECOVERY:
Keep native tools for normal coding. If one edit_file fails with a missing or
ambiguous anchor, STOP guessing edits. Use load_skill(skill_name="state-guard")
once, then run_skill_script with skill_name="state-guard",
file_path="scripts/state_guard.py", and args fields action, file, start/end
or old/new. First action=read the current relevant lines; then action=edit
with a verified exact unique anchor. This script remembers failed edits and
unchanged reads across calls in the task sandbox: blocked_duplicate means
change the source range/anchor, never repeat it. Limit recovery to 2 calls
before changing strategy or submitting an existing valid patch. Keep the
skill out of the normal fast path; verify requested behavior after any edit.
