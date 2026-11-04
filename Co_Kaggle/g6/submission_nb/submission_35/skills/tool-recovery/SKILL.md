---
name: tool-recovery
description: Consult only after native edit_file fails twice at one site, mandatory arguments repeatedly fail, or a focused check contradicts the patch. Provides safe recovery with native tools, not a replacement tool API.
---

# Tool failure recovery (native tools only)

You still own the original issue. Do not use `run_skill_script`, invent a
patch tool, or hand off to another agent. This skill adds NO executable tools.

## Identify the failure from the last observed response

- **Invalid tool dispatch or mandatory parameters missing**: The call did
  NOT execute. Supply all three required strings to native `edit_file`:
  `filepath`, `old_string`, `new_string`. `allow_multiple=false`.
  If two corrected calls still fail before execution, STOP this route.
- **FileEditError: anchor not found/ambiguous**: This is NOT a syntax
  error and not success. Read the precise location, then change the anchor
  using 2–8 observed unique lines. No identical third retry.
- **No changes to apply**: Compare current code against the intended
  result. It may already be present. Do not attempt a no-op replacement.
- **Test/command failed**: Distinguish a genuine assertion failure from
  missing fixtures/dependencies and from zero collected tests. Only a
  relevant executed assertion supplies behavior evidence.

## Escalation: one bounded git apply, not destructive file rewrite

If the source and target change are still certain, and native edit_file
has repeatedly failed, you may use ONE `run_command(command=...)` call.
Use `/tmp` for the patch so it does not enter the final diff. Construct
a minimal unified diff with exact context lines copied from fresh source:

```bash
cat > /tmp/s35-recovery.patch <<'PATCH'
diff --git a/pkg/module.py b/pkg/module.py
--- a/pkg/module.py
+++ b/pkg/module.py
@@ -12,3 +12,3 @@
 existing context line
-old code line
+new code line
 another context line
PATCH
test "$(wc -l < /tmp/s35-recovery.patch)" -le 80 &&
git apply --check /tmp/s35-recovery.patch &&
git apply /tmp/s35-recovery.patch &&
git diff --check
```

**Adapt the file, hunk header and exact code lines from the actual read.**
The example is not a literal patch to apply. `git apply --check` must
succeed before `git apply`. Never replace a whole existing file with
`write_file`, `cat > source`, `sed -i`, or a Python file rewrite. Never
generate a patch longer than 80 lines. If this fallback fails, STOP editing
that site and preserve any earlier valid source change.

## Verify and stop

Run a focused behavior check for the original issue after the final edit,
inspect the final bounded diff, and submit once. Don't claim an edit is
correct just because the tool reports `status=ok`. If verification is
blocked, report it honestly rather than inventing a passing test.
