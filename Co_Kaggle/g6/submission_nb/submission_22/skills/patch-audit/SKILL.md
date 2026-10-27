---
name: patch-audit
description: Optional deterministic edit recovery, structural patch inspection and focused candidate checks; native tools remain directly available.
---

Native edit_file / read_file / run_command / submit_patch are the primary tools.
Only when helpful:
* scripts/recover.py --path <relative-file> --old <exact-snippet> --new <replacement>
  applies one unique safe replacement, returns bounded context on a mismatch,
  refuses repeated identical failures, and reverts on syntax failure.
* scripts/audit.py (NO arguments) inspects the current repository diff, protected
  paths, syntax, changed definitions and patch digest. Never proves behavior.
* scripts/verify.py --module <module> --probe <Python assertion> checks the final
  candidate, local import provenance and changed-line execution, not the
  unmodified baseline. A candidate
  pass alone never establishes regression correctness.

Use run_skill_script(skill_name="patch-audit",file_path="scripts/NAME.py",
args={...}) for recovery/verification. No arguments for audit. If a skill
dispatcher fails, fall back to native tools; do not invent a new edit gateway.
