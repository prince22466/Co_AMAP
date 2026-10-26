---
name: patch-audit
description: Read-only patch syntax, removed-definition and protected-path audit. Not behavioral verification.
---
Run run_skill_script(skill_name="patch-audit", file_path="scripts/audit.py").
No args, action, module, source code or repository filepath is needed.
This skill never edits files or runs repository code. Continue using native read_file/edit_file.
Repair blockers. Review removed definitions against the task. behavior_verified is always false:
a clean structural audit never substitutes for an output-sensitive test of the requested behavior.
