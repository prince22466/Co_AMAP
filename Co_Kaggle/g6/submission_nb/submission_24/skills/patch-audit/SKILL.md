---
name: patch-audit
description: Inspect the current patch for syntax errors, lost top-level definitions and empty changes during the independent review stage.
---

Run scripts/audit.py with no arguments using run_skill_script. Read its blockers
and changed paths. It is read-only, does not submit, and never proves behavior.
Use native commands if skill dispatch fails; do not invent other skill paths.
