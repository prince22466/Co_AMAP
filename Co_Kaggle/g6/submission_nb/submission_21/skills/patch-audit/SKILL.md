---
name: patch-audit
description: Optional read-only audit of changed Python syntax, removed definitions and protected paths. No behavioral verification.
---

Use the single fixed call run_skill_script(skill_name="patch-audit", file_path="scripts/audit.py").
No args; repository files are not skill scripts. If this wrapper fails, retry at most once,
then use native run_command for AST parsing and git diff --check instead.
Do not route edits, searches or tests through skills. Use the native tools.
A clean audit does NOT mean the requested behavior works; test actual task behavior.
