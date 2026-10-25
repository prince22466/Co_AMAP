---
name: repo-verification
description: Verifier workbench with mandatory actions, bounded attempts and patch-bound assertion evidence.
---

Call run_skill_script with skill_name="repo-verification", file_path="scripts/workbench.py".
ALWAYS supply args.action, for example args={"action":"audit"}.
Role: verifier. Allowed actions: audit, check, inspect, verify.
Inspection: args={"action":"inspect","path":"package","query":"def render"}.
Edit example: args={"action":"replace","path":"pkg/__init__.py","old":"return 41","new":"return 42","module":"pkg","code":"import pkg\nassert pkg.answer() == 42"}.
Edits require module/code BEFORE writing and automatically execute that assertion afterward.
A failed assertion retains the patch for repair; it never means verified. module="-" explicitly
selects non-importable artifact assertions without module-origin checking.
Audit is repeatable. verify re-runs the persisted check. All dispatch attempts, including rejected
calls, are counted; 28 non-audit attempts are allowed per task. These script limits do not force
an ADK stage handoff. The shared harness budget never resets between rounds.
No modifications to evaluator tests/configuration. .github workflows are permitted repository artifacts.
Checks are sandbox code, not a security boundary. Assertion coverage remains the model's responsibility.
