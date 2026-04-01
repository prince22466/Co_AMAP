---
name: pde_error_tracker
description: Tracks mistakes and enforces correction loops.
---

# Error Categories

- classification error
- wrong method
- boundary condition error
- initial condition mismatch
- algebra mistake
- logic gap
- skipped verification

---

# Workflow

For each error:

1. identify type
2. explain failure
3. show corrected step
4. extract rule

---

# Rule

If same error repeats → force drill

---

# File Integration

All errors MUST be persisted in:

logs/error_log.md

---

# Logging Format (STRICT)

Use this exact structure:

### [YYYY-MM-DD] [Topic] - [Method]

Problem:
<short description>

Error Type:
- ...

What I did wrong:
<1–2 lines>

Correct reasoning:
<1–2 lines>

Fix rule:
<actionable rule>

---

# Summary Update

At top of file, maintain counters:

- classification errors: X
- method errors: X
- BC/IC errors: X
- algebra errors: X
- logic errors: X
- verification missed: X

---

# Weak Area Detection

If same error appears twice:

Add/update section:

## Weak Areas

- <topic>: weak

AND notify orchestrator to assign drills
