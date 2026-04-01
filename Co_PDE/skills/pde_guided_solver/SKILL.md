---
name: pde_guided_solver
description: Step-by-step PDE solving with strict structure and minimal hints.
---

# Mandatory First Step: Problem Scan

Always extract:

- PDE type (elliptic/parabolic/hyperbolic)
- order
- linearity
- domain
- IC/BC
- target

---

# Method Selection

State:
- candidate methods
- chosen method
- why
- why not others

---

# Hint Ladder

1. classification hint
2. setup hint
3. first key step
4. partial derivation
5. full solution (only if needed)

---

# Verification (MANDATORY)

- check PDE
- check IC/BC
- check constants

---

# Output Format

- Problem scan
- Method
- Steps
- Verification
- Key pattern
