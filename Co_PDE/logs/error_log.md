# PDE Error Log

## Summary (auto-updated)
- classification errors: 0
- method errors: 0
- BC/IC errors: 0
- algebra errors: 0
- logic errors: 0
- verification missed: 0

---

## Entries

### [2026-04-01] Diffusion - Separation of Variables

Problem:
Heat equation with Dirichlet BC

Error Type:
- boundary condition error
- method detail error

What I did wrong:
Used cosine basis instead of sine → violated u(0,t)=0

Correct reasoning:
Dirichlet zero BC → sine eigenfunctions

Fix rule:
Always derive eigenfunctions from BC, not memory

---

### [2026-04-01] First-order PDE - Characteristics

Error Type:
- logic error

What I did wrong:
Forgot constant depends on invariant

Correct reasoning:
C = C(invariant), not constant scalar

Fix rule:
Always express solution in terms of characteristic invariants
