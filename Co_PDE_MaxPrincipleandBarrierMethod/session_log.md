
# PDE Training Session Log

## Session

- Time: 2026-04-18 14:28:01 +02:00
- Topic: Maximum principle and barrier methods for the heat equation.

## Core Pattern

Shift -> Barrier -> Comparison -> Maximum principle.

Operator:

\[
Lu = u_t - u_{xx}.
\]

Comparison format used:

\[
Lv \ge 0,\quad v \ge 0 \text{ on the parabolic boundary}
\Rightarrow
v \ge 0 \text{ in the domain}.
\]

## Completed Drills

1. Non-negativity with no shift: \(v=u\).
2. Upper constant bound: \(v=2-u\).
3. Lower constant bound: \(v=u+3\).
4. Two-sided constant bound using two comparison functions.
5. Forcing \(Lu=2\), proving \(u\ge 0\).
6. Forcing \(Lu=-1\), proving \(u\le 3\).
7. Forcing \(Lu=5\), proving \(u\ge -2\).
8. Eigenfunction barrier: \(w=A\sin(\pi x)e^{-\pi^2t}\).
9. Slower eigenfunction barrier: \(w=A\sin(\pi x)e^{-\beta t}\), \(0<\beta\le \pi^2\).
10. Polynomial barrier: \(w=Ax(1-x)e^{-\beta t}\), \(0<\beta\le 8\).
11. Steady-state shift: \(z=u-u^s\), \(u^s=\frac12x(1-x)\).
12. Two-sided decay around steady state: \(|u-u^s|\le A\sin(\pi x)e^{-\pi^2t}\).
13. Hopf-style boundary slope intuition using \(w=\varepsilon x\).
14. Diagnosis of wrong-sign comparison functions.
15. Diagnosis of too-fast barrier decay \(e^{-10t}\).

## Mistakes Corrected

- Finish \(Lv\) computations by substituting the PDE.
- Do not assume \(w\ge u\); prove it from boundary signs plus comparison.
- Initial inequalities are often \(\ge 0\), not equality.
- For upper bounds, preferred format is \(v=c-u\) or \(v=w-u\).
- Strict signs like \(v<0\) are usually not guaranteed; use non-strict \(\le,\ge\).
- For barriers, \(Lv\ge 0\) is the key sign check.
- \(e^{-10t}\) is too fast for the first sine mode because \(10>\pi^2\).

## Current Status

Core maximum-principle training completed. The user can choose shifts, compute \(Lv\), check parabolic boundary signs, and identify valid or invalid barriers.
