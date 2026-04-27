---
name: harmonic-functions-pde-study
description: Use this skill when studying harmonic functions, Laplace equation, maximum principle, Harnack inequality, Hopf principle, removable singularities, radial replacement, or related beginner PDE proofs. The learner is new to PDE and wants intuition plus proof structure.
---

# Harmonic Functions PDE Study Skill

## Purpose
Guide a beginner through harmonic-function PDE material using intuition, examples, proof templates, and small verification tasks. This skill is for study, explanation, and experiment generation, not for producing production software.

## Activation Cues
Use this skill for prompts involving:
- `Delta u = 0`, Laplace equation, harmonic functions
- mean value property
- maximum principle or strong maximum principle
- subharmonic functions
- Harnack inequality
- Hopf boundary point principle
- removable singularities
- radial replacement / spherical averages
- Green's functions or radial harmonic functions
- beginner PDE confusion

## Teaching Protocol
For every topic, follow this order:

1. **Plain meaning first**
   Explain the result in one or two sentences without heavy notation.

2. **Assumptions**
   State exactly what is assumed, for example:
   - `u` harmonic: `Delta u = 0`
   - `u >= 0` or `u > 0`
   - domain is all of `R^n` or a punctured ball
   - boundedness or growth assumptions
   - compactness of `K`

3. **Tool used**
   Identify the main tool:
   - mean value property
   - maximum principle
   - Harnack inequality
   - barrier function
   - derivative estimate
   - scaling
   - Weierstrass M-test

4. **Proof skeleton**
   Give numbered steps. Avoid hiding key inequalities.

5. **2D example**
   Use `u(x,y)=x^2-y^2`, `u(x,y)=log sqrt(x^2+y^2)`, or `u(x,y)=x` when appropriate.

6. **Common confusion**
   Add a small section: “What usually confuses people here.”

7. **Application or purpose**
   Explain why the result matters in PDE, physics, optimization, or robotics.

## Core Definitions

### Harmonic
`u` is harmonic if

```math
\Delta u=0.
```

In 2D:

```math
\Delta u=u_{xx}+u_{yy}.
```

Interpretation: local equilibrium; no source or sink inside the domain.

### Subharmonic
`w` is subharmonic if

```math
\Delta w\ge 0.
```

Interpretation: center value is at most the surrounding average.

### Ball and Boundary
`B_R(p)` means the ball centered at `p` with radius `R`.

`partial B_R(p)` means all points exactly distance `R` from `p`.

### Boundary Size
In estimates,

```math
\max_{\partial B_R(p)}|u|
```

means the largest absolute value of `u` on the boundary sphere. It does not mean the geometric area of the sphere.

### Compact Set
In `R^n`, compact means closed and bounded. Compactness matters because:
- finite ball covers exist;
- max and min exist for continuous functions;
- the set stays away from the boundary if compactly contained in an open domain.

### Uniform Convergence on K
For partial sums `S_N(x)`, convergence is uniform on `K` if

```math
\sup_{x\in K}|S_N(x)-U(x)|\to 0.
```

Meaning: one sufficiently large `N` works for all points in `K`.

## Key Results and How to Explain Them

### Mean Value Property
For harmonic `u`,

```math
u(p)=\text{average of }u\text{ on a surrounding sphere or ball}.
```

Use the intuition: the point is balanced by its surroundings.

### Maximum Principle
A nonconstant harmonic function cannot have an interior maximum or minimum.

Use the average argument: if the center equals the average and all nearby values are less than or equal to the center, then all nearby values must equal the center.

### Derivative Estimate
For multi-index `alpha`,

```math
|D^\alpha u(p)|\le \frac{(n|\alpha|)^{|\alpha|}}{R^{|\alpha|}}\max_{\partial B_R(p)}|u|.
```

Explain as:

```math
\text{derivative order }k \lesssim \text{boundary size}/R^k.
```

### Analyticity
Harmonic functions equal their Taylor series locally.

Proof idea:
- derivative estimates control Taylor coefficients;
- Taylor remainder goes to zero.

### Convex Composition
If `u` is harmonic and `F` is convex, then `F(u)` is subharmonic:

```math
\Delta(F(u))=F''(u)|\nabla u|^2+F'(u)\Delta u.
```

Since `Delta u=0` and `F''>=0`, get `Delta(F(u))>=0`.

### L2 Liouville
If `u` is harmonic on all `R^n` and `int u^2 < infinity`, then `u=0`.

Key inequality:

```math
0\le u^2(x)\le M/|B_R(x)|.
```

Valid for every `R>0`; send `R` to infinity.

### Harnack Inequality
For positive harmonic functions, values inside a smaller ball are comparable:

```math
\max_{B_{R/2}}u\le 3^n\min_{B_{R/2}}u.
```

Do not obsess over `3^n`. Emphasize:

```math
\text{positive harmonic functions cannot jump wildly nearby.}
```

### Harnack Chain on Compact Sets
To prove

```math
\max_K u\le \gamma\min_K u,
```

use overlapping balls. Each ball gives one comparison factor. Finitely many balls give a finite total factor.

### Series of Positive Harmonic Functions
If `u_i>=0` are harmonic and `sum u_i(x0)` converges, then the series converges uniformly on compact sets.

Use Harnack:

```math
u_i(x)\le \gamma u_i(x0)
```

then Weierstrass M-test.

### Hopf Principle
If `u>0` inside, `u(x0)=0` at the boundary, and an interior ball touches the boundary at `x0`, then

```math
\partial_\nu u(x0)>0.
```

Explain with the barrier picture: `u` is above a simple function that rises linearly inward.

### Removable Singularity
If `u` is harmonic on a punctured domain and bounded near the missing point, then the missing point can be filled in harmonically.

2D barrier:

```math
h(x)=2M\frac{\log(|x|/R)}{\log(r/R)}.
```

Key idea: shrink the inner circle and the barrier collapses to zero at fixed points.

### Radial Replacement
Define

```math
u^R(x)=\frac{1}{n\omega_n |x|^{n-1}}\int_{\partial B_{|x|}(0)}u(\sigma)d\sigma.
```

This averages over angles and keeps only dependence on radius.

For harmonic `u` on a punctured unit ball with zero boundary limit:
- `n=2`: `u^R=C log |x|`
- `n>=3`: `u^R=C(|x|^{2-n}-1)`

## Study Exercises to Generate
When asked for practice, generate small exercises such as:

1. Verify that `u(x,y)=x^2-y^2` is harmonic.
2. Check that `u(x,y)=x^2+y^2` is not harmonic.
3. Compute `Delta(u^2)` when `u` is harmonic.
4. Show `log r` is harmonic in 2D away from `0`.
5. Show `r^{2-n}` is harmonic in `n>=3` away from `0`.
6. Numerically average `u(x,y)=x^2-y^2` on a circle and compare to center value.
7. Build a finite-difference Laplace solver on a square and verify the maximum principle numerically.
8. Implement a harmonic potential field for simple 2D robot navigation with obstacles.

## Coding Experiment Guidelines
When generating code:
- Prefer Python + NumPy + Matplotlib.
- Keep scripts short and runnable.
- Avoid over-engineering.
- Include visualizations:
  - heatmap of harmonic solution;
  - contour plot of potential field;
  - check max/min values against boundary values;
  - circle averages for mean value property.
- Add comments explaining the PDE meaning of each step.

## Common Mistakes to Catch
- Thinking `Delta u=0` means first derivative is zero.
- Confusing boundary size with boundary area.
- Forgetting Harnack requires nonnegative or positive harmonic functions.
- Applying maximum principle on unbounded domains without a cutoff or barrier.
- Sending `R -> infinity` when the domain is not all of `R^n`.
- Forgetting that derivatives commute only under sufficient regularity, though harmonic functions are eventually smooth/analytic.
- Treating pointwise convergence as uniform convergence.
- Ignoring that removable singularity requires boundedness near the missing point.

## Preferred Output Format
Use this structure:

```markdown
## Plain meaning
...

## Assumptions
...

## Main tool
...

## Proof idea
1. ...
2. ...
3. ...

## Simple example
...

## Why this matters
...

## Common confusion
...
```

## Main Mental Model
Harmonic functions are equilibrium functions. Boundary behavior strongly controls the interior. Most proofs are variations of:

```math
\text{harmonicity + maximum principle/barrier/Harnack} \Rightarrow \text{strong control}.
```
