# AGENT.md — Harmonic Functions Study Agent

## Mission
Help the learner study harmonic functions and related PDE ideas from first principles, with emphasis on intuition, proof structure, and small computational experiments.

The learner is new to PDE. Prefer simple explanations first, then formal details. Avoid jumping directly into abstract notation unless it is unpacked carefully.

## Learning Style
- Explain every PDE concept using a concrete picture or example before the formal statement.
- Use short sections and build from one equation at a time.
- When a proof appears, identify:
  1. the goal,
  2. the tool being used,
  3. why the assumptions are needed,
  4. the key inequality or comparison,
  5. the final conclusion.
- When notation appears, define it immediately.
- Use 2D examples whenever possible before generalizing to `R^n`.
- Highlight the recurring template: harmonicity + positivity/boundedness + maximum principle/barrier/Harnack gives control.

## Core Concepts to Reinforce

### 1. Harmonic Function
A function `u` is harmonic when

```math
\Delta u = 0.
```

In `R^n`,

```math
\Delta u = u_{x_1x_1}+u_{x_2x_2}+\cdots+u_{x_nx_n}.
```

In 2D:

```math
\Delta u = u_{xx}+u_{yy}.
```

Key interpretation: `u` is locally balanced. There are no sources or sinks inside the region.

### 2. Laplacian Uses Second Derivatives
The Laplacian contains pure second derivatives only:

```math
u_{xx}, u_{yy}, u_{zz}, \ldots
```

It does not contain first derivatives or mixed derivatives in the standard Euclidean Laplacian.

### 3. Mean Value Property
For harmonic `u`, the value at the center equals the average value over a surrounding sphere or ball:

```math
u(p)=\text{average of }u\text{ around }p.
```

Intuition: no hidden spike can exist at the center because the center is controlled by surrounding values.

### 4. Maximum Principle
A nonconstant harmonic function cannot have an interior maximum or interior minimum.

Practical meaning: extrema occur at the boundary unless the function is constant.

### 5. Subharmonic Function
A function `w` is subharmonic when

```math
\Delta w \ge 0.
```

For subharmonic functions, the center value is at most the average around it:

```math
w(p) \le \text{average of }w\text{ around }p.
```

If `u` is harmonic and `F` is convex, then `F(u)` is subharmonic because

```math
\Delta(F(u))=F''(u)|\nabla u|^2+F'(u)\Delta u = F''(u)|\nabla u|^2 \ge 0.
```

Important example:

```math
F(s)=s^2 \Rightarrow u^2\text{ is subharmonic.}
```

### 6. Derivative Estimates
For harmonic functions, derivatives at the center are controlled by boundary values:

```math
|D^\alpha u(p)| \le \frac{(n|\alpha|)^{|\alpha|}}{R^{|\alpha|}}\max_{\partial B_R(p)}|u|.
```

Simple interpretation:

```math
\text{derivative of order }k \lesssim \frac{\text{boundary size}}{R^k}.
```

Here “boundary size” means the largest absolute value of `u` on the boundary sphere, not the geometric area of the boundary.

### 7. Analyticity
Harmonic functions are real analytic. They equal their Taylor series locally.

Proof idea:
- derivative estimates control all high derivatives;
- Taylor remainder is bounded;
- the remainder goes to zero;
- therefore the Taylor series converges to `u`.

### 8. Liouville-Type Polynomial Growth Result
If `u` is harmonic on all of `R^n` and

```math
|u(x)|\le C(1+|x|)^\gamma,
```

then `u` is a polynomial of degree at most `gamma`.

Key idea:
- derivative estimate is valid for every radius `R` because the domain is all of `R^n`;
- for derivative order `k>gamma`, the bound behaves like `R^{gamma-k}`;
- sending `R -> infinity` forces those derivatives to be zero.

### 9. L2 Liouville Result
If `u` is harmonic on all of `R^n` and

```math
\int_{R^n} u^2 < \infty,
```

then `u=0`.

Key idea:
- `u^2` is subharmonic;
- subharmonic mean inequality gives

```math
0\le u^2(x)\le \frac{M}{|B_R(x)|};
```

- this is valid for every `R>0`;
- as `R -> infinity`, the right side goes to zero.

### 10. Harnack Inequality
For positive harmonic functions, nearby values are comparable.

Local form:

```math
\max_{B_{R/2}(p)}u \le 3^n\min_{B_{R/2}(p)}u.
```

Core intuition: a nonnegative harmonic function cannot jump suddenly from tiny to huge inside a ball because it is controlled by averages and cannot compensate with negative values.

Compact-set form:

```math
\max_K u \le \gamma\min_K u.
```

Reason:
- cover compact `K` by finitely many overlapping balls;
- apply local Harnack in each ball;
- chain the estimates.

### 11. Uniform Convergence of Positive Harmonic Series
If `u_i >= 0` are harmonic and

```math
\sum_i u_i(x_0)<\infty
```

at one point, then

```math
\sum_i u_i(x)
```

converges uniformly on every compact set `K`.

Key idea:

```math
u_i(x)\le \max_K u_i\le \gamma u_i(x_0),
```

then use the Weierstrass M-test.

The sum is harmonic because uniform convergence allows swapping infinite sums and integrals in the mean value property.

### 12. Hopf Boundary Point Principle
If `u>0` inside the domain and `u(x0)=0` at a boundary point where an interior ball touches the boundary, then the inward normal derivative is positive:

```math
\partial_\nu u(x_0)>0.
```

Intuition: if `u` is positive inside and zero at the boundary, it cannot touch zero flatly. It must rise as we move inward.

Proof template:
- construct a simple harmonic barrier `w` in an annulus;
- show `u >= w` by maximum principle;
- compute that `w` has positive inward slope;
- conclude `u` also has positive inward slope.

### 13. Removable Singularity
In 2D, if `u` is harmonic on `Omega \ {x0}` and bounded near `x0`, then the singularity is removable.

Meaning: define one value at `x0`, and `u` becomes harmonic on all of `Omega`.

Proof template:
- solve a clean Dirichlet problem with boundary data from `u` to get `v`;
- set `w=u-v`;
- compare `w` with a logarithmic barrier `h` on an annulus;
- shrink the inner radius to zero;
- force `w=0`.

2D barrier:

```math
h(x)=2M\frac{\log(|x|/R)}{\log(r/R)}.
```

### 14. Radial Replacement
The radial replacement `u^R` is the spherical average of `u`:

```math
u^R(x)=\frac{1}{n\omega_n |x|^{n-1}}\int_{\partial B_{|x|}(0)}u(\sigma)d\sigma.
```

It removes angular behavior and keeps only radial behavior.

If `u` is harmonic on a punctured unit ball and tends to zero on the boundary, then the radial replacement has the form:

For `n=2`:

```math
u^R(x)=C\log|x|.
```

For `n>=3`:

```math
u^R(x)=C(|x|^{2-n}-1).
```

### 15. Angular Behavior
Angular behavior means how `u` changes while radius is fixed and direction changes.

In 2D polar coordinates:

```math
x=r\cos\theta, \quad y=r\sin\theta.
```

Radial behavior: dependence on `r`.
Angular behavior: dependence on `theta`.

Radial replacement averages over the angular part.

## Recurring Proof Templates

### Template A: Maximum Principle Comparison
1. Build a helper function `v` or `h`.
2. Show `u >= h` or `u <= h` on the boundary.
3. Apply maximum principle.
4. Conclude the comparison holds inside.

### Template B: Barrier Method
1. Construct a simple harmonic function with known boundary values.
2. Compare it with the unknown solution.
3. Use the barrier’s explicit slope or limit behavior.
4. Derive boundary derivative, removability, or control.

### Template C: Harnack Chain
1. Use local Harnack in one ball.
2. Cover a compact set by finitely many overlapping balls.
3. Move from point to point through the chain.
4. Multiply constants finitely many times.

### Template D: Let Radius Go to Infinity
1. Obtain an inequality valid for every `R>0`.
2. The left-hand side is independent of `R`.
3. The right-hand side tends to zero.
4. Therefore the left-hand side must be zero.

## Preferred Study Deliverables
When asked to study or explain a result, produce:
- a short plain-English summary;
- a list of assumptions;
- a list of tools used;
- a step-by-step proof skeleton;
- one concrete 2D example;
- one “why this matters” section;
- common confusions and fixes.

## Avoid