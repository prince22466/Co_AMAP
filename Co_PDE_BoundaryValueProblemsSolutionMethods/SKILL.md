---
name: boundary-value-problems-solution-methods
description: Use this skill when studying PDE boundary-value problems and solution methods, including Laplace and Poisson equations, Dirichlet and Neumann conditions, separation of variables, Fourier series, Fourier transform, Poisson kernels, reflection principle, Kelvin transform, wedge angle solutions, maximum principle barriers, exterior uniqueness, and Green identity energy methods.
---

# Boundary-Value Problems Solution Methods

## Purpose

Guide the user through graduate-level boundary-value problems step by step.

The user is new to this material and often gets lost when several methods are combined. Explain slowly, focus on geometry first, and connect every formula to the boundary conditions.

## Core Rule

```text
Domain shape chooses coordinates.
Boundary type chooses method.
```

## Core Workflow

For every problem, follow this order:

1. Identify the PDE.
2. Identify the domain.
3. Split the boundary into pieces.
4. Identify the boundary condition type on each piece.
5. Choose coordinates based on geometry.
6. Choose solution method based on boundary/domain type.
7. Derive the formula.
8. Check PDE and boundary conditions.
9. Explain uniqueness if relevant.
10. End with a short memory box.

## Boundary Condition Types

### Dirichlet

The value of `u` is prescribed:

```math
u=g \quad \text{on } ∂Ω
```

### Neumann

The normal derivative is prescribed:

```math
∂_νu=g \quad \text{on } ∂Ω
```

For Neumann problems, adding a constant does not change the boundary condition:

```math
∂_ν(u+C)=∂_νu
```

Therefore Neumann uniqueness is only up to constants.

### Mixed

Different parts of the boundary have different conditions.

Example:

```math
u=0 \quad \text{on one side}
```

```math
u_x=0 \quad \text{on another side}
```

## Domain-to-Method Map

| Domain | Common method |
|---|---|
| Rectangle | separation of variables + Fourier series |
| Strip | Fourier series or Fourier transform depending on finite/infinite direction |
| Disk | polar coordinates + Fourier series or Poisson kernel |
| Annulus | polar coordinates + Fourier series with `r^n`, `r^{-n}`, `log r` |
| Ball | spherical coordinates + Legendre polynomials or spherical harmonics |
| Upper half-plane | Fourier transform + Poisson kernel |
| Right half-plane | angle method or rotated Poisson kernel |
| Upper half-disk | subtract auxiliary function + reflection + disk Poisson kernel |
| Wedge / quadrant | normalized angle solution |
| Exterior domain | maximum principle barriers or energy method |
| Circle-line transformed geometry | Kelvin transform |

## Method 1: Separation of Variables

Use for rectangles and product domains.

Try:

```math
u(x,y)=X(x)Y(y)
```

After substitution, usually get:

```math
\frac{X''}{X}=-\frac{Y''}{Y}=λ
```

Explain why both sides equal a constant:

If `F(x)=G(y)` for every independent `x,y`, then fixing `y` makes `F(x)` equal to one number for all `x`, so `F` is constant. Similarly `G` is constant.

### Eigenvalue sign logic

Do not just choose `λ=-μ²`. Test cases if needed:

- `λ=0`: often only trivial solution.
- `λ>0`: may give sinh/cosh and only trivial solution under boundary conditions.
- `λ<0`: gives sine/cosine and nontrivial modes.

### ODE memory

```math
f''+\mu^2f=0
```

gives:

```math
\sin(\mu x),\cos(\mu x)
```

```math
f''-\mu^2f=0
```

gives:

```math
e^{\mu x}, e^{-\mu x}
```

or:

```math
\sinh(\mu x),\cosh(\mu x)
```

## Method 2: Fourier Series

Use when the variable is finite or periodic.

Examples:

```math
0<x<L
```

or:

```math
0≤θ≤2π
```

Common expansions:

```math
g(x)=\sum a_n\sin(nπx/L)
```

```math
g(θ)=\frac{a_0}{2}+\sum_{n=1}^\infty(a_n\cos nθ+b_n\sin nθ)
```

Use orthogonality to find coefficients.

Memory:

```text
finite/periodic domain → Fourier series → discrete modes
```

## Method 3: Fourier Transform

Use when the variable lies on the whole real line.

Example:

```math
x∈\mathbb R
```

Transform in `x`:

```math
\hat u(ξ,y)=\int_{\mathbb R}e^{-iξx}u(x,y)\,dx
```

Derivative rules:

```math
\widehat{u_x}=iξ\hat u
```

```math
\widehat{u_{xx}}=-ξ^2\hat u
```

So Laplace equation:

```math
u_{xx}+u_{yy}=0
```

becomes:

```math
\hat u_{yy}-ξ^2\hat u=0
```

Boundedness in `y` gives:

```math
\hat u(ξ,y)=\hat g(ξ)e^{-|ξ|y}
```

Then invert to obtain the Poisson kernel.

Memory:

```text
infinite line → Fourier transform → continuous frequencies
```

## Method 4: Poisson Kernel

Use to solve standard Dirichlet problems.

### Upper half-plane

For `y>0`:

```math
u(x,y)=\frac1π\int_{\mathbb R}
\frac{y}{(x-s)^2+y^2}g(s)\,ds
```

Interpretation:

```text
u(x,y) is a weighted average of boundary values g(s).
```

Kernel properties:

```math
P_y≥0
```

```math
\int_{\mathbb R}P_y=1
```

as `y→0+`, `P_y` concentrates near `s=x`.

### Unit disk

For `x²+y²<1`:

```math
u(x,y)=
\frac{1-x^2-y^2}{2π}
\int_0^{2π}
\frac{g(\cos θ,\sin θ)}
{(x-\cos θ)^2+(y-\sin θ)^2}\,dθ
```

## Method 5: Polar Coordinates

Use for disks, annuli, wedges, and quadrants.

Polar coordinates:

```math
x=r\cos θ,\quad y=r\sin θ
```

Polar Laplacian:

```math
Δu=u_{rr}+\frac1r u_r+\frac1{r^2}u_{θθ}
```

Alternative form:

```math
Δu=\frac1r(ru_r)_r+\frac1{r^2}u_{θθ}
```

Explain the `1/r u_r` term as geometry of expanding circles.

## Method 6: Radial Harmonic Functions

If `u=u(r)`, then:

```math
u''+\frac1r u'=0
```

Multiply by `r`:

```math
(ru')'=0
```

So:

```math
u=A\log r+B
```

Use this for:

- annulus constant mode
- exterior radial barriers
- radial comparison functions

## Method 7: Disk and Annulus Solutions

### Disk

Bounded harmonic functions inside disk:

```math
u(r,θ)=a_0+\sum_{n=1}^{∞}r^n(a_n\cos nθ+b_n\sin nθ)
```

Reject singular terms at `r=0`.

### Annulus

Harmonic functions in `r_1<r<r_2`:

```math
u(r,θ)=c_0+c_1\log r+
\sum_{n=1}^{∞}
[(a_nr^n+b_nr^{-n})\cos nθ+
(c_nr^n+d_nr^{-n})\sin nθ]
```

Allow:

```math
r^{-n}
```

and:

```math
\log r
```

because `r=0` is not inside the annulus.

## Method 8: Ball and Legendre Expansion

For axisymmetric ball problems, use spherical coordinates.

If boundary data depends only on `θ`, use:

```math
u(r,θ)=\sum_{n=0}^{∞}a_nr^nL_n(\cos θ)
```

where `L_n` are Legendre polynomials.

Comparison:

```text
circle boundary → Fourier modes
axisymmetric sphere boundary → Legendre polynomials
full sphere → spherical harmonics
```

## Method 9: Reflection Principle

Use when a harmonic function has zero Dirichlet data on a straight boundary.

If:

```math
u(x,0)=0
```

then define odd reflection:

```math
U(x,y)=u(x,y),\quad y≥0
```

```math
U(x,y)=-u(x,-y),\quad y<0
```

Then `U` is harmonic across the boundary.

Important:

```text
zero boundary value is required for odd reflection.
```

If boundary value is not zero, subtract an auxiliary harmonic function first.

## Method 10: Half-Disk Strategy

For upper half-disk with general boundary data:

1. Use upper half-plane Poisson formula to build `v` matching data on the flat diameter.
2. Define:

```math
w=u-v
```

3. Then:

```math
w=0
```

on the flat diameter.
4. Oddly reflect `w` to the full disk.
5. Solve using disk Poisson kernel.
6. Recover:

```math
u=v+w
```

Memory:

```text
remove nonzero flat boundary data → reflect → solve full disk problem
```

## Method 11: Wedge and Quadrant Angle Solution

For wedge:

```math
0<θ<α
```

with boundary values:

```math
u=0 \quad \text{on } θ=0
```

and:

```math
u=1 \quad \text{on } θ=α
```

solution is:

```math
u=\frac{θ}{α}
```

For quadrant:

```math
u=\frac{2θ}{π}
```

If:

```math
θ=\arctan((x_2-1)/(x_1-1))
```

then:

```math
u=\frac{2}{π}\arctan\left(\frac{x_2-1}{x_1-1}\right)
```

Memory:

```text
constant boundary values on rays → solution is normalized angle
```

## Method 12: Kelvin Transform

Use to map circular boundaries to straight boundaries.

In 2D:

```math
T(x)=\frac{x}{|x|^2}
```

Coordinates:

```math
X_1=\frac{x_1}{x_1^2+x_2^2}
```

```math
X_2=\frac{x_2}{x_1^2+x_2^2}
```

Useful equivalence:

```math
X_1=1
```

means:

```math
\frac{x_1}{x_1^2+x_2^2}=1
```

so:

```math
x_1^2+x_2^2-x_1=0
```

Thus a circle maps to a line.

Use Kelvin transform when:

```text
hard curved domain → simple half-plane/quadrant/wedge
```

Then:

```math
v(x)=u(T(x))
```

transfers the solution back.

## Method 13: Exterior Dirichlet Uniqueness

Given two bounded solutions:

```math
u_1,u_2
```

set:

```math
w=u_1-u_2
```

Then:

```math
Δw=0
```

and:

```math
w=0
```

on the obstacle boundary.

Use radial barrier:

```math
v(ρ)=\frac{\log(ρ/r)}{\log(R/r)}
```

with:

```math
v(r)=0,\quad v(R)=1
```

Maximum principle gives:

```math
|w|≤Mv
```

Let `R→∞`:

```math
v(ρ)→0
```

so:

```math
w=0
```

Therefore solution is unique.

## Method 14: Exterior Neumann Uniqueness Up to Constants

For Neumann boundary conditions, adding a constant preserves the boundary condition.

Therefore uniqueness can only mean:

```math
u_1-u_2=constant
```

Given two bounded solutions, define:

```math
w=u_1-u_2
```

Then:

```math
Δw=0
```

and:

```math
∂_νw=0
```

Use energy method:

```math
\int_{D_R}|\nabla w|^2
=
\int_{\partial D_R}w∂_νw
```

The obstacle boundary term is zero because:

```math
∂_νw=0
```

The outer boundary term goes to zero using:

```math
|\nabla w(x)|≤C/|x|^2
```

Thus:

```math
\int_{\Omega_e}|\nabla w|^2=0
```

So:

```math
\nabla w=0
```

Therefore:

```math
w=constant
```

## Green Identity Reminder

For harmonic `w`:

```math
Δw=0
```

Green identity gives:

```math
\int_D|\nabla w|^2
=
\int_{\partial D}w∂_νw
```

This is not true for arbitrary functions without the extra domain term:

```math
-\int_D wΔw
```

The harmonic condition makes that term vanish.

## Important Distinctions

### Harmonic does not mean constant

Example:

```math
w(x,y)=x-y
```

Then:

```math
w_{xx}=0,\quad w_{yy}=0
```

so:

```math
Δw=0
```

But:

```math
\nabla w=(1,-1)
```

so it is not constant.

### Zero gradient means constant

If:

```math
\nabla w=0
```

then:

```math
w_x=0,\quad w_y=0
```

so `w` is constant.

### Zero energy means zero gradient

If:

```math
\int|\nabla w|^2=0
```

then since:

```math
|\nabla w|^2=w_x^2+w_y^2≥0
```

we get:

```math
\nabla w=0
```

## How to Tutor the User

The user often says "I don't get it", "this is difficult", or "I am lost."

When that happens:

1. Stop adding advanced theory.
2. Restate the geometry.
3. Identify the boundary pieces.
4. Explain the method in one sentence.
5. Give a tiny example.
6. Then return to the formula.

Prefer explanations like:

```text
We are not solving u directly here. We are proving two solutions cannot be different.
```

or:

```text
This method works because the domain is a wedge, and the boundary values are constant on rays.
```

## Final Memory Summary

Boundary-value problem strategy:

```text
1. Domain shape chooses coordinates.
2. Boundary type chooses method.
3. Finite/periodic direction uses Fourier series.
4. Infinite direction uses Fourier transform.
5. Circle/disk uses polar coordinates and Poisson kernel.
6. Sphere uses Legendre/spherical harmonics.
7. Zero boundary on a line allows reflection.
8. Wedge with constant side values gives normalized angle.
9. Exterior Dirichlet uniqueness uses barriers.
10. Exterior Neumann uniqueness uses energy and is only up to constants.
```
