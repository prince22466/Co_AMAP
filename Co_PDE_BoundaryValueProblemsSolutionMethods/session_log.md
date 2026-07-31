# PDE Boundary-Value Problems — Session Log

## Session 1 — 2026-07-30

### Main method studied

Separation of variables for Laplace's equation on the unit square:

\[
\Omega=(0,1)\times(0,1).
\]

Core workflow used:

1. Identify the PDE.
2. Identify the domain and its four boundary pieces.
3. Classify the boundary conditions.
4. Choose Cartesian coordinates and separation of variables.
5. Set \(u(x,y)=X(x)Y(y)\).
6. Solve the resulting ODEs.
7. Apply and check all boundary conditions.

### Concepts understood

- \(\Delta u=u_{xx}+u_{yy}=0\) is Laplace's equation.
- Prescribing the value of \(u\) gives Dirichlet boundary conditions.
- For \(u=XY\):

  \[
  u_{xx}=X''Y,\qquad u_{yy}=XY''.
  \]

- Separation gives:

  \[
  X''+\lambda X=0,\qquad Y''-\lambda Y=0.
  \]

- The right boundary \(u(1,y)=0\) implies \(X(1)=0\) for a nontrivial separated solution.
- The ODE

  \[
  X''+\mu^2X=0
  \]

  has the general solution

  \[
  X=A\cos(\mu x)+B\sin(\mu x).
  \]

- Conditions \(X(0)=X(1)=0\) give the modes

  \[
  X_n(x)=\sin(n\pi x).
  \]

- The ODE with the opposite sign,

  \[
  Y''-\mu^2Y=0,
  \]

  uses exponentials or \(\sinh\) and \(\cosh\).

### Exercise 1 — completed

Problem:

\[
\Delta u=0
\]

in the unit square, with zero values on the left, right, and bottom, and

\[
u(x,1)=\sin(2\pi x).
\]

Solution:

\[
\boxed{
u(x,y)=
\frac{\sinh(2\pi y)}{\sinh(2\pi)}
\sin(2\pi x)
}
\]

The PDE and all four boundary conditions were checked.

### Exercise 2 — completed

Top boundary:

\[
u(x,1)=2\sin(\pi x)+3\sin(2\pi x).
\]

Using linearity, the solution is

\[
\boxed{
u(x,y)=
2\frac{\sinh(\pi y)}{\sinh(\pi)}\sin(\pi x)
+
3\frac{\sinh(2\pi y)}{\sinh(2\pi)}\sin(2\pi x)
}
\]

Key reminder: each Fourier coefficient stays attached to its corresponding mode.

### Exercise 3 — in progress

Top boundary:

\[
u(x,1)=x(1-x).
\]

The boundary function must be expanded as a Fourier sine series:

\[
x(1-x)=\sum_{n=1}^{\infty}b_n\sin(n\pi x),
\]

where

\[
b_n=2\int_0^1x(1-x)\sin(n\pi x)\,dx.
\]

The factor \(2\) was derived using sine orthogonality:

\[
\int_0^1\sin^2(n\pi x)\,dx=\frac12.
\]

For the first coefficient, we computed

\[
\boxed{b_1=\frac{8}{\pi^3}}.
\]

Integration reminders covered:

\[
\int\sin(\pi x)\,dx=-\frac{\cos(\pi x)}{\pi},
\]

\[
\int\cos(\pi x)\,dx=\frac{\sin(\pi x)}{\pi}.
\]

### Resume point

Continue Exercise 3 by finding the general coefficient \(b_n\), identifying which coefficients vanish, and inserting them into

\[
u(x,y)=
\sum_{n=1}^{\infty}
b_n
\frac{\sinh(n\pi y)}{\sinh(n\pi)}
\sin(n\pi x).
\]

### Memory box

\[
\boxed{
\text{Rectangle}
\longrightarrow
\text{separation of variables}
\longrightarrow
\text{Fourier sine modes}
}
\]

For top data \(g(x)=\sum b_n\sin(n\pi x)\) and zero data on the other three sides:

\[
\boxed{
u(x,y)=
\sum_{n=1}^{\infty}
b_n
\frac{\sinh(n\pi y)}{\sinh(n\pi)}
\sin(n\pi x)
}
\]

## Session 2 — 2026-07-31

### Exercise 3 — completed

For the top boundary data

\[
u(x,1)=x(1-x),
\]

the Fourier sine coefficients were computed:

\[
b_n=2\int_0^1x(1-x)\sin(n\pi x)\,dx
=\frac{4(1-(-1)^n)}{n^3\pi^3}.
\]

Therefore,

\[
b_n=
\begin{cases}
\dfrac{8}{n^3\pi^3},&n\text{ odd},\\[4pt]
0,&n\text{ even}.
\end{cases}
\]

The solution is

\[
\boxed{
u(x,y)=
\sum_{\substack{n=1\\n\text{ odd}}}^{\infty}
\frac{8}{n^3\pi^3}
\frac{\sinh(n\pi y)}{\sinh(n\pi)}
\sin(n\pi x)
}.
\]

### Laplace equation with two nonzero boundary sides

For

\[
u(x,1)=\sin(\pi x),\qquad
u(1,y)=\sin(\pi y),
\]

and zero data on the left and bottom, the problem was split as

\[
u=u_1+u_2.
\]

The two parts are

\[
u_1(x,y)=
\frac{\sinh(\pi y)}{\sinh(\pi)}\sin(\pi x),
\]

\[
u_2(x,y)=
\frac{\sinh(\pi x)}{\sinh(\pi)}\sin(\pi y).
\]

Thus,

\[
\boxed{
u(x,y)=
\frac{\sinh(\pi y)}{\sinh(\pi)}\sin(\pi x)
+
\frac{\sinh(\pi x)}{\sinh(\pi)}\sin(\pi y)
}.
\]

The origin of the normalized hyperbolic factor was derived:

\[
Y''-\pi^2Y=0,\qquad Y(0)=0,\qquad Y(1)=1,
\]

which gives

\[
Y(y)=\frac{\sinh(\pi y)}{\sinh(\pi)}.
\]

The PDE check was also completed:

\[
(u_1)_{xx}=-\pi^2u_1,\qquad
(u_1)_{yy}=\pi^2u_1,
\]

so

\[
\Delta u_1=0,
\]

and similarly \(\Delta u_2=0\).

### Boundary-side variable rule

For data on the top boundary:

\[
\sin(n\pi x)
\]

describes the boundary shape, while

\[
\frac{\sinh(n\pi y)}{\sinh(n\pi)}
\]

moves the solution from zero at the bottom to the prescribed value at the top.

For data on the right boundary, the variables exchange roles:

\[
\frac{\sinh(n\pi x)}{\sinh(n\pi)}
\sin(n\pi y).
\]

Important correction:

```text
The side carrying the nonzero data determines which variable belongs
to the normalized hyperbolic factor. The frequency does not cause the
variables to alternate.
```

### Poisson equation with zero boundary data

The new PDE type was introduced:

\[
\Delta u=f(x,y).
\]

For

\[
\Delta u=\sin(\pi x)\sin(\pi y),
\qquad u=0\text{ on all four sides},
\]

the trial function

\[
u=A\sin(\pi x)\sin(\pi y)
\]

gave

\[
\Delta u=-2A\pi^2\sin(\pi x)\sin(\pi y).
\]

Therefore,

\[
\boxed{
u(x,y)=
-\frac{1}{2\pi^2}\sin(\pi x)\sin(\pi y)
}.
\]

For the general single mode,

\[
\Delta u=\sin(m\pi x)\sin(n\pi y),
\]

the solution is

\[
\boxed{
u(x,y)=
-\frac{1}{(m^2+n^2)\pi^2}
\sin(m\pi x)\sin(n\pi y)
}.
\]

The squares \(m^2+n^2\) arise because the sine functions are
differentiated twice.

### Multiple source modes

For

\[
\Delta u=
2\sin(\pi x)\sin(\pi y)
+
3\sin(2\pi x)\sin(\pi y),
\]

with zero boundary data, linearity gives

\[
\boxed{
u(x,y)=
-\frac{1}{\pi^2}\sin(\pi x)\sin(\pi y)
-
\frac{3}{5\pi^2}\sin(2\pi x)\sin(\pi y)
}.
\]

### General double sine series

For

\[
f(x,y)=
\sum_{m=1}^{\infty}\sum_{n=1}^{\infty}
f_{mn}\sin(m\pi x)\sin(n\pi y),
\]

the coefficients are

\[
f_{mn}
=
4\int_0^1\int_0^1
f(x,y)\sin(m\pi x)\sin(n\pi y)\,dx\,dy.
\]

The factor \(4\) is \(2\cdot2\), one sine-series normalization factor
from each coordinate.

The zero-boundary Poisson solution is

\[
\boxed{
u(x,y)=
-\sum_{m=1}^{\infty}\sum_{n=1}^{\infty}
\frac{f_{mn}}{(m^2+n^2)\pi^2}
\sin(m\pi x)\sin(n\pi y)
}.
\]

### Poisson equation with nonzero boundary data

For

\[
\Delta u=f\quad\text{in }\Omega,\qquad
u=g\quad\text{on }\partial\Omega,
\]

the problem was split as

\[
u=v+w,
\]

where

\[
\Delta v=f,\qquad v=0\text{ on }\partial\Omega,
\]

and

\[
\Delta w=0,\qquad w=g\text{ on }\partial\Omega.
\]

Concrete example:

\[
\Delta u=\sin(\pi x)\sin(\pi y),
\qquad
u(x,1)=\sin(2\pi x),
\]

with zero values on the other three sides. The two parts are

\[
v(x,y)=
-\frac{1}{2\pi^2}\sin(\pi x)\sin(\pi y),
\]

\[
w(x,y)=
\frac{\sinh(2\pi y)}{\sinh(2\pi)}
\sin(2\pi x).
\]

Hence,

\[
\boxed{
u(x,y)=
-\frac{1}{2\pi^2}\sin(\pi x)\sin(\pi y)
+
\frac{\sinh(2\pi y)}{\sinh(2\pi)}
\sin(2\pi x)
}.
\]

Important correction:

```text
Matching the boundary values is not enough. A proposed function must
also satisfy the PDE inside the domain. For example,
y sin(2 pi x) matches the relevant boundary values but is not harmonic.
```

### Tutoring calibration

Continue interactively, but ask questions that test PDE reasoning,
method selection, boundary checks, and coefficient derivations. Avoid
questions that only test elementary arithmetic.

### Resume point

The rectangle method has been completed through Poisson problems with
sources and nonzero boundary data.

The next topic is Laplace's equation in the unit disk:

\[
x^2+y^2<1.
\]

Resume by asking which coordinates fit the disk geometry, then derive
the polar-coordinate Laplacian and begin separation of variables in
\((r,\theta)\).

### Session 2 memory box

\[
\boxed{
\text{Poisson with nonzero boundary data}
=
\text{zero-boundary source problem}
+
\text{harmonic boundary problem}
}
\]

\[
\boxed{
\text{Disk geometry}
\longrightarrow
\text{polar coordinates}
}
\]
