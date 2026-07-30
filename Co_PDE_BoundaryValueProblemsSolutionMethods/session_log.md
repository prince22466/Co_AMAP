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
