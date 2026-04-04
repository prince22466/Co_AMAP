# PDE Error Log

## Summary (auto-updated)
- classification errors: 2
- method errors: 2
- BC/IC errors: 5
- algebra errors: 0
- logic errors: 5
- verification missed: 0

---

## Entries

### 2026-04-04 Day 1 Problem 1
- Error type(s): classification, method choice, BC/IC handling
- Concise explanation: The PDE type and domain were left unidentified. The initial and boundary conditions were swapped. Finite differences is a numerical method, but this exercise structure points to an exact analytic method.
- Corrected reasoning: `u_t = k u_xx` is a second-order linear parabolic PDE on the space-time domain `0 < x < L`, `t > 0`. The boundary conditions are `u(0,t)=0` and `u(L,t)=0`; the initial condition is `u(x,0)=f(x)`. Because the PDE is linear with homogeneous spatial boundary conditions on a finite interval, separation of variables is the standard first method.
- Fix rule: Identify PDE class and domain first; then separate BC from IC before choosing an analytic method.

### 2026-04-04 Day 1 Problem 1 Retry 2
- Error type(s): classification, BC/IC handling
- Concise explanation: The PDE order was given as 1 instead of 2, and the boundary and initial conditions were still mixed up.
- Corrected reasoning: The highest derivative present is `u_xx`, so the PDE is second order. The domain is `0 < x < L`, `t > 0`. Both `u(0,t)=0` and `u(L,t)=0` are boundary conditions because they are imposed at spatial endpoints. The condition `u(x,0)=f(x)` is the initial condition because it is imposed at initial time.
- Fix rule: Determine order from the highest derivative, then classify BC vs IC by whether the condition is on space boundaries or at `t=0`.

### 2026-04-04 Day 1 Problem 1 Setup
- Error type(s): logic
- Concise explanation: No product ansatz was attempted after choosing separation of variables.
- Corrected reasoning: Once separation of variables is chosen, the first move is to assume `u(x,t)=X(x)T(t)`, substitute into the PDE, and divide by `kXT` or `XT` to isolate a purely `t`-dependent term and a purely `x`-dependent term.
- Fix rule: After selecting separation of variables, immediately write a product ansatz before doing any further reasoning.

### 2026-04-04 Day 1 Problem 1 Eigenstep
- Error type(s): logic
- Concise explanation: The eigenvalue problem `X'' + lambda X = 0` with `X(0)=X(L)=0` was not carried through to the standard sine modes.
- Corrected reasoning: Imposing both endpoint conditions forces discrete eigenvalues `lambda_n = (n pi / L)^2` and eigenfunctions `X_n(x) = sin(n pi x / L)`, `n=1,2,...`. Then `T' + k lambda_n T = 0` gives `T_n(t) = exp(-k lambda_n t)`.
- Fix rule: For `X'' + lambda X = 0` on `(0,L)` with homogeneous Dirichlet BCs, recall the standard sine spectrum immediately.

### 2026-04-04 Day 1 Problem 2 Scan
- Error type(s): BC/IC handling
- Concise explanation: The Neumann boundary conditions were not identified correctly. `u_x(0,t)=0` and `u_x(L,t)=0` are zero-flux boundary conditions, not arbitrary constants.
- Corrected reasoning: The domain is `0 < x < L`, `t > 0`. The boundary conditions are homogeneous Neumann conditions `u_x(0,t)=0`, `u_x(L,t)=0`. The initial condition is `u(x,0)=f(x)`.
- Fix rule: Read the exact boundary operator carefully: `u=0` and `u_x=0` lead to different eigenfunctions and must not be blurred together.

### 2026-04-04 Day 1 Problem 2 Setup
- Error type(s): BC/IC handling, logic
- Concise explanation: The product ansatz was started, but the substitution and separated form were omitted, and the boundary conditions were rewritten as values of `u` instead of derivatives.
- Corrected reasoning: With `u(x,t)=X(x)T(t)`, substitution into `u_t = k u_xx` gives `X T' = k X'' T`, hence `T'/(kT) = X''/X = -lambda`. Since `u_x = X' T`, the Neumann conditions become `X'(0)=0` and `X'(L)=0`.
- Fix rule: When the PDE BC is on `u_x`, translate it immediately into BCs on `X'` after writing the product ansatz.

### 2026-04-04 Day 1 Problem 2 Eigenstep
- Error type(s): logic
- Concise explanation: The Neumann eigenvalue problem was not carried through, including the zero mode.
- Corrected reasoning: For `X'' + lambda X = 0` with `X'(0)=X'(L)=0`, there is a zero mode `lambda_0=0`, `X_0(x)=1`. For `n>=1`, the eigenvalues are `lambda_n = (n pi / L)^2` and eigenfunctions are `X_n(x)=cos(n pi x / L)`. The time factors are `T_n(t)=exp(-k lambda_n t)` for `n>=1`, while the zero mode is time-independent.
- Fix rule: For homogeneous Neumann BCs on `(0,L)`, recall cosine modes plus the constant zero mode.

### 2026-04-04 Day 1 Problem 2 Coefficients
- Error type(s): BC/IC handling
- Concise explanation: The coefficients were said to be determined from the boundary conditions, but they are determined from the initial condition.
- Corrected reasoning: The Neumann boundary conditions are already encoded in the cosine basis. The coefficients `a_0, a_n` are found by matching `u(x,0)=f(x)`, i.e. expanding `f` in a Fourier cosine series.
- Fix rule: Boundary conditions choose the eigenbasis; the initial condition chooses the coefficients.

### 2026-04-04 Day 1 Problem 3 Scan
- Error type(s): classification, method choice, logic
- Concise explanation: The equation was classified as diffusion and solved by separation of variables, but it is a first-order transport equation.
- Corrected reasoning: `u_t + c u_x = 0` is a first-order linear PDE. With the initial condition `u(x,0)=g(x)`, the natural domain is typically `x in R, t > 0` unless a bounded spatial interval is specified. The correct method is characteristics, because the PDE transports data along straight lines in the `(x,t)` plane.
- Fix rule: If the PDE has the form `u_t + a u_x = 0`, think transport and choose characteristics first, not separation of variables.

### 2026-04-04 Day 1 Problem 3 Characteristics
- Error type(s): logic
- Concise explanation: The characteristic invariant and final solution were identified incorrectly. The line geometry was not translated into transported initial data.
- Corrected reasoning: From `dx/ds=c` and `dt/ds=1`, the characteristics satisfy `dx/dt=c`, so `x-ct = const`. Since `du/ds=0`, the solution is constant along each characteristic, hence `u(x,t)=g(x-ct)`.
- Fix rule: For transport equations, first find the characteristic invariant such as `x-at`, then compose the initial data with that invariant.
