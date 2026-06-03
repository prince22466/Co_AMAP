# PDE Training Session Log

## Date
2026-04-25

## Scope Covered
- Started PDE training from zero following `AGENT.md` and `SKILL.md`.
- Focused on intuition first, with short interactive checks after each concept.
- Continued training with the method of characteristics and compared it against Fourier-based methods.

## Concepts Completed
- Meaning of `u(x,t)` as a quantity depending on space `x` and time `t`.
- Meaning of `u_t` as change in time.
- Meaning of `u_x` as change in space.
- Meaning of `u_{xx}` as change of slope in space, interpreted as curvature.
- Heat equation intuition: `u_t = u_{xx}` means time evolution is driven by spatial curvature.
- If the profile is linear in `x`, then `u_{xx} = 0`, so the profile does not change in time under the heat equation.

## Fourier Method
- Main idea: break a complicated profile into simple wave modes.
- Key rule: spatial differentiation becomes multiplication in Fourier space.
- Interpretation of "diagonalizing the PDE":
  each Fourier mode with frequency `k` evolves independently.
- Heat equation in Fourier space:
  `û_t = -k^2 û`
- This is easier because the PDE becomes separate ODEs in time, one for each `k`.

## Smoothing / Decay
- Solution factor in Fourier space:
  `e^{-k^2 t}`
- High frequencies (large `k`) decay faster.
- Physical meaning:
  rough or sharp features disappear first, so heat flow smooths the profile.

## Convolution / Heat Kernel
- Later solution is a weighted average of nearby initial values.
- Initial data means the profile at time `t = 0`.
- Physical interpretation:
  heat moves and mixes neighboring spatial points over time.
- Connection to PDE form:
  because `u_{xx}` depends on nearby values, the time change at one point is influenced by nearby points.

## Boundaries and Reflection
- Boundaries refer to constraints on the spatial domain, such as `x > 0` or `0 < x < L`.
- Ordinary modes like `e^{ikx}` do not automatically satisfy boundary conditions like `u(0,t) = 0`.
- Odd reflection:
  mirror the right side and reverse the sign.
- Used naturally for boundary condition:
  `u(0,t) = 0`
- Even reflection:
  mirror without sign change.
- Used naturally for boundary condition:
  `u_x(0,t) = 0`

## User Understanding Reached
- Fourier method simplifies the heat PDE into ODEs for each mode.
- High-frequency waves decay fast.
- Solution can be understood as a weighted average of initial data.
- Odd/even reflections are used to enforce boundary conditions.

## Characteristics Method
- Transport equation intuition:
  `u_t + u_x = 0` moves the profile without smoothing it.
- Difference between first and second spatial derivatives:
  `u_x` indicates transport/shift, while `u_{xx}` indicates curvature and diffusion.
- Chain rule along a path:
  for `u(x(t), t)`,
  `d/dt u(x(t), t) = u_t + x'(t) u_x`
- For `u_t + u_x = 0`, choose `x'(t) = 1`, giving characteristics
  `x - t = C`
- General transport solution with initial data `u(x,0)=g(x)`:
  `u(x,t) = g(x-t)`
- Information interpretation:
  the value at `(x,t)` comes from the initial point `(x-t,0)`.

## Variable-Speed Transport
- For `u_t + a u_x = 0`, the profile shifts with speed `a`:
  `u(x,t) = g(x-at)`
- Sign of `a` determines direction:
  `a > 0` moves right, `a < 0` moves left.
- For `u_t + x u_x = 0`, the speed depends on position, so the profile distorts.
- Characteristics satisfy
  `x'(t) = x`
  so
  `x(t) = C e^t`
  and
  `x e^{-t} = C`
- Corresponding solution with initial data:
  `u(x,t) = g(x e^{-t})`
- Physical interpretation:
  points move away from the origin exponentially, so the profile stretches outward.

## Source Terms Along Characteristics
- For `u_t + u_x = 1`, the characteristics are still `x - t = C`, but now
  `du/dt = 1`
- Solution with initial data:
  `u(x,t) = g(x-t) + t`
- For `u_t + x u_x = 2`, the characteristics come from `x'(t)=x`, while along them
  `du/dt = 2`
- Solution with initial data:
  `u(x,t) = g(x e^{-t}) + 2t`
- Key split reinforced:
  transport determines where the value came from;
  source term determines how the value changes while traveling.

## Method Choice Comparison
- Fourier is natural for smoothing/diffusion problems like
  `u_t = u_{xx}`
- Characteristics are natural for transport problems like
  `u_t + a(x,t) u_x = b`
- Compact rule retained:
  modes -> Fourier
  paths -> characteristics

## Remaining Skill Topics
- `SKILL.md` topics now substantially covered:
  Fourier basics, heat equation intuition, boundary/reflection intuition, and characteristics.
- Still not fully covered as worked methods:
  sine transform and method of images.

## Stop Point
- We finished the core introduction to characteristics and compared characteristics versus Fourier.
- The next missing major block from `SKILL.md` is Level 3 boundary-solving methods.
- Recommended entry problem for next session:
  half-line heat equation with boundary condition `u(0,t) = 0`
- Intended next intuition:
  enforce boundaries by reflection, then connect that to the method of images and sine transform.

## Notes For Next Session
- Start with a worked method-of-images example for the heat equation on `x > 0`.
- Show why odd reflection enforces Dirichlet boundary data `u(0,t)=0`.
- Then connect the same boundary problem to the sine transform viewpoint.
- Keep the same teaching style:
  intuition first, short check questions, then formulas.

## Session Progress - 2026-06-03

## Half-Line Heat Equation With Boundary
- Continued from the planned topic:
  `u_t = u_{xx}` on `x > 0`.
- Main boundary cases:
  - Dirichlet: `u(0,t)=0`
  - Neumann: `u_x(0,t)=0`
- Reinforced method-choice rule:
  - whole line, no boundary -> Fourier transform
  - half-line, zero value boundary -> sine transform
  - half-line, zero slope boundary -> cosine transform

## Odd Reflection, Images, and Sine Transform
- For `u(0,t)=0`, use odd reflection:
  `G(-x)=-G(x)`.
- Reason:
  odd functions satisfy `u(-x,t)=-u(x,t)`, so at `x=0`,
  `u(0,t)=-u(0,t)`, hence `u(0,t)=0`.
- Method of images form for Dirichlet boundary:
  `K(x-y,t)-K(x+y,t)`.
- Interpretation:
  real source minus negative mirror source.
- The negative mirror cancels values at the boundary.
- Sine transform viewpoint:
  sine modes satisfy `sin(0)=0`, so the boundary condition is automatic.
- Connection understood:
  odd reflection in physical space corresponds to sine transform in frequency space.

## Convolution Intuition
- Convolution was clarified as spreading/weighted averaging.
- Example understanding reached:
  `u(5,t)` is mainly influenced by initial values near `5`, with smaller influence from farther points such as `7`.
- Heat kernel `K(x-y,t)` gives spatial spreading weights over time.
- Frequency coefficients like `B(xi,0)` describe initial frequency content, not spatial averaging.

## Sine Transform Worked Form
- For
  `u_t=u_{xx}`, `x>0`, `u(0,t)=0`, `u(x,0)=g(x)`,
  use
  `u(x,t)=int_0^infty B(xi,t) sin(xi x) dxi`.
- Second derivative rule:
  `d^2/dx^2 sin(xi x) = -xi^2 sin(xi x)`.
- Therefore each coefficient solves:
  `B_t(xi,t) = -xi^2 B(xi,t)`.
- Solution:
  `B(xi,t)=B(xi,0)e^{-xi^2 t}`.
- Key interpretation:
  - `B(xi,0)` = initial amount of sine mode `xi`
  - `e^{-xi^2 t}` = decay of that mode over time
  - `sin(xi x)` = spatial wave shape
- High frequencies decay quickly because `xi^2` is large.

## Example Covered
- Example problem:
  `u_t=u_{xx}`, `x>0`, `u(0,t)=0`, `u(x,0)=e^{-x}`.
- Sine coefficient using one common normalization:
  `B(xi,0) = (2/pi) int_0^infty e^{-x} sin(xi x) dx`
  `= (2/pi) xi/(1+xi^2)`.
- Full solution:
  `u(x,t)=int_0^infty (2/pi) xi/(1+xi^2) e^{-xi^2 t} sin(xi x) dxi`.
- Checked:
  - boundary: sine modes vanish at `x=0`
  - PDE: both time derivative and second spatial derivative produce multiplier `-xi^2`
  - initial condition: recovered when `t=0`

## Even Reflection, Images, and Cosine Transform
- For `u_x(0,t)=0`, use even reflection:
  `G(-x)=G(x)`.
- Reason:
  even symmetry makes the graph flat at the center, so the slope at `x=0` is zero.
- Cosine modes satisfy:
  `d/dx cos(xi x)|_{x=0} = -xi sin(0)=0`.
- Method of images form for Neumann boundary:
  `K(x-y,t)+K(x+y,t)`.
- Interpretation:
  real source plus positive mirror source.
- Positive mirror cancels slopes at the boundary.

## Boundary Method Summary Learned
- Dirichlet:
  `u(0,t)=0` -> odd reflection -> negative image -> sine transform.
- Neumann:
  `u_x(0,t)=0` -> even reflection -> positive image -> cosine transform.
- User could distinguish:
  zero value boundary versus zero slope boundary.
- User correctly identified:
  - Fourier for whole-line heat equation
  - sine transform for half-line zero value boundary
  - cosine transform for half-line zero slope boundary

## Stop Point - 2026-06-03
- Began transition to bounded intervals `0<x<L`.
- Key first idea introduced:
  on a finite interval, allowed waves must fit between boundaries.
- Example:
  for `u(0,t)=0` and `u(L,t)=0`, use
  `sin(n pi x / L)`.
- New concept just introduced:
  finite intervals use discrete frequencies
  `xi_n = n pi / L`, not continuous `xi`.

## Notes For Next Session - 2026-06-03
- Resume with bounded interval heat equation:
  `u_t=u_{xx}`, `0<x<L`.
- Start by contrasting:
  half-line continuous frequencies `xi`
  versus bounded interval discrete frequencies `n pi/L`.
- Recommended next problem:
  `u_t=u_{xx}`, `0<x<L`,
  `u(0,t)=0`, `u(L,t)=0`, `u(x,0)=g(x)`.
- Explain why sine modes must fit both endpoints.
- Then derive:
  `u(x,t)=sum_{n=1}^infty b_n e^{-(n pi/L)^2 t} sin(n pi x/L)`.
