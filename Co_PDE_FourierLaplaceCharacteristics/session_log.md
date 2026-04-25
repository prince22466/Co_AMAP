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
