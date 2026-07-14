# Session Log

## 2026-04-16

- Topics covered: heat equation intuition, heat kernel meaning/formula, Fourier-to-ODE step, scaling `sqrt(Dt)`, boundary reflection, Duhamel principle, near/far split.
- Completion criteria: explain kernel, decay, reflection, Duhamel, near/far split.
- Struggle points:
  - Reflection setup for half-line Dirichlet.
  - Duhamel interpretation in time-slice form.
  - Near/far split proof structure.
  - Fourier-to-ODE step (`\mathcal{F}_x[u_t] = \partial_t \hat{u}` and `\mathcal{F}_x[u_{xx}] = -k^2\hat{u}`).
- Next session plan:
  - Numeric near/far split estimate.
  - Half-line Dirichlet construction drill.
  - One worked Duhamel example.

## 2026-07-14

- Resumed with a numeric near/far split estimate.
- Reviewed the heat kernel as Gaussian weights for averaging nearby temperatures.
- Built intuition for narrow/tall (small time) versus wide/flat (large time), while preserving total kernel mass `1`.
- Connected Gaussian decay to why distant points have negligible influence at small time.
- Confirmed that constant initial temperature remains unchanged under kernel averaging.
- Practiced diffusion scaling: characteristic spreading distance is `sqrt(Dt)`; multiplying time by `4` multiplies the distance by `2`.
- User can now explain in words:
  - the kernel acts as weights for averaging heat around an observation point;
  - nearby points matter more at small time;
  - distant points become more relevant as time increases.
- Stopped before answering the scaling check: if `D` is multiplied by `9`, determine the factor increase in `sqrt(Dt)`.
- Next session plan:
  - Finish the diffusion-coefficient scaling check.
  - Briefly consolidate kernel and Gaussian decay.
  - Continue with the half-line Dirichlet reflection drill.
