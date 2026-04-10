---
name: pde-training
description: Use this skill to teach PDE concepts step-by-step when the user asks to learn, study, or train PDE.
---

# 🎯 When to use this skill

Use this skill if the user says:
- "start training"
- "learn PDE"
- "teach me PDE"
- or shows confusion about PDE concepts

---

## Objective
Master:
- Heat equation intuition
- Heat kernel (Gaussian)
- Boundary handling (reflection)
- Duhamel principle (forcing)
- Convergence as t → 0

---

# 1. Core PDE

u_t - D u_xx = 0

### Meaning
- diffusion = averaging
- sharp → smooth
- local → spreads over time

---

# 2. Heat Kernel

Γ(x-y,t) = (1 / sqrt(4πDt)) * exp(-(x-y)^2 / (4Dt))

### Interpretation
- point source at y spreads to x
- Gaussian = probability density

### Key Properties
- ∫ Γ dy = 1
- symmetric
- concentrates as t → 0
- width ~ √t

---

# 3. Solution (Whole Line)

u(x,t) = ∫ Γ(x-y,t) g(y) dy

### Meaning
solution = weighted average of initial data

---

# 4. Small-Time Behavior

Γ → δ (delta function)

⇒ u(x,t) → g(x)

---

## Proof Strategy (Template)

Split domain:
- near region → use continuity
- far region → use Gaussian decay

---

# 5. Gaussian Decay (CRITICAL)

exp(-(x-y)^2 / (4Dt))

### Insight
- |x - y| large → exponential ≈ 0
- t small → decay stronger

### Scaling
effective distance ~ √t

---

# 6. Domain Types

## Whole line
(-∞, ∞)

## Half-line
[0, ∞)

## Interval
[0, L]

---

# 7. Boundary Conditions

## Dirichlet
u = 0 → absorbing

## Neumann
u_x = 0 → reflecting

---

# 8. Reflection Method

### Idea
extend to whole line using symmetry

### Rules
- Dirichlet → odd extension
- Neumann → even extension

---

## Kernel Construction

Half-line:
Γ(x-y,t) ± Γ(x+y,t)

Interval:
sum over reflections:
Γ(x - y - 2nL) ± Γ(x + y - 2nL)

---

# 9. Duhamel Principle

u_t - D u_xx = f(x,t)

### Idea
solution = sum of time-shifted heat responses

### Formula
u(x,t) = ∫₀ᵗ ∫ N(x,y,t-τ) f(y,τ) dy dτ

### Interpretation
At time τ:
- inject heat
- diffuse until t

---

# 10. Superposition

linear PDE → solutions add

---

# 11. Probability Interpretation

Γ(x-y,t) = density of Brownian motion

⇒ u(x,t) = expected value of g

---

# 12. Master Pattern

initial / forcing
→ weighted by kernel
→ integrated
→ solution

---

# 13. Problem-Solving Template

1. Identify domain
2. Identify boundary
3. Choose method:
   - whole line → kernel
   - half-line → reflection
   - interval → reflections / Fourier
4. If forcing → Duhamel

---

# 14. Key Techniques

- near/far split
- continuity locally
- exponential decay globally
- change of variables (√t scaling)

---

# 15. Minimal Recall

- Heat = smoothing
- Kernel = Gaussian
- Small time = localization
- Boundary = reflection
- Forcing = time integration

---

# 16. Practice Tasks

1. Show u(x,t) → g(x)
2. Build half-line solution (Dirichlet)
3. Explain Gaussian decay
4. Explain Duhamel without formula

---

# 17. Completion Criteria

You can:
- explain kernel intuitively
- explain decay
- explain reflection
- explain Duhamel
- explain near/far split

---

# 🔚 One-Line Summary

Heat equation = Gaussian-weighted averaging evolving over space and time with boundary effects handled by symmetry.
