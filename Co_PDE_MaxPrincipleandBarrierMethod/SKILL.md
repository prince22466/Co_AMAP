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

# 🎯 GOAL

Master the core PDE reasoning pattern:

[
\boxed{
\text{Build a simple function → Compare → Apply maximum principle}
}
]

You are NOT solving PDEs explicitly.
You are **controlling them**.

---

# 🧠 CORE CONCEPTS (Minimal Set)

## 1. PDE Operator

[
L u = u_t - u_{xx}
]

👉 Means: “plug (u) into the PDE”

---

## 2. Maximum Principle

[
\boxed{
\text{No new max/min inside the domain}
}
]

* Max/min come from boundary or initial condition
* Interior cannot create extremes

---

## 3. Comparison Principle (MAIN TOOL)

If:

* (Lv \ge 0)
* (v \ge 0) on boundary

Then:
[
\boxed{v \ge 0 \text{ everywhere}}
]

---

## 4. Shift Trick

[
v = u - c \quad \text{or} \quad v = u - u^s
]

👉 Purpose:

* normalize problem to “≥ 0”

---

## 5. Barrier Function

[
w = \text{simple function}
]

👉 Purpose:

* compare with (u)
* inject known behavior

---

## 6. Comparison Setup

Define:

[
v = u - w \quad \text{or} \quad v = w - u
]

Then:

* check (Lv \ge 0)
* check boundary
* apply maximum principle

---

# ⚙️ BARRIER DESIGN (SYSTEMATIC)

## Step 1 — Choose template

| Goal           | Template                |
| -------------- | ----------------------- |
| decay          | ( \phi(x)e^{-\beta t} ) |
| boundary slope | ( e^x )                 |
| simple bounds  | polynomial              |

---

## Step 2 — Apply operator

Compute:
[
Lw
]

---

## Step 3 — Tune parameters

Choose constants so:
[
Lw \ge 0 \quad \text{or} \quad Lw \le 0
]

---

## Step 4 — Match boundary

Ensure:

* (w \le u) or (w \ge u) on boundary

---

## Step 5 — Compare

Apply maximum principle

---

# 🔑 EIGENFUNCTION INSIGHT

Solve:

[
-\phi'' = \lambda \phi
]

Then:

[
w = \phi(x)e^{-\lambda t}
]

👉 Gives **natural decay**

---

## Important result:

[
\boxed{\lambda_1 = \pi^2}
]

👉 slowest decay mode

---

# 🔥 STRUCTURE OF HEAT EQUATION

[
\boxed{
u = \text{steady state} + \text{decaying modes}
}
]

---

# 📌 EXAMPLE 1 — Non-negativity

### Problem:

[
u_t - u_{xx} = 0,\quad u(x,0)\ge 0,\ u(0,t),u(1,t)\ge 0
]

---

### Solution pattern:

Define:
[
v = u
]

Check:

* (Lv = 0)
* boundary ≥ 0

Apply maximum principle:

[
\boxed{u \ge 0}
]

---

# 📌 EXAMPLE 2 — Upper bound via barrier

### Goal:

[
u(x,t) \le \alpha x(1-x)e^{-\beta t}
]

---

### Step 1:

Choose:
[
w = \alpha x(1-x)e^{-\beta t}
]

---

### Step 2:

Compute:
[
Lw = \alpha(2 - \beta x(1-x))e^{-\beta t}
]

---

### Step 3:

Use:
[
x(1-x)\le \tfrac14
]

Choose:
[
\beta \le 8
]

---

### Step 4:

Define:
[
v = w - u
]

Apply maximum principle:

[
\boxed{u \le w}
]

---

# 📌 EXAMPLE 3 — Optimal barrier (eigenvalue)

### Use:

[
w = \alpha \sin(\pi x)e^{-\pi^2 t}
]

---

### Why?

[
-\phi'' = \pi^2 \phi
]

So:

[
Lw = (\pi^2 - \beta)\phi e^{-\beta t}
]

Choose:
[
\beta = \pi^2
]

👉 gives optimal decay

---

# 📌 EXAMPLE 4 — Steady state + forcing

### PDE:

[
u_t - u_{xx} = 1
]

---

### Step 1:

Solve steady state:

[
-u'' = 1 \Rightarrow u^s = \tfrac12 x(1-x)
]

---

### Step 2:

Define:
[
v = u - u^s
]

Then:
[
v_t - v_{xx} = 0
]

---

### Step 3:

Apply max principle:

[
u \to u^s
]

---

# 📌 EXAMPLE 5 — Hopf Lemma (boundary slope)

### Setup:

* minimum at boundary
* interior values larger

---

### Barrier:

[
w = e^x - 1
]

Properties:

* (w(0)=0)
* (w'(0)>0)
* (Lw < 0)

---

### Compare:

[
v \ge w \Rightarrow v'(0) \ge w'(0) > 0
]

---

### Result:

[
\boxed{u_x > 0}
]

---

# 🧪 INTERACTIVE DRILLS

## Drill 1 — Pattern recognition

Given:
[
u_t - u_{xx} = 0
]

Task:

* define (v)
* check boundary
* apply maximum principle

---

## Drill 2 — Build a barrier

Goal:
[
u \le Ce^{-t}
]

Try:
[
w = Ae^{-\beta t}
]

👉 compute (Lw)

---

## Drill 3 — Eigenfunction

Solve:
[
-\phi'' = \lambda \phi,\quad \phi(0)=\phi(1)=0
]

👉 find:

* (\phi_1)
* (\lambda_1)

---

## Drill 4 — Hopf intuition

Sketch:

* boundary minimum
* interior larger

👉 predict sign of derivative

---

# 🧠 FINAL MEMORY

## Keep ONLY this:

[
\boxed{
\text{Shift → Barrier → Compare → Max principle}
}
]

---

## And this:

[
\boxed{
\text{We control PDEs, not solve them}
}
]


