# SKILL.md — Quantum Gates & Circuits Teaching Skill

## Skill Name
Quantum Gates and Circuits Beginner Tutor

## Purpose
Teach and test the learner on the specific quantum computing knowledge covered in the chat: ket notation, vectors, amplitudes, probabilities, single-qubit gates, controlled gates, simple circuits, measurement, and post-measurement states.

This skill is optimized for Codex to act as a tutor and examiner, using Markdown exercises and optional `session_log.md` updates.

## When to Use This Skill
Use this skill when the learner asks to:

- Learn quantum gates or quantum circuits.
- Understand ket notation such as `|0>`, `|1>`, `|+>`, `|000>`, `|111>`.
- Calculate the output of a quantum circuit.
- Calculate measurement probabilities.
- Practice quiz questions on `X`, `Z`, `H`, `CNOT/CX`, `CZ`, or measurement.
- Review mistakes from prior questions in this topic.

## Required Starting Assumptions
Use these conventions unless the prompt explicitly says otherwise:

```text
2-qubit state: |q1 q0>
3-qubit state: |q2 q1 q0>
4-qubit state: |q3 q2 q1 q0>
```

For circuit diagrams, the top wire is normally the first bit in the ket.

Standard basis order for 3 qubits:

```text
|000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>
```

## Core Knowledge to Teach

### 1. Ket notation and vectors

One qubit:

```text
|0> = [1, 0]^T
|1> = [0, 1]^T
```

Three qubits have 8 basis states:

```text
|000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>
```

Examples:

```text
|000> = [1,0,0,0,0,0,0,0]^T
|010> = [0,0,1,0,0,0,0,0]^T
|111> = [0,0,0,0,0,0,0,1]^T
```

Rule: convert the binary label to decimal index, then add 1 for human vector position.

```text
010_2 = 2 -> third position
111_2 = 7 -> eighth position
```

### 2. Amplitudes and probabilities

A term like:

```text
0.6|000>
```

means amplitude `0.6` on basis state `|000>`.

Probability:

```text
|0.6|^2 = 0.36
```

For complex amplitude:

```text
alpha = a + bi
|alpha|^2 = a^2 + b^2
```

For a valid normalized state:

```text
sum of all probabilities = 1
```

Example:

```text
0.6|0> + 0.8|1>
P(0)=0.36
P(1)=0.64
```

### 3. Plus and minus states

```text
|+> = (|0> + |1>)/sqrt(2)
|-> = (|0> - |1>)/sqrt(2)
```

The `+` or `-` inside the ket is a state label, not a classical bit.

Both produce 50/50 measurement outcomes in the computational basis, but they have different phases and behave differently under gates.

### 4. X gate

```text
X|0> = |1>
X|1> = |0>
```

`X_i` applies X to qubit `i` only.

Using `|q2 q1 q0>`:

```text
X_2|000> = |100>
X_1|000> = |010>
X_0|000> = |001>
X_2|111> = |011>
X_1|111> = |101>
X_0|111> = |110>
```

### 5. Z gate

```text
Z|0> = |0>
Z|1> = -|1>
```

`Z` is a phase flip, not a bit flip.

On plus/minus states:

```text
Z|+> = |->
Z|-> = |+>
```

For a GHZ-like state:

```text
Z_i(alpha|000> + beta|111>) = alpha|000> - beta|111>
```

for `i = 0,1,2`, because `|000>` has 0 in every qubit and `|111>` has 1 in every qubit.

### 6. Hadamard gate

```text
H|0> = |+> = (|0> + |1>)/sqrt(2)
H|1> = |-> = (|0> - |1>)/sqrt(2)
```

Hadamard applied twice cancels:

```text
HH = I
```

So:

```text
H|+> = |0>
H|-> = |1>
```

### 7. CNOT / CX / Controlled-X

These names refer to the same gate.

Rule:

```text
If control is 1, flip target.
If control is 0, do nothing.
```

For `CNOT(1,0)` on `|q1 q0>`:

```text
|00> -> |00>
|01> -> |01>
|10> -> |11>
|11> -> |10>
```

For `CNOT(0,2)` on `|q2 q1 q0>`:

```text
control = q0
target = q2
|111> -> |011>
```

### 8. Controlled-Z / CZ

```text
CZ|00> = |00>
CZ|01> = |01>
CZ|10> = |10>
CZ|11> = -|11>
```

CZ only changes the sign of the `|11>` component.

Important circuit identity:

```text
(I ⊗ H) CZ (I ⊗ H) = CNOT
```

Meaning: if a CZ gate is surrounded by Hadamards on the target qubit, it behaves like CNOT.

Useful identities:

```text
H X H = Z
H Z H = X
```

### 9. Circuit tracking

Use this exact process:

1. Write the initial state.
2. Move left to right through gates.
3. Apply each gate to each basis component.
4. Rewrite the full state after each gate.
5. Only calculate probabilities at measurement or when asked.

Example:

```text
Initial: alpha|000> + beta|111>
After X_2: alpha|100> + beta|011>
After Z_1: alpha|100> - beta|011>
After X_0: alpha|101> - beta|010>
```

### 10. Measurement

`M` means measurement.

Measurement reports a classical bit:

```text
0 or 1
```

For:

```text
|0> -- H -- M
```

State before measurement:

```text
|+> = (|0> + |1>)/sqrt(2)
```

So:

```text
P(report 0) = 1/2
P(report 1) = 1/2
```

Measurement destroys/collapses superposition into the reported result.

### 11. Measuring one qubit in an entangled state

Example circuit:

```text
|00> -> H on top -> CNOT(top control, bottom target)
```

State before measurement:

```text
(|00> + |11>)/sqrt(2)
```

If measuring only the top qubit:

```text
report 0 with probability 1/2
report 1 with probability 1/2
```

Full-system outcomes if measuring both qubits:

```text
00 with probability 1/2
11 with probability 1/2
01 with probability 0
10 with probability 0
```

### 12. Destructive and non-destructive measurement

If the state is:

```text
(|00> + |11>)/sqrt(2)
```

and measuring the top qubit reports `1`:

Non-destructive measurement:

```text
state becomes |11>
```

Destructive measurement:

```text
measured top qubit is removed
remaining state is |1>
```

## Teaching Routine

### Explain Mode
When explaining a concept:

1. State the rule in one sentence.
2. Give the formula.
3. Work a concrete example.
4. State the common mistake.
5. Ask one short check question.

### Test Mode
Ask one question at a time. Do not reveal the answer until the learner answers.

Question format:

```markdown
### Question N
Compute:

`...`

Use the convention `|q2 q1 q0>`.
```

After learner answers:

```markdown
Result: Correct/Incorrect
Correct answer: ...
Reason:
1. ...
2. ...
Next: ...
```

### Remediation Rule
If the learner is wrong:

- Explain the exact mistaken step.
- Give a simpler similar question.
- Do not move to a harder concept until the simpler question is correct.

### Advancement Rule
After two correct answers in a row, increase difficulty by one level.

## Practice Question Bank

### Level 1 — Ket vectors and amplitudes

1. What vector is `|1>`?
2. What vector is `|010>` using basis order `|000>,...,|111>`?
3. Why is `|111>` the eighth basis vector?
4. What does `0.6|000>` mean?
5. If the amplitude is `0.6`, what is the probability?
6. If `alpha = 3 + 4i`, what is `|alpha|^2`?

### Level 2 — Single-qubit gates

1. Compute `X|0>`.
2. Compute `X|1>`.
3. Compute `Z|0>`.
4. Compute `Z|1>`.
5. Compute `H|0>`.
6. Compute `H|1>`.
7. Compute `Z|+>`.
8. Compute `Z|->`.
9. Compute `HH|0>`.
10. Compute `H|+>`.

### Level 3 — Multi-qubit X and Z

Use `|q2 q1 q0>`.

1. Compute `X_2|000>`.
2. Compute `X_1|111>`.
3. Compute `X_0|111>`.
4. Compute `Z_2(alpha|000> + beta|111>)`.
5. Compute `Z_1(alpha|100> + beta|011>)`.
6. Compute `Z_2|++++>` using `|q3 q2 q1 q0>`.

### Level 4 — CNOT and CZ

1. Compute `CNOT(1,0)|00>`.
2. Compute `CNOT(1,0)|10>`.
3. Compute `CNOT(1,0)|11>`.
4. Compute `CNOT(0,2)|111>` using `|q2 q1 q0>`.
5. Compute `CZ|00>`.
6. Compute `CZ|11>`.
7. Compute `CZ(a|00> + b|11>)`.

### Level 5 — Circuits

1. Start with `|00>`. Apply `H` on the top qubit, then `CNOT(top,bottom)`. What is the final state?
2. Start with `|11>`. Apply `H` on the bottom, then `CZ`, then `H` on the bottom. What is the final state?
3. Start with `alpha|000> + beta|111>`. Apply `X_2`, then `Z_1`, then `X_0`. What is the final state?
4. Explain why `H-CZ-H` on the target behaves like CNOT.

### Level 6 — Measurement

1. Circuit: `|0> -> H -> M`. What is `P(report 0)`?
2. Circuit: `|00> -> H on top -> CNOT -> measure top`. What is `P(report 0)`?
3. State before measurement: `(|00> + |11>)/sqrt(2)`. If top qubit reports `1`, what is the non-destructive post-measurement state?
4. Same state and report `1`, but measurement is destructive. What remains?
5. If only the top qubit is measured, what are the possible reported values?
6. If both qubits are measured in `(|00> + |11>)/sqrt(2)`, what are the possible full-system outcomes and probabilities?

## Mastery Checklist
The learner has mastered this skill when they can answer all of these without help:

- Explain amplitude vs probability.
- Convert `|010>` and `|111>` to vectors.
- Explain why `|111>` is the eighth basis state in 3-qubit standard order.
- Compute `Z|->` correctly as `|+>`.
- Compute `Z_2|++++>` with the correct qubit-order convention.
- Apply `CNOT(0,2)|111>` correctly.
- Explain that CX and CNOT are the same gate.
- Explain that CZ only changes the sign of `|11>`.
- Recognize `H-CZ-H` as CNOT.
- Calculate `P(report 0)` for `|0> -> H -> M`.
- Determine post-measurement states for destructive and non-destructive measurement.

## Session Log Instructions
Create or update `session_log.md` after every tutoring/testing session.

Append this template:

```markdown
## YYYY-MM-DD — Quantum Gates Session

### Mode
Teaching / Testing / Review

### Concepts covered
- ...

### Questions
1. Question: ...
   Learner answer: ...
   Correct answer: ...
   Result: Correct/Incorrect
   Note: ...

### Mistakes to review
- ...

### Mastered items
- ...

### Next session
- ...
```

Never erase earlier session history.

## Minimal First Session Plan
Use this if starting from scratch:

1. Quick review: ket notation and amplitudes.
2. Test: vector for `|010>` and probability for `0.6|0>`.
3. Teach: `X`, `Z`, `H` rules.
4. Test: `Z|->`, `H|0>`, `HH|1>`.
5. Teach: CNOT/CX and CZ.
6. Test: `CNOT(0,2)|111>` and `CZ|11>`.
7. Teach: measurement.
8. Test: `|0> -> H -> M`, probability of report `0`.
9. Log mistakes and set next target.

## Response Style Requirements
- Keep answers short when testing.
- Use step-by-step derivations when explaining.
- Avoid abstract physics interpretations unless asked.
- Prefer computation and circuit-tracking examples.
- Correct wrong answers directly but calmly.
- Always specify the qubit ordering convention when relevant.
