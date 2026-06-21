# SKILL.md — Control Cabinet Assembly Tutor Skill

## Skill name

Control Cabinet Assembly Tutor

## Skill purpose

Use this skill to teach and test a beginner who wants to pass basic job requirements for **monter / elektromonter szaf sterowniczych** roles in Poland.

The target is practical job readiness:

- recognizing cabinet components,
- reading simple technical documentation,
- understanding point-to-point wiring,
- checking work quality,
- explaining basic relay/control logic,
- answering beginner interview questions honestly in Polish.

This skill does **not** certify the learner as an electrician. It prepares the learner for entry-level screening, supervised workshop work, and further employer training.

## Required companion file

Use `SESSION_LOG.md` with this skill.

At the start of each session:

1. Read `SESSION_LOG.md`.
2. Identify the current level and weak points.
3. Continue from the next unfinished task.

At the end of each session:

1. Update the master progress table.
2. Add a new session entry.
3. Record score, mistakes, corrections, vocabulary, and next-session instruction.

## Learner profile assumptions

The learner may:

- have limited electrical background,
- know some English but need Polish job vocabulary,
- want fast practical preparation,
- need frequent testing rather than long theory.

Default language: English explanations with Polish technical terms.

## Safety rule

Never instruct the learner to work on live mains voltage.

Permitted practice types:

- paper wiring exercises,
- diagram reading,
- simulator exercises,
- battery-powered circuits,
- safe low-voltage circuits such as supervised 24V DC trainers.

For real workplace electrical work, always emphasize:

- disconnect power,
- verify absence of voltage,
- follow workplace procedures,
- use supervision and proper authorization,
- do not perform energized work as a beginner.

## Teaching loop

For every module, use this loop:

1. Teach one small concept.
2. Give 5–10 test questions or one small exercise.
3. Wait for the learner's answer.
4. Grade with score and decision: `PASS`, `REVIEW`, or `REPEAT`.
5. Correct mistakes.
6. Log the result in `SESSION_LOG.md`.
7. Move forward only if the pass rule is satisfied.

Do not dump full theory. Keep each teaching unit small and testable.

## Global pass criteria

The learner completes the skill when they:

- pass Levels 1–5,
- pass the final job simulation with at least 80%,
- can give a simple, honest Polish answer about their technical level.

Final label to use if successful:

> Prepared beginner for supervised entry-level control cabinet assembly work.

Do not call the learner certified or fully qualified.

---

# Level 1 — Component Recognition

## Goal

The learner recognizes common cabinet components, abbreviations, and Polish names.

## Teach these items

| Marking | Component | Polish | Function |
|---|---|---|---|
| Q1 | circuit breaker / switch-disconnector | wyłącznik / rozłącznik | disconnects or protects a circuit |
| F1 | fuse | bezpiecznik | protects against overcurrent |
| K1 | relay or contactor | przekaźnik / stycznik | switches a circuit |
| S1 | switch / button / sensor input | przycisk / łącznik / czujnik | gives an input signal |
| M1 | motor | silnik | driven load |
| X1 | terminal strip | listwa zaciskowa | connection point for wires |
| H1 | signal lamp | lampka sygnalizacyjna | shows status |
| A1/A2 | coil terminals | zaciski cewki | energize relay/contactor coil |
| 13/14 | NO contact terminals | styk normalnie otwarty | closes when activated |
| 21/22 | NC contact terminals | styk normalnie zamknięty | opens when activated |
| L/N/PE | live/neutral/protective earth | faza / neutralny / ochronny | AC power conductors |
| +24V/0V | DC control supply | zasilanie sterowania | typical control voltage |

## Test questions

Ask 10 questions:

1. What is a **szafa sterownicza**?
2. What is a **listwa zaciskowa**?
3. What does **K1** usually mean?
4. What does **S1** usually mean?
5. What does **X1:4** mean?
6. What are **A1/A2** on a relay or contactor?
7. What is the difference between **NO** and **NC**?
8. What is **PE**?
9. What is a **stycznik** used for?
10. What does **+24V / 0V** usually mean?

## Passing rule

Pass if at least 8/10 are correct.

If failed, reteach weak items and retest with new examples.

---

# Level 2 — Symbols and Basic Electrical Logic

## Goal

The learner understands contact, coil, NO/NC logic, and simple signal flow.

## Teach these concepts

- **NO — normally open / normalnie otwarty**: open by default, closes when activated.
- **NC — normally closed / normalnie zamknięty**: closed by default, opens when activated.
- **Coil / cewka**: when energized, changes relay/contactor contact states.
- **Contact / styk**: a switching element controlled by a relay, contactor, switch, or button.
- **START/STOP logic**: STOP is usually NC; START is usually NO.

## Core example

```text
+24V → S1 STOP NC → S2 START NO → K1:A1
K1:A2 → 0V
```

Expected explanation:

+24V goes through the NC STOP button, then through the NO START button, then to K1 coil terminal A1. A2 returns to 0V. Pressing START energizes K1 if STOP is not pressed. Pressing STOP opens the circuit and de-energizes K1.

## Test questions

Ask 8 questions:

1. Why is STOP usually NC?
2. Why is START usually NO?
3. What happens to K1 when S2 START is pressed?
4. What happens when S1 STOP is pressed?
5. What is the return path to 0V?
6. What does the coil do?
7. What does a contact do?
8. What happens if a wire before K1:A1 is disconnected?

## Passing rule

Pass if the learner can explain the circuit in their own words and answer at least 6/8 correctly.

---

# Level 3 — Reading Simple Wiring Diagrams

## Goal

The learner can follow a basic control circuit from supply to return.

## Reading algorithm

Teach this algorithm:

1. Find the power source: `+24V/0V` or `L/N`.
2. Identify protection: `F1`, `Q1`.
3. Identify inputs: `S1`, `S2`, sensors.
4. Identify controlled element: `K1` coil, `H1` lamp, `M1` motor.
5. Trace current path from supply to return.
6. State what condition makes the output active.

## Practice diagrams

### Diagram A

```text
+24V → F1 → S1 NO → H1:+
H1:- → 0V
```

Expected explanation: When S1 is pressed, H1 receives 24V and turns on.

### Diagram B

```text
+24V → F1 → S1 STOP NC → S2 START NO → K1:A1
K1:A2 → 0V
```

Expected explanation: STOP must be closed and START must be pressed to energize K1.

### Diagram C

```text
+24V → F1 → K1:13/14 → H1:+
H1:- → 0V
```

Expected explanation: If K1 contact 13/14 is closed, H1 turns on.

## Test questions

For each diagram, ask:

1. What is the power source?
2. What is the protection device?
3. What are the inputs?
4. What is the output/load?
5. What condition makes the output active?
6. What would you check if the output does not work?

## Passing rule

Pass if the learner can read 2 of 3 diagrams correctly without major guidance.

---

# Level 4 — Wiring Documentation and Terminal Numbers

## Goal

The learner understands point-to-point wiring instructions, terminal numbering, and component terminals.

## Teach notation

- **X1:1** = terminal strip X1, terminal number 1.
- **K1:A1** = coil terminal A1 on component K1.
- **S1:13 / S1:14** = terminals of a NO contact on switch/button S1.
- **S1:21 / S1:22** = terminals of a NC contact on switch/button S1.
- **Wire 101** = wire label/number 101.

## Example documentation

```text
Wire 101: X1:1 → K1:A1
Wire 102: K1:A2 → 0V
Wire 103: X1:2 → S1:13
Wire 104: S1:14 → K1:13
```

Expected interpretation:

Wire 101 connects terminal strip X1 terminal 1 to coil terminal A1 of component K1.

## Test questions

Ask:

1. Where does wire 101 go?
2. What does X1:1 mean?
3. What does K1:A1 mean?
4. Where does wire 103 start and end?
5. How would you check whether wire 104 is installed correctly?
6. What is the difference between a component label and a terminal label?
7. What does a wire number help with?
8. Why must terminal numbers be checked carefully?

## Passing rule

Pass if at least 6/8 are correct and the learner can translate one wiring instruction into real work.

---

# Level 5 — Quality Control

## Goal

The learner can detect simple assembly and wiring mistakes.

## Teach checklist

The learner should memorize this inspection checklist:

1. correct component,
2. correct terminal,
3. correct wire number,
4. correct wire color,
5. ferrule/crimp correct,
6. screw tightened,
7. no exposed copper,
8. label readable,
9. clean wire route,
10. PE/ground connection checked.

## Mistake examples

### Example 1

```text
Documentation:
Wire 201: X1:4 → K2:A1

Actual work:
Wire 201: X1:5 → K2:A1
```

Expected answer:

Wrong. Wire 201 should start from X1:4, not X1:5.

### Example 2

```text
Documentation:
Wire 301: K1:A2 → 0V

Actual work:
Wire 301: K1:A1 → 0V
```

Expected answer:

Wrong. Wire 301 should connect K1:A2 to 0V, not K1:A1 to 0V.

## Test questions

Give 5 mismatch examples. Ask the learner to identify:

1. whether the work is correct,
2. what is wrong,
3. how to correct it,
4. what checklist item catches the mistake.

## Passing rule

Pass if the learner correctly identifies at least 4/5 mistakes.

---

# Final Job Simulation

## Goal

The learner demonstrates beginner job readiness.

## Simulation task

Use this task:

```text
Task: Wire a basic 24V relay/lamp circuit.

Documentation:
+24V → F1 → S1 STOP NC → S2 START NO → K1:A1
K1:A2 → 0V
K1:13/14 → H1:+
H1:- → 0V
```

Ask the learner to do all of the following:

1. List all components.
2. Explain the circuit function.
3. Write point-to-point wiring steps.
4. Explain how to inspect the work.
5. Explain their skill level in simple Polish for an interview.

## Expected technical answer

The learner should identify:

- `+24V/0V` control supply,
- `F1` protection,
- `S1 STOP NC`,
- `S2 START NO`,
- `K1` relay/contactor coil,
- `K1:13/14` NO auxiliary contact,
- `H1` signal lamp.

The learner should explain:

- STOP must be closed.
- START must be pressed to energize K1.
- When K1 is energized, contact 13/14 closes.
- H1 turns on when contact 13/14 is closed.
- Inspection checks terminal numbers, wire numbers, labels, crimp/ferrule quality, screw tightness, clean route, and PE if relevant.

## Polish interview answer

Teach and test this answer:

```text
Umiem czytać podstawowe schematy elektryczne. Rozpoznaję elementy takie jak bezpiecznik, przekaźnik, stycznik, zasilacz, zaciski, przyciski NO/NC i oznaczenia przewodów. Nie jestem jeszcze zaawansowany, ale rozumiem logikę prostych obwodów sterowania i szybko uczę się pracy z dokumentacją techniczną.
```

English meaning:

I can read basic electrical diagrams. I recognize elements such as a fuse, relay, contactor, power supply, terminals, NO/NC buttons, and wire markings. I am not advanced yet, but I understand the logic of simple control circuits and quickly learn to work with technical documentation.

## Final passing rule

Pass if the learner scores at least 80% and gives a safe, honest Polish interview answer.

---

# Remediation rules

If the learner fails a level:

1. Identify the exact weak concept.
2. Give a shorter explanation.
3. Give 3 easier examples.
4. Retest with similar but not identical questions.
5. Log the weakness in `SESSION_LOG.md`.

Common weak points:

| Weak point | Remediation |
|---|---|
| Confuses contact and coil | Use K1 coil vs K1 contact examples repeatedly |
| Confuses NO and NC | Use door/button analogy and START/STOP examples |
| Cannot read X1:4 | Drill component label vs terminal number |
| Cannot trace circuit | Force supply → protection → input → output → return method |
| Quality mistakes | Use mismatch examples and checklist mapping |
| Polish wording weak | Provide one simple sentence and make learner repeat/adapt it |

---

# Optional extension after passing

Only after the learner passes the beginner track, optionally introduce:

- basic multimeter concepts,
- ferrules and crimp quality,
- DIN rail layout,
- wire colors,
- PLC input/output basics,
- SEP G1 vocabulary,
- simple panel layout reading.

Do not introduce these too early if the learner still cannot handle relay logic and terminal numbers.
