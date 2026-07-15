# Portable ONNX Optimization Workflow

This document is a reusable operating procedure for optimizing task-specific
ONNX models in ChatGPT, ChatGPT Work, or Codex. It is designed to prevent the
most expensive failure modes: invalid interfaces, overfitting visible cases,
numerically different outputs, inaccurate cost projections, profiler
incompatibility, and submissions containing several unisolated changes.

The workflow assumes files are paired by numeric task ID:

- `taskXXX.json` contains input/output examples.
- `taskXXX.onnx` is the model for exactly that JSON file.
- A model must never be evaluated against a differently numbered JSON file.

---

## 1. Non-negotiable rules

### 1.1 Independent-work and non-cheating rule

Do not search for, inspect, download, adapt, or reverse-engineer existing
solutions for the supplied task IDs.

Prohibited actions include:

- Web searches containing a supplied task ID together with terms such as
  `solution`, `onnx`, `ARC`, `solver`, `Kaggle`, or equivalent wording.
- Searching public repositories, notebooks, model hubs, competition
  discussions, datasets, or submissions for matching tasks or models.
- Downloading a public model that solves the same task and comparing its graph
  or weights.
- Asking another agent to locate public task-specific solutions.
- Using leaked, private, reverse-engineered, or leaderboard-derived outputs.

Allowed sources are limited to:

- Files supplied by the user.
- First-principles mathematics and computer science.
- General ONNX specifications and operator documentation, if needed.
- Generic graph optimization and numerical analysis references.
- Locally generated tests.
- Scores from the user's own submissions as controlled experimental evidence.

When uncertain whether a source is task-specific, do not use it.

### 1.2 Correctness precedes score

An optimization that is smaller but functionally wrong has value zero.

Never accept a candidate merely because:

- It matches the visible JSON grids.
- It produces the same ArgMax.
- It is algebraically equivalent over real numbers.
- It runs in one ONNX Runtime implementation.
- It has a lower estimated cost.

The required notion of functional equivalence must be discovered and written
down before optimization begins. Until proven otherwise, require complete
floating-point output-tensor equivalence.

### 1.3 Preserve a rollback point

Keep three immutable directories:

```text
original/            # exact user uploads
confirmed/           # last leaderboard-confirmed working batch
candidates/          # experiments only
```

Never overwrite `original/` or `confirmed/`.

### 1.4 One task ID, one file pair

Create and verify an explicit mapping before any execution:

```text
task015.onnx -> task015.json
task081.onnx -> task081.json
...
```

Abort if a pair is missing, duplicated, or ambiguous.

---

## 2. Copy-paste agent brief

Paste this section into a new ChatGPT Work or Codex session together with the
files:

> Optimize the supplied ONNX models using only the supplied ONNX/JSON pairs and
> first principles. Do not search for, inspect, or use public task-specific
> solutions. General ONNX documentation is allowed only when necessary.
>
> Pair files strictly by numeric task ID. Preserve the original files and keep
> a last-confirmed rollback batch.
>
> Before optimizing, determine and record the exact input/output interface,
> static-shape requirements, allowed/disallowed operators, evaluator notion of
> functional correctness, and cost formula. Do not assume that matching ArgMax
> is sufficient. Treat complete output-tensor equality as the default.
>
> Establish baseline outputs and costs first. For every candidate, run the
> validation gates in this workflow: ONNX checker, static shape inference,
> supplied examples, complete-tensor comparison, random full-domain tests,
> exhaustive local tests where feasible, task-specific generated tests,
> multiple runtimes, profiler compatibility, and archive inspection.
>
> Prefer whole-node and intermediate-tensor elimination over approximate
> semantic rewrites. Do not alter bias levels, dtype, accumulation order,
> padding, shape, output names, or numerical scale unless complete equivalence
> is proven under the evaluator.
>
> Change one model or one optimization principle per submission. Predict the
> score before submission and use leaderboard results as controlled A/B
> evidence. Never package an unverified experimental model with a confirmed
> batch.

---

## 3. Phase A — Inventory and immutable baseline

### Step A1: Inventory files

For every task, record:

| Field | Example |
|---|---|
| Task ID | `task015` |
| JSON path | `original/task015.json` |
| ONNX path | `original/task015.onnx` |
| ONNX bytes | 2,150 |
| JSON splits | train, test, generated |
| Example count | 265 |

Verify hashes of the originals and never modify them.

### Step A2: Inspect the external contract

Record for every model:

- Input and output tensor names.
- Input and output dtypes.
- Input and output shapes.
- Opset and IR version.
- Static or symbolic dimensions.
- Model file size.
- Initializer dtypes and shapes.
- Operator sequence.
- Disallowed operators.

Example contract:

```text
input:  name=input,  dtype=float32, shape=[1,10,30,30]
output: name=output, dtype=float32, shape=[1,10,30,30]
opset: 10
all dimensions static
```

Do not change this contract unless the evaluator explicitly permits it and a
separate compatibility submission proves it works.

### Step A3: Run the original models

For every supplied example:

1. Encode the JSON input exactly as the evaluator does.
2. Run the original model.
3. Store the complete raw output tensor.
4. Store its dtype, shape, minimum, maximum, and hash.
5. Decode the result using the known evaluator rule.
6. Confirm that the decoded result matches the JSON output.

The raw outputs form the regression oracle.

### Step A4: Determine functional equivalence

Possible evaluator definitions include:

1. Bit-exact output tensor.
2. Output tensor within a numerical tolerance.
3. Same positive/negative mask.
4. Same rounded tensor.
5. Same ArgMax class grid.

Do not choose definition 5 merely because the model resembles a classifier.
Infer the rule from authoritative specifications or controlled experiments.
If still uncertain, require definition 1.

---

## 4. Phase B — Baseline profiling

Use the same profiler/version as the evaluator whenever possible. A profiler
that cannot parse the candidate makes that candidate unsafe even if ONNX
Runtime executes it.

For each node, record:

- Operator type.
- Input/output static shapes.
- Output dtype.
- Parameter elements.
- Parameter bytes.
- Output activation bytes.
- Multiply-accumulate operations, if relevant.
- Whether the output is the final graph output.

Keep these concepts separate:

| Quantity | Meaning |
|---|---|
| File bytes | Serialized ONNX size |
| Parameter elements | Number used by the scoring formula |
| Parameter bytes | Storage consumed by parameter dtypes |
| Total activation bytes | Sum of tensor outputs under the evaluator's rule |
| Peak live bytes | Runtime liveness estimate; may differ from competition memory |
| MACs | Arithmetic operation estimate, if scored |

Never substitute ordinary process RSS for the evaluator's declared memory
formula.

### Baseline score table

| Model | Parameters | Scored memory | Other cost | Cost | Predicted score |
|---|---:|---:|---:|---:|---:|
| taskXXX |  |  |  |  |  |

Calculate the original batch score and compare it with a known leaderboard
score. If they disagree, calibrate the cost interpretation before optimizing.

---

## 5. Phase C — Analyze the graph from first principles

For each initializer and operator, inspect:

- Zero-weight ratio and exact nonzero locations.
- Identity mappings.
- Common biases.
- Channel-to-channel connectivity.
- Repeated spatial kernels.
- Dtype conversions.
- Intermediate tensors with large static shapes.
- Nodes whose outputs are used once.
- Optional inputs containing all zeros.
- Padding, stride, dilation, group, and kernel orientation.

Derive the mathematical function, but do not replace the network with that
function until the evaluator's required output semantics are known.

### Optimization priority

Use this order:

1. Remove provably unused constants or all-zero optional inputs.
2. Remove true no-op nodes.
3. Collapse operations while preserving complete numerical outputs.
4. Eliminate full-size intermediate tensors.
5. Prune channels only when zero contributions and accumulation order are
   preserved.
6. Reduce parameter count.
7. Change dtype only with a numerical proof and complete-tensor tests.
8. Consider semantic rewrites only as isolated experimental candidates.

Whole-node elimination is often more valuable than compressing weights while
leaving large intermediate tensors intact.

---

## 6. Numerical-safety rules

### 6.1 Real-number equivalence is not floating-point equivalence

The following can change results:

- Reordering additions.
- Splitting one convolution into several convolutions.
- Moving a bias between nodes.
- Removing a common bias.
- Scaling every logit by a positive constant.
- Changing fp16 accumulation to fp32.
- Changing where rounding occurs.
- Replacing convolution with a threshold or pooling rule.

Even when ArgMax is theoretically invariant, the evaluator may inspect
absolute logits or numerical masks.

### 6.2 Bias rule

Do not remove or shift a common bias unless:

- the evaluator is proven to use only ArgMax; and
- the complete private-domain behaviour is protected against ties; and
- a controlled submission verifies the change.

Otherwise preserve the full original bias tensor.

### 6.3 Cast-collapse rule

A path such as:

```text
float32 -> Cast(fp16) -> Conv(fp16) -> Cast(float32)
```

may be replaced by one float32 convolution only if output equality is proven.
Useful proof conditions include:

- Inputs are exactly representable in fp16, such as binary one-hot tensors.
- Weights and biases are exactly representable in both dtypes.
- Every possible accumulation result lies in a range and lattice exactly
  representable in fp16.
- No overflow, underflow, or different rounding boundary can occur.
- Complete output tensors compare bit-exactly on supplied and generated tests.

If weights are arbitrary learned fp16 values, do not assume equality. Use
margin analysis and controlled leaderboard evidence.

### 6.4 Sparse-representation rule

Do not submit sparse initializers merely because ONNX Runtime accepts them.
Require all of:

- ONNX checker acceptance.
- Evaluator-profiler acceptance.
- Correct parameter and memory accounting.
- Static-shape acceptance.
- Multi-runtime execution.

If the profiler throws or silently densifies the weights, reject the candidate.

---

## 7. Phase D — Candidate risk levels

Classify every candidate:

### Level 0 — Original

Unmodified user model. Always retained as fallback.

### Level 1 — Bit-exact

Complete output tensors are identical to the original. Preferred for batch
submissions.

Examples:

- Removing an all-zero optional bias.
- Collapsing casts where all arithmetic is provably exact.
- Removing dead nodes.

### Level 2 — Numerically equivalent under evaluator

Outputs are not bit-exact, but the evaluator-accepted function has strong
evidence:

- exhaustive or large generated tests;
- stable numerical margins;
- multiple runtimes;
- controlled leaderboard confirmation.

Only one Level 2 change should be introduced per submission.

### Level 3 — Semantic or approximate

The graph implements the inferred task rule rather than the original numerical
network. Keep isolated. Do not place it in a main batch until independently
confirmed by the real evaluator.

---

## 8. Phase E — Validation gates

A candidate advances only after passing every applicable gate.

### Gate 1: ONNX validity

- ONNX checker passes.
- Strict shape inference passes.
- Input/output contract is unchanged.
- Every tensor and initializer has a positive static shape.
- No disallowed operator appears.
- Model size is below the limit.

### Gate 2: Supplied decoded outputs

- Every training example passes.
- Every test example passes.
- Every generated example passes.
- Pairing is by exact task ID.

### Gate 3: Complete output tensors

Compare candidate and original tensors using:

- exact equality;
- maximum absolute error;
- maximum relative error;
- number and location of differing elements;
- sign-mask differences;
- ArgMax differences;
- tie differences.

Record all metrics even if the evaluator is believed to use only one.

### Gate 4: Random full-domain tests

Generate inputs using all valid input channels, not only colors observed in the
small training split.

Test:

- sparse grids;
- dense grids;
- borders and corners;
- repeated patterns;
- overlapping influences;
- adversarial ties;
- all-zero and all-one-channel cases.

Use a fixed seed and record the number of cases.

### Gate 5: Exhaustive local tests

For local receptive fields, enumerate all feasible patches.

Examples:

- Binary 3x3 field: `2^9 = 512` cases.
- Four-color 3x3 field: `4^9 = 262,144` cases.
- Use category reduction when several colors have identical effects.

Compare original and candidate at the center cell, including full logits when
possible.

### Gate 6: Task-specific generator

Create an independent generator from the mathematical rule inferred from the
supplied files. Cover:

- minimum and maximum object counts;
- valid overlaps;
- boundary placements;
- varying canvas usage within the fixed tensor;
- distractor colors;
- symmetry and orientation variations.

Do not use public task generators or solutions.

### Gate 7: Multiple runtimes

Run at least two independent implementations when available, for example:

- ONNX Runtime native CPU.
- ONNX Runtime Web/WASM.

Runtime agreement reduces the chance of relying on implementation-specific
behaviour.

### Gate 8: Profiler compatibility

- The evaluator-like profiler loads the model.
- Shape inference succeeds.
- Parameter count is finite and plausible.
- Memory count is finite and plausible.
- Unsupported representations are rejected before submission.

### Gate 9: Archive inspection

- ZIP contains exactly the intended ONNX files.
- Files are at the archive root if required.
- Names exactly match `taskXXX.onnx`.
- No duplicate or stale files exist.
- Archive integrity test passes.
- Record SHA-256 hashes.

---

## 9. Score estimation and calibration

### Step 1: Implement the declared formula

Example:

```text
cost_i = parameters_i + memory_i
score_i = max(1, 25 - ln(cost_i))
batch_score = sum(score_i for correct models)
```

Do not guess what `memory_i` includes. Candidate definitions may include:

- all node outputs;
- non-final node outputs only;
- peak live tensors;
- parameter bytes;
- static weights per consumer;
- dtype-specific activation bytes.

### Step 2: Calibrate using a confirmed batch

Calculate the score under each interpretation and compare with the real
leaderboard score. Select the interpretation that reproduces the confirmed
result.

### Step 3: Predict deltas, not only totals

For a candidate changing one model:

```text
expected_batch_new = confirmed_batch_score
                     - confirmed_model_score
                     + candidate_model_score
```

This is more reliable than re-estimating the whole batch.

### Step 4: Require an explicit score manifest

| Model | Confirmed correct? | Params | Scored memory | Cost | Score | Delta |
|---|---|---:|---:|---:|---:|---:|
| taskXXX | yes/no |  |  |  |  |  |

Do not submit if the predicted improvement cannot be attributed to a specific
graph change.

---

## 10. Leaderboard submissions as controlled experiments

Leaderboard results are scarce experimental data. Use them carefully.

### Submission rule

Starting from the last confirmed batch, change only:

- one model; or
- one shared optimization principle whose affected models can be identified.

Avoid changing dtype, shape, bias, operator structure, and several models at
the same time.

### Interpreting an aggregate score

If only an aggregate score is returned:

1. Calculate the expected score of every changed model.
2. Enumerate plausible subsets of functionally correct models.
3. Find which subset sum matches the observed aggregate score.
4. Treat this as an inference, not proof.
5. Confirm with a subsequent single-change submission.

### Rollback rule

If a submission scores below the confirmed baseline:

- do not patch the failed batch in place;
- return to `confirmed/`;
- isolate one suspected cause;
- build a new candidate from the confirmed files.

---

## 11. Recommended working files

```text
project/
├── original/
│   ├── taskXXX.json
│   └── taskXXX.onnx
├── confirmed/
│   └── taskXXX.onnx
├── candidates/
│   └── candidate_name/taskXXX.onnx
├── tests/
│   ├── supplied_validation.*
│   ├── tensor_equivalence.*
│   ├── random_validation.*
│   ├── exhaustive_local.*
│   └── task_generators.*
├── profiles/
│   ├── baseline.json
│   └── candidate_name.json
├── session_log.md
├── model_manifest.csv
└── submission.zip
```

---

## 12. Session log template

```markdown
# Session log

## Confirmed baseline
- Batch name:
- Leaderboard score:
- Archive SHA-256:
- Functional status:

## Evaluator contract
- Input:
- Output:
- Static-shape rule:
- Disallowed ops:
- Functional comparison rule:
- Cost formula:
- Calibrated memory definition:

## Current experiment
- Candidate name:
- Starting batch:
- Models changed:
- One optimization principle:
- Mathematical justification:
- Numerical risk:

## Validation
- ONNX checker:
- Static shapes:
- Supplied cases:
- Complete tensor exact cases:
- Max absolute error:
- Random cases and seed:
- Exhaustive cases:
- Generated cases:
- Runtimes:
- Profiler:

## Cost
- Per-model old/new cost:
- Predicted score:
- Predicted delta:

## Result
- Submitted archive hash:
- Leaderboard score:
- Interpretation:
- Promote to confirmed? yes/no
```

---

## 13. Model manifest template

```csv
task_id,json_file,original_model,candidate_model,input_name,input_dtype,input_shape,output_name,output_dtype,output_shape,opset,parameters,scored_memory,predicted_score,supplied_passes,tensor_exact_cases,random_cases,generated_cases,risk_level,status
```

Status values:

- `original`
- `experimental`
- `locally_verified`
- `leaderboard_confirmed`
- `rejected`

---

## 14. Pre-submission checklist

Do not submit until every box is checked:

- [ ] No task-specific online solution was searched or used.
- [ ] Candidate started from the last confirmed batch.
- [ ] Exact task-ID pairing was verified.
- [ ] External interface is unchanged.
- [ ] All shapes are static.
- [ ] No disallowed operator exists.
- [ ] ONNX checker passes.
- [ ] Supplied examples pass.
- [ ] Complete output tensors were compared.
- [ ] Random full-domain tests pass.
- [ ] Exhaustive local tests pass where feasible.
- [ ] Task-specific generated tests pass.
- [ ] At least two runtimes agree.
- [ ] Evaluator-like profiler parses the model.
- [ ] Score and delta are calculated.
- [ ] Only one controlled change was introduced.
- [ ] ZIP contents and hashes were inspected.
- [ ] Rollback batch remains untouched.

---

## 15. Practical principles distilled from experience

1. **The evaluator defines equivalence.** Mathematical elegance does not.
2. **Compare raw tensors first.** ArgMax is a secondary diagnostic.
3. **Preserve biases and rounding boundaries.** Absolute logits may matter.
4. **Eliminate intermediates before compressing parameters.** Large tensors
   often dominate cost.
5. **Static-shape compliance is part of correctness.** Not an afterthought.
6. **A runtime pass is not a profiler pass.** Both are mandatory.
7. **Visible examples are not a proof.** Add random, exhaustive, and generated
   tests.
8. **One leaderboard change at a time.** Aggregate scores can then identify the
   working component.
9. **Predict before submitting.** A correct cost model turns the leaderboard
   into a scientific instrument.
10. **Keep a confirmed rollback batch.** Every experiment should be reversible.
11. **Do not use public task solutions.** Independent first-principles work is
    both an integrity rule and a better path to generalizable optimization.

