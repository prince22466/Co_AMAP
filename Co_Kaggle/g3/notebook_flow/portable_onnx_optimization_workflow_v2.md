# Portable ONNX Optimization Postmortem and Workflow

## 1. Final outcome and the central lesson

The best **confirmed** submission in this optimization run scored **237.8**, up from roughly **170** for the supplied ONNX models. That is a substantial improvement of about **67.8 points**.

The final attempts projected near 240 locally but scored about 224. Reverting one suspected model did not recover the score. This proved the most important lesson of the whole run:

> Local correctness on every supplied example is necessary, but it does not prove correctness on Kaggle's hidden inputs. When several task models are changed in one submission, the aggregate leaderboard score cannot reliably identify which rewrite failed.

The stable checkpoint must therefore remain the 237.8 package. Any future work should branch from that checkpoint and test one changed task at a time.

## 2. What the score history taught us

| Submission or stage | Kaggle result | Main lesson |
|---|---:|---|
| Supplied ONNX models | ~170 | Functionally valid, but expensive. |
| Conservative optimizer package | 162 | Generic graph optimization can make the score worse. |
| Original plus task379 | 172 | Isolated task replacement can produce a measurable gain. |
| Projected 188 package | 189 | The cost projection became useful after calibration. |
| Projected 204 package | 208 | Structural task-specific rewrites produced large gains. |
| Projected 218 package | 218.8 | Projection and Kaggle score were closely aligned. |
| Projected 227 package | 227 | The workflow was still well calibrated. |
| Projected 234 package | 94.24 | A locally plausible semantic rewrite can catastrophically fail hidden cases. |
| Projected 231 package | 231 | Returning to safer transformations recovered the score. |
| Sparse uint8 TopK package | Processing error | Operator name alone is insufficient; dtype/signature compatibility matters. |
| Float TopK package | 234.25 | Float TopK signatures were processed successfully. |
| Quantized probe | 236.67 | Carefully selected quantized operators were useful. |
| If-guard package | 207.7 | Control-flow guards were not a safe portability strategy. |
| Hidden-safe confirmed package | 237.19 | A strong confirmed checkpoint. |
| Raw-exact probe | ~221 | Raw equality on supplied examples still did not establish hidden generalization. |
| Hidden-safe raw package | **237.8** | Best confirmed result. |
| Projected 240 package | 224.04 | At least one new task rewrite failed hidden behavior. |
| Same package with task085 reverted | 223.94 | The aggregate score had falsely suggested task085 was the sole culprit. |

The last two results are especially important. A score loss that resembles one task's projected contribution is **not enough to identify that task** when many tasks changed together.

## 3. Empirical score model

The local score projection was calibrated as:

\[
S_i = \max\left(1, 25 - \ln(K_i)\right)
\]

where

\[
K_i = P_i + A_i - 36000
\]

and:

- \(P_i\) is the number of initializer elements;
- \(A_i\) is the total number of inferred intermediate-output bytes;
- 36,000 bytes is the fixed float32 output tensor of shape `[1, 10, 30, 30]`;
- the package projection is the sum over all task models.

This estimate predicted several successful packages very closely. It only estimates efficiency, however. It cannot prove hidden functional correctness.

The useful transformation-level estimate is:

\[
\Delta S_i \approx \ln\left(\frac{K_{old}}{K_{new}}\right)
\]

This means percentage reduction matters more than the absolute number of removed bytes.

Examples:

| Remaining cost | Per-task gain |
|---:|---:|
| 90% of old cost | 0.105 |
| 75% of old cost | 0.288 |
| 50% of old cost | 0.693 |
| 33% of old cost | 1.109 |
| 10% of old cost | 2.303 |

Small dtype and node reductions are valuable, but they cannot normally create a package-level jump of 10 points.

## 4. What to inspect in every ONNX model

### 4.1 Establish the data contract first

Before optimizing, determine:

- exact input name, type, and shape;
- exact output name, type, and shape;
- padding convention;
- channel ordering;
- whether the evaluator compares the full output tensor or only decoded colors;
- operator-set version and domains;
- whether every intermediate shape can be inferred.

In this work, the model contract was a top-left-padded one-hot input and float32 one-hot output with shape `[1, 10, 30, 30]`.

The decisive discovery was that preserving only `ArgMax(output)` was not safe. Several affine-logit or semantic-output rewrites decoded correctly on supplied examples but lost large numbers of leaderboard points. The terminal must preserve the required raw one-hot tensor.

### 4.2 Profile parameters and every node output

For each node, record:

- operator type;
- output shape;
- output dtype;
- output byte count;
- whether the value is consumed more than once;
- whether it is a view-like operation that can be removed;
- whether it duplicates another mask or coordinate tensor.

Sort node outputs by bytes. The largest opportunities usually come from:

- full `[1, 10, 30, 30]` float32 intermediates: 36,000 bytes each;
- full int64 index grids: 7,200 bytes;
- full float32 scalar-color grids: 3,600 bytes;
- multi-color cropped basis tensors;
- repeated full-grid `Where`, `Equal`, `Cast`, `Pad`, `Reshape`, and `Transpose` outputs;
- large mostly-zero convolution initializers used to crop and project colors;
- repeated flood-fill or shift stages.

### 4.3 Look for task semantics, not only graph syntax

The high-value improvements came from understanding what a task did:

- clearing pixels with a Boolean mask;
- painting one dynamic color;
- detecting a bounding rectangle;
- expanding or shifting a component;
- mapping a small set of colors;
- drawing a fixed number of rows, columns, diagonals, or rays;
- selecting rare or frequent colors;
- constructing a three- or four-color output.

Once the semantic operation is known, replace the long learned/generated graph with a smaller exact construction using portable operators.

## 5. Transformations that are usually easy and safe

These should be attempted first because they make few or no assumptions about hidden inputs.

### 5.1 Dead-node and unused-initializer pruning

- Trace backward from graph outputs.
- Remove unreachable nodes.
- Remove initializers unused by live nodes.
- Promote simple `Constant` tensors to initializers if that reduces overhead.
- Re-run shape inference after pruning.

This is low risk and should be applied after every structural rewrite.

### 5.2 Exact algebraic simplification

Examples:

- replace `Equal(x, false)` with `Not(x)`;
- reuse an already computed Boolean mask;
- eliminate redundant `Squeeze`/`Unsqueeze`/`Reshape`/`Transpose` chains;
- fold scalar arithmetic into initializers;
- remove a cast when the consumer accepts the existing dtype;
- use one comparison instead of comparison-plus-conversion when semantics are identical.

The transformation must preserve shape, broadcasting, dtype, tie behavior, and padding.

### 5.3 Early spatial cropping

If task inputs are known to occupy at most a smaller rectangle, perform expensive work on the cropped area and pad only once at the end.

This can reduce every downstream activation. It is safe only when the maximum supported task dimensions are established, including hidden cases.

### 5.4 Boolean and uint8 intermediates

Binary masks and small exact counts should use `bool` or `uint8` when consumers support them. This can reduce activation memory by four or eight times compared with float32 or int64.

Good candidates include:

- object masks;
- row/column presence;
- local-neighbor counts;
- flood states;
- interval masks;
- quantized convolution or matrix-multiplication inputs.

Always check the exact operator signature. A supported operator with an unsupported dtype still fails.

### 5.5 Direct `Where` output for clear/paint tasks

When the result is simply the input with a mask cleared or painted with one one-hot color, the safest terminal is often:

```text
Where(mask, replacement_one_hot, input)
```

This avoids reconstructing the full output through an int64 color grid and `OneHot`.

### 5.6 Sparse color basis plus 1×1 convolution

If the output contains only a small number of colors:

1. build one Boolean plane per output color over the cropped area;
2. cast the compact basis to float32;
3. create a small dynamic mapping from basis planes to the 10 output channels;
4. use a 1×1 convolution with asymmetric bottom/right padding.

This worked well for several tasks. The common failure modes are wrong color count, wrong basis ordering, and symmetric padding that shifts the crop away from the top-left corner.

### 5.7 Quantized convolution and matrix multiplication

`QLinearConv` and `QLinearMatMul` were useful for binary masks and small integer counts. They are especially effective when they replace float32 spatial intermediates.

The quantization must be exact for the value range. Watch for saturation and scale/zero-point mismatches.

## 6. Transformations that are dangerous

### 6.1 Preserving only decoded `ArgMax`

Do not replace exact one-hot output with arbitrary logits merely because `ArgMax` is unchanged locally. The Kaggle behavior showed that raw output representation matters.

### 6.2 Shortcutting semantic conditions

The failed direct-ArgMax task085 rewrite is a representative example. It assumed that taking an ArgMax over an intermediate count was equivalent to the original foreground-selection logic. It passed all supplied examples but introduced an unproven hidden invariant.

Even though reverting task085 did not recover the final package, the rewrite itself still illustrates the danger: an apparent equivalence on the observed dataset is not a proof.

### 6.3 Reducing fixed iteration or propagation depth

Flood-fill, dilation, propagation, or repeated-shift depth can look excessive on supplied examples. Reducing it may pass almost every example and still fail a larger or more separated hidden object.

Task279 demonstrated this: depth four passed 264 of 266 supplied examples, while depth five was required for all examples.

### 6.4 Inferring active area from scalar colors carelessly

Color zero and padded zero are not automatically the same thing. Some compact color projections encoded active color zero as `0.5`, used float comparisons to detect activity, and then cast it to uint8 zero for color operations.

Reusing a `color > 0` uint8 mask in that situation silently deletes legitimate background pixels.

### 6.5 Assuming color frequency order

Do not assume that outer color, inner color, marker color, or background color is always the most or least frequent unless the task grammar proves it. Ties and size changes can reverse the ordering.

### 6.6 Relying on arbitrary random tests

Completely random grids often violate the task grammar and are not informative. A candidate and the original model can differ on nonsense inputs without indicating a real hidden failure—or agree on nonsense inputs while failing a meaningful edge case.

Use structured metamorphic tests instead.

### 6.7 Combining many unconfirmed changes

This was the most expensive process mistake. The projected-240 package changed several task models at once. Its aggregate score fell to 224, and reverting task085 alone produced 223.94. Because no one-task ablation submissions existed, the failing rewrite could not be identified from the leaderboard result.

Never promote several unconfirmed semantic rewrites in the same package.

## 7. Operator and signature compatibility learned in this run

| Operator/signature | Observed status |
|---|---|
| `Loop` | Forbidden |
| `Scan` | Forbidden |
| `NonZero` | Forbidden |
| `Unique` | Forbidden |
| `Script` | Forbidden |
| `Function` | Forbidden |
| `If` | Avoid; control-flow probe was not reliable |
| `TopK(float32)` | Processed successfully |
| `TopK(float16)` | Processed in a scored package, but still use only with exact validation |
| `TopK(uint8)` | Processing errors occurred on specific tasks |
| `OneHot(int64)` | Supported |
| `OneHot(int32)` | Local runtime lacked implementation |
| `OneHot(uint8)` | Do not use; local runtime lacked implementation |
| `ReduceSum(uint8)` | Rejected by local runtime |
| `QLinearConv` | Processed successfully with tested signatures |
| `QLinearMatMul` | Processed successfully with tested signatures |
| `ScatterND` | Processed in scored models with tested signatures |
| `Pad` | Processed successfully |
| `MaxPool` | Processed successfully |

Compatibility must be tracked by the full tuple:

```text
(operator, opset, input dtypes, output dtype, attributes, tensor ranks)
```

An operator being absent from the forbidden list does not prove that every signature is supported.

## 8. Correct validation ladder

Every candidate should pass all levels below before promotion.

### Level 1: Static ONNX validation

- `onnx.checker.check_model(..., full_check=True)`;
- strict shape inference;
- zero missing intermediate shapes;
- expected input/output metadata;
- no forbidden operators;
- no unapproved domains or functions;
- no unused nodes or initializers after final pruning.

### Level 2: Runtime validation

- create an ONNX Runtime CPU session;
- execute every encodable supplied example;
- reject runtime warnings that reveal type or schema problems;
- test every new operator signature in a minimal candidate before using it broadly.

### Level 3: Raw-output equivalence

Compare the complete float32 output tensor against the required one-hot output:

```python
np.array_equal(candidate_raw, expected_one_hot)
```

Also check decoded output, but never substitute decoded equality for raw equality.

In the last local validation, 3,744 encodable examples matched exactly. Thirty-seven JSON examples exceeded the fixed 30×30 ONNX input and could not be represented by the local encoder.

### Level 4: In-domain metamorphic tests

Use the original scored ONNX model as an oracle. Generate task-valid variants and compare candidate raw output with original raw output.

Depending on the task, test:

- all valid palette permutations;
- color zero in every semantic role;
- minimum and maximum dimensions;
- rectangular versus square canvases;
- objects touching every boundary;
- all supported orientations and reflections;
- maximum marker or object counts;
- tied color counts;
- maximum component separation;
- longest required propagation distance;
- one-pixel and degenerate components;
- translated objects with identical relative geometry;
- reordered independent objects.

These tests are much more useful than unrestricted random grids.

### Level 5: One-task Kaggle ablation

Given confirmed baseline `B` and candidate rewrite `C_taskXXX`, submit:

```text
B with only taskXXX replaced
```

Interpret the result:

- score gain close to projection: promote the rewrite;
- processing error naming that task: reject the signature;
- large score loss: hidden semantic failure;
- small unexplained difference: investigate raw format, numerical precision, or partial hidden coverage.

Only after a rewrite is confirmed should it enter the next baseline.

### Level 6: Stacked milestone package

Combine only individually confirmed task replacements. Validate the assembled ZIP again and submit it as a milestone.

## 9. More efficient end-to-end workflow

### Phase A: Freeze and inventory

1. Keep the best confirmed package immutable.
2. Record its Kaggle score, model hashes, per-task profiles, operator signatures, and validation report.
3. Create a candidate ledger.

Recommended ledger columns:

| Field | Purpose |
|---|---|
| Task | Task identifier |
| Baseline hash | Prevent accidental baseline drift |
| Candidate hash | Identify exact model submitted |
| Transformation | One concise description |
| Old/new cost | Measure structural improvement |
| Projected delta | Expected efficiency gain |
| Supplied raw exact | Required local result |
| Metamorphic exact | Hidden-generalization evidence |
| New signatures | Compatibility risk |
| Ablation ZIP | Exact submission artifact |
| Kaggle result | Confirmation or rejection |
| Status | Experimental / confirmed / rejected |

### Phase B: Rank opportunities

For every task, estimate:

- percentage of cost in parameters;
- percentage in activation bytes;
- largest five node outputs;
- fixed output-terminal overhead;
- potential cost after a semantic rewrite;
- hidden-invariant risk;
- signature risk.

Prioritize tasks with both a large percentage-reduction opportunity and strong, testable semantics.

### Phase C: Work one task at a time

1. Infer the task grammar.
2. Write down every assumed invariant explicitly.
3. Create adversarial examples for each invariant.
4. Implement one structural idea.
5. Validate raw output against the original model.
6. Profile the candidate.
7. Reject it if the projected gain is too small for its semantic risk.
8. Build a one-task ablation ZIP.

### Phase D: Promote only confirmed changes

After Kaggle confirmation:

1. copy the task model into a new immutable baseline;
2. update the baseline score and hashes;
3. rerun package validation;
4. start the next task from that new baseline.

### Phase E: Package carefully

Before submission:

- include exactly the required `taskXXX.onnx` files at ZIP root;
- exclude reports and directories;
- run archive integrity testing;
- load every model from the final extracted ZIP, not merely the source directory;
- verify hashes and model count;
- rerun forbidden-operator and missing-shape checks.

## 10. How to pursue a genuine +10 improvement

It is not realistic to guarantee +10 on every submission, especially near 238. The score is logarithmic and hidden correctness risk increases as rewrites become more aggressive.

For a package gain of 10, the required average cost reduction is approximately:

| Number of improved tasks | Average remaining cost | Average reduction |
|---:|---:|---:|
| 5 tasks | 13.5% | 86.5% |
| 10 tasks | 36.8% | 63.2% |
| 15 tasks | 51.3% | 48.7% |

Therefore, +10 requires multiple structural rewrites, not micro-optimization.

A practical +10 strategy is:

1. Find four to six tasks whose graphs can be reduced by roughly 10×.
2. Prefer tasks with simple, closed-form semantics and strong metamorphic symmetries.
3. Develop and submit each task as an isolated ablation.
4. Accumulate confirmed gains privately in a new baseline.
5. Release a milestone package only after the sum of confirmed projected gains exceeds 10 with margin.

For example, five independently confirmed 10× cost reductions contribute approximately:

\[
5\ln(10) \approx 11.51
\]

By contrast, reducing every task by only 10% gives approximately:

\[
15\ln(1/0.9) \approx 1.58
\]

So the process should distinguish two kinds of submissions:

- **diagnostic ablations**, which may gain only a fraction of a point but establish correctness;
- **milestone packages**, which stack enough confirmed ablations to target +10.

Trying to force every diagnostic submission to gain +10 encourages unsafe batching and was one cause of the final regression.

## 11. Recommended decision rules

### Accept immediately

- dead-code or initializer pruning;
- exact algebraic identity with identical dtypes and broadcasting;
- removal of redundant view nodes;
- reuse of an identical existing tensor;
- smaller exact initializer with unchanged runtime computation.

### Require metamorphic testing and one-task Kaggle ablation

- semantic graph replacement;
- reduced propagation depth;
- new color-selection logic;
- changed tie behavior;
- smaller spatial bound;
- sparse output basis;
- dynamic coordinate construction;
- dtype/signature change;
- quantized arithmetic;
- replacement of `TopK`, `ArgMax`, or sorting logic.

### Reject unless independently proven

- decoded-only equivalence;
- hidden-size assumptions inferred only from supplied examples;
- unimplemented local operator signatures;
- forbidden control flow;
- many unconfirmed tasks in one ZIP;
- a projected gain that is small relative to the hidden-correctness risk.

## 12. Compact checklist for another Codex session

```text
1. Start only from the best Kaggle-confirmed baseline.
2. Never search online for task solutions; use first principles and the supplied models.
3. Profile parameters and inferred activation bytes per node.
4. Rank opportunities by percentage cost reduction and hidden risk.
5. Optimize one task at a time.
6. Write every assumed invariant before implementation.
7. Preserve exact float32 one-hot output, not only ArgMax.
8. Run checker, strict shape inference, runtime, raw equality, and metamorphic tests.
9. Treat operator compatibility as signature-specific.
10. Submit one-task ablations before stacking.
11. Promote only Kaggle-confirmed rewrites.
12. Keep immutable hashes and a candidate ledger.
13. Use diagnostic submissions for attribution and milestone submissions for large gains.
14. Never claim a hidden-safe result from supplied-example equality alone.
```

## 13. Final recommendation

If optimization resumes, use the **237.8 submission as the only baseline**. Do not branch from either 224-point package.

The next step should not be another multi-task package. It should be an ablation series in which each new submission replaces exactly one of the models changed after 237.8. That is the only reliable way to identify which transformations were hidden-safe and which caused the regression.

