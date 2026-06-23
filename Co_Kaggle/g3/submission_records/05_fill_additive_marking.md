# Fill / Additive Marking Submission Score Record

This file records effective submissions for the `fill_enclosed_regions` task family. Use it to connect notebook-visible metrics with Kaggle leaderboard movement and decide the next improvement direction.

Family label: `fill_enclosed_regions`  
Notebook: `submission_nbs/05_fill_additive_marking.ipynb`  
Task count in current map: `59`  
Theoretical max for this family: `59 * 25 = 1475`

## How To Use

For each meaningful submission, add one row to the summary table and one detail block below it.

Only record submissions that teach something:

- new solver version
- new subset of tasks
- better visible accuracy
- lower parameter or memory cost
- better public leaderboard score
- worse public score that reveals overfitting or private-set weakness

## Summary Table

| Date | Submission ID | Version | Tasks Included | Visible Pass Tasks | Public LB Score | Public LB Delta | Est. Family Points | Main Change | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| YYYY-MM-DD | pending | `fill-additive-v0.1` | 0 / 59 | 0 / 59 | TBD | TBD | TBD | Baseline notebook scaffold, no real solver yet | Do not submit unless using placeholders intentionally |

## Detail Template

Copy this block for each effective submission.

```text
## Submission YYYY-MM-DD-N

Submission ID:
Kaggle file:
Notebook:
Model version:
Git/workspace note:

### Task Coverage

Family:
Mapped tasks:
Submitted task models:
Tasks changed from previous submission:
Tasks removed from previous submission:

Task selection rule:

### Model / Architecture

Architecture family:
ONNX ops:
Node count:
Initializer shapes:
File size bytes, min/avg/max:
Parameter count, min/avg/max:
Static memory bytes, min/avg/max:
Runtime memory bytes, min/avg/max:
Estimated cost = params + memory, min/avg/max:

### Visible Performance

Train exact match:
Test exact match:
ARC-GEN exact match:
Visible all exact match:

Failed visible tasks:
Common failure pattern:

### Kaggle Result

Public LB score:
Public LB delta vs previous best:
Estimated private-risk:

### Interpretation

What improved:
What regressed:
Likely reason:
Next experiment:
Keep / revert / modify:
```

## Submission Details

## Submission 2026-06-23-001

Submission ID: `not_submitted`  
Kaggle file: none  
Notebook: `submission_nbs/05_fill_additive_marking.ipynb`  
Model version: `fill-additive-v0.1`  
Git/workspace note: prototype reporting scaffold created

### Task Coverage

Family: `fill_enclosed_regions`  
Mapped tasks: `59`  
Submitted task models: `0`  
Tasks changed from previous submission: N/A  
Tasks removed from previous submission: N/A

Task selection rule:

```text
primary_family == "fill_enclosed_regions"
```

### Model / Architecture

Architecture family: pending  
ONNX ops: pending  
Node count: pending  
Initializer shapes: pending  
File size bytes, min/avg/max: pending  
Parameter count, min/avg/max: pending  
Static memory bytes, min/avg/max: pending  
Runtime memory bytes, min/avg/max: pending  
Estimated cost = params + memory, min/avg/max: pending

### Visible Performance

Train exact match: pending  
Test exact match: pending  
ARC-GEN exact match: pending  
Visible all exact match: pending

Failed visible tasks: pending  
Common failure pattern: pending

### Kaggle Result

Public LB score: not submitted  
Public LB delta vs previous best: N/A  
Estimated private-risk: high until real solver passes visible examples

### Interpretation

What improved:

- The notebook now shows the right information for solver development: selected tasks, version, architecture, parameters, memory profile, split performance, and submission packaging.

What regressed:

- No model is generated yet for this family.

Likely reason:

- `train_family_task` is still a placeholder for fill/additive marking.

Next experiment:

- Implement a first narrow solver for tasks where all nonzero input cells are preserved and output only adds one new color to background cells.
- Start with a small subset such as `task002`, `task027`, `task042`, `task081`, and `task095`.
- Record visible pass rate and model cost before any Kaggle submission.

Keep / revert / modify:

- Keep the reporting scaffold.
- Modify only the solver implementation cell in `05_fill_additive_marking.ipynb` for the next prototype.

## Improvement Directions

Prioritize prototypes in this order:

1. **Single-added-color local marking**
   - Candidate signal: one new output color, input nonzero preserved ratio near `1.0`.
   - Likely ONNX approach: small local convolution.
   - Goal: pass tasks where local neighborhoods determine added cells.

2. **True enclosed-region fill**
   - Candidate signal: boundary color preserved, background cells inside boundary become fill color.
   - Likely ONNX approach: multi-layer propagation or bounded local fill approximation.
   - Goal: start with `task002`-like cases.

3. **Template-based additive marking**
   - Candidate signal: output adds cells in repeated or symmetric positions.
   - Likely ONNX approach: fixed convolution plus thresholding.

4. **Multi-color additive marking**
   - Candidate signal: multiple new output colors.
   - Higher risk; handle after single-added-color solvers.

## Metrics To Watch

For every effective submission, compare:

- `visible_accuracy`
- `train_accuracy`
- `test_accuracy`
- `arc_gen_accuracy`
- `params`
- `runtime_memory_bytes`
- `static_memory_bytes`
- `file_size_bytes`
- public leaderboard score delta

Good direction:

```text
visible pass tasks increase
public LB score increases
params + memory stay small
arc-gen remains strong
```

Warning signs:

```text
visible train/test improves but arc-gen fails
public LB drops after adding risky tasks
model cost increases without solving more tasks
many tasks pass only because of over-specific constants
```
