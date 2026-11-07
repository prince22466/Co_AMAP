# E4B capability ladder — research-only diagnostic

Goal: separate *ability to propose a fix*, *ability to express a valid tool
operation*, and *ability to navigate and repair autonomously*. This is **not**
Submission 37, not a Kaggle agent package, and does not change S29 or grading.

## Three levels, matched task IDs

| Level | Inputs | Model surface | Output | Verification |
| --- | --- | --- | --- | --- |
| L1: reasoning | Issue, hints, pristine source excerpt at the reference-patch location | One direct E4B chat response, no tools; returns edit intent as JSON | One bounded exact-string edit → unified diff | Official phase-2 verifier, separately |
| L2: tool syntax | **Same** issue and pristine excerpts | One E4B tool call with edit_file schema (one turn), parsed by the active vLLM tool parser | Host emulates a strict exact, unique edit → unified diff | **Same** official phase-2 verifier |
| L3: autonomy | Issue and repository, **without location oracle** | Unmodified S29 six native tools + existing ADK/harness context handling | Submitted git patch | Existing official S29 evaluator |

L1 and L2 use **oracle localization** obtained by reading *only* reference
patch file headers and hunk original line numbers, then reading source from
the **unmodified** task snapshot. NO reference added/deleted lines, no
reference patch body, and NO test_patch are sent to E4B. It is *not*
valid to treat L1/L2 as competition scores. Raw public tasks and reference
patches must remain outside a submission. The result quantifies what changes
when the model is handed the right source location, not ordinary agent skill.

L1 and L2 are deliberately **one-turn, single-edit** probes. They cannot
establish the true maximum E4B capability, especially on multi-file tasks.
L2 uses an OpenAI-compatible function-call request with the native
edit_file argument shape, but a **stricter host-side replacement executor**;
it is not the ADK tool implementation. L1 is free-form JSON, while L2 uses
tool syntax; those output formats also introduce a confound. Note both
limitations in all conclusions.

## Setup

Use the exact E4B QAT model, same vLLM server and task data/wheelhouse across
the three arms. vLLM needs Gemma 4 tool-choice support for L2 (typically the
Gemma 4 tool-call parser with auto-tool-choice enabled).

Use fresh paths for every run and keep all evidence; the tools refuse to
overwrite an output directory.

From Co_Kaggle/g6/tests_on_tasks/capability_ladder:

### 1. Generate assisted edits (GPU model server, no grader required)

~~~bash
python probe.py --level reasoning --stage pilot \
  --tasks /data/tasks.jsonl --snapshots /data/snapshots \
  --model-url http://localhost:8000/v1 \
  --output /work/cap_l1_seed1

python probe.py --level tool --stage pilot \
  --tasks /data/tasks.jsonl --snapshots /data/snapshots \
  --model-url http://localhost:8000/v1 \
  --output /work/cap_l2_seed1
~~~

Default five-task pilot matches the existing S29 official pilot IDs:
rich_4077, rich_3905, rich_3518, rich_3454, rich_3882. Use
--stage s1 for the 15 historic S1 tasks. For fresh tasks, use the SAME
explicit --task-ids value in **all three** arms, and choose them *before*
seeing scores (ideally include Requests and FastAPI). Use --seed 2 with a
new output directory for another run. Preserve the served model, sampling,
runtime and task hashes per run.

Probe results: manifest.json, probes.jsonl (one row per task with the
generated patch or failure reason), and raw_model/<id>.json (actual response
for diagnosing malformed or truncated tool calls). A successful edit produces
edit_valid=true and resolved=null; it is NEVER counted as a correct fix yet.

The direct request uses temperature 0, top-p 0.95, max_tokens 4096, and a
fixed seed. **It does not reproduce ADK thinking-budget/compaction settings**:
this is a purposefully isolated API probe, not a strict model-config ablation.

### 2. Verify assisted patches independently (CPU/Docker; no GPU model calls)

Install the same organizer wheelhouse and provision a working official
verification Docker environment. The verifier is based on the published
swegemma verify_task interface and is imported only when this step runs.
If your installed wheel has a different signature, it must fail as a
BLOCKED verifier—not silently use an invented scorer.

~~~bash
python verify.py --probes /work/cap_l1_seed1/probes.jsonl \
  --tasks /data/tasks.jsonl --snapshots /data/snapshots \
  --submission-dir ../../submission_nb/submission_29 \
  --output /work/cap_l1_verified_seed1

python verify.py --probes /work/cap_l2_seed1/probes.jsonl \
  --tasks /data/tasks.jsonl --snapshots /data/snapshots \
  --submission-dir ../../submission_nb/submission_29 \
  --output /work/cap_l2_verified_seed1
~~~

For **each** task, official verification executes all three independent
checks on a fresh baseline:

1. Unchanged repository must FAIL relevant tests meaningfully.
2. The published reference fix must PASS the same evaluator.
3. Only then is the generated patch graded as resolved or unresolved.

Reference fixes and test patches exist **only inside phase-2 verification**
and are not provided to the model. If either control is invalid, the
result is blocked_verifier, with resolved=null, excluded from denominators.
Never label a dependency/fixture error as a model failure. The reference
fix is not the candidate and is never an output solution.

This verification bridge deliberately requires Docker-compatible official
wheels. It has not been checked against every Kaggle wheelhouse revision;
check verified.jsonl status and the verifier-source digest in manifest.json.
Official verification may take substantially longer than generation.

### 3. Run true autonomous S29 (existing official harness)

~~~bash
python run_autonomy.py --stage pilot --repeats 2 --time-limit 4 \
  --tasks /data/tasks.jsonl --snapshots /data/snapshots \
  --wheels /data/wheels --output /work/cap_l3
~~~

This rebuilds the *unmodified S29 package*, uses its original prompt/tools,
and delegates to the existing run_submission24_eval.py baseline preflight,
sandbox and verifier. It does not add oracle context. Inspect the
baseline-blocked task list and task_results; do not treat a failed baseline
as evidence of inability.

If running held-out tasks, pass the same explicit --task-ids list to
run_autonomy.py, probe.py L1, and probe.py L2. The S1 and pilot presets
are matched. A 4-minute run is comparable to intended agent budgets; a
0-minute (unlimited) test is a separate experiment and must not be pooled.

### 4. Compare only matched scorable tasks

~~~bash
python report.py \
  --reasoning /work/cap_l1_verified_seed1/verified.jsonl \
  --tool /work/cap_l2_verified_seed1/verified.jsonl \
  --autonomous /work/cap_l3/candidate_1/notebook_task_results.jsonl \
  --output /work/cap_ladder_report_seed1.json
~~~

The report rejects missing/duplicate IDs. It counts blocked cases separately.
Interpretation of *candidate* bottlenecks:

- L1 verified success but L2 failure: likely tool-expression/dispatch issue;
  inspect the raw tool call and exact replacement failure.
- L2 verified success but L3 failure: likely navigation, context, workflow,
  or different-generation-regime issue; inspect full ADK trace.
- L1/L2 failures: no proof of model incapacity; this is a one-edit upper
  assistance probe, not an omniscient oracle or a multi-step solver.
- All levels solve: a retained success; no immediate need to modify it.
- L1 false, L2 true: prompts/sampling or interface requirements confound
  a clean causal attribution.

Compare multiple matched repeats and held-out issues. Do not pick the best
seed or treat five or fifteen tasks as statistically conclusive.

## Test / safety contract

~~~bash
python -m py_compile probe.py verify.py report.py run_autonomy.py test_ladder.py
python -m unittest -v test_ladder.py
~~~

CPU tests cover raw-source-only oracle generation (canary added to the
reference fix/test patch must not appear in the prompt), traversal/missing
file rejection, unique and bounded replacements, malformed tool calls,
no-premature-resolution gating, reference/negative verification controls,
and exact paired task membership.

Neither this directory nor the generated data is added to submission.zip.
Never put task reference patches in model-facing agent instructions,
callbacks, optional skills, or input messages. Data inspection and grading
must happen outside the submitted agent. Probes may contain task prompts
and model outputs; store results privately and review them before publishing.

## Stop/go criteria

Make a decision from repeated, environment-valid evidence:

* If L1 achieves many more *verified* resolutions than L3, there is
  worthwhile headroom in retrieval/interface/workflow engineering.
* If L1 resolves many but L2 loses a large fraction, prioritize reliable,
  bounded tool serialization and exact-edit semantics.
* If L2 succeeds but L3 loses, investigate navigation/feedback and attention
  before any more prompt polishing.
* If all arms struggle despite verified reference controls, this *suggests*
  a model or constrained-probe capability ceiling; test a broader assisted
  probe before claiming a fundamental limitation.

**Do not build S37 automatically. This PR supplies the measurement system,
not a promised scoring improvement or a completed GPU experiment.**
