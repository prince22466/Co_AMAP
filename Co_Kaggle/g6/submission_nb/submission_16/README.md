# Submission 16 — checked edits and reliable tool arguments

Uses **gemma-4-12b-it-qat-w4a16-ct** for all three LLM stages. Built from Submission 15;
no benchmark improvement has yet been measured. Submission 15's archived first-ten run produced
no edits, no behavioral checks, and no nonempty patches. These changes address that execution failure.

## Changes

- `args.action` is mandatory, with complete examples in each stage prompt. Missing or unknown
  arguments return a structured diagnostic instead of silently auditing an empty patch.
- Every workbench dispatch attempt is counted, including rejected repetitions. Audits are repeatable
  and remain available after the 28 non-audit attempt limit. A failed edit retains its specific diagnostic.
- `replace`/`create` require `module` and assertion `code` **before writing**. After a valid edit, the
  assertion is executed automatically in a fresh process. Failure retains the patch for repair.
  `verify` replays the persisted check. Audit ties evidence to the current patch hash and exposes the
  failing check/diagnostic. Transport `success` is never treated as a passing assertion.
- Separate role-scoped skill copies reject edit actions in the verifier and permit only audit in the
  finalizer. They are self-contained because ADK stages script bodies in temporary directories;
  the builder checks that copies differ only in their role constant.
- Scoped literal search and directory listing reduce wasted navigation. Repository `.github` workflows
  can be edited; `.git`, evaluation tests and test-runner configuration remain protected.
- All stages use `thinking_budget: 0`. In **adk-submission 0.2.13**, the compiler explicitly translates
  this into `extra_body.chat_template_kwargs.enable_thinking=false` and removes `thinking_token_budget`.
  Runtime validation checks the resulting LiteLlm objects. This mitigates the repetitive channel-token
  failures seen in Submission 15; its effectiveness with actual Gemma is still unmeasured.
- Explicit `{candidate?}` / `{verification?}` interpolation is removed. Workbench evidence, rather than
  prior prose, drives the prompts. ADK can still include a prior agent reply in current-turn context;
  this is not a claim of complete model-output sanitization.

## What is hard-enforced, and what is not

The official SequentialAgent / LoopAgent fixes stage order and allows at most two rounds:
`coder -> verifier -> coder -> verifier -> finalizer`. Only the finalizer has `submit_patch`.
The workbench enforces action validation, guarded writes, automatic post-edit assertions, role action
permissions, evidence freshness and its own work-attempt cap. Assertion quality remains the model's
responsibility. `module="-"` explicitly selects assertions for non-importable documentation/workflow/
script artifacts and does not claim module import provenance. Arbitrary check code is sandboxed;
these action restrictions are reliability controls, not a security boundary.

**Per-stage tool/time allocations and yielding after two failures remain instructions.** A script
returning an error cannot end an LlmAgent turn. Inspection of the official 0.2.11 harness showed that
`compile_submission` is called without a callback registry, so this package does not invent a custom
host callback or claim a hard per-stage cutoff. A stubborn model can still consume the shared budget.
The real-model smoke gate must demonstrate cooperative handoffs before a ten-task acceptance run.
The finalizer captures available work honestly even when unverified; no claim of guaranteed verified
submission or hidden-test success is made.

## Release and validation

From `Co_Kaggle/g6/tests_on_tasks/tools`:

```bash
python -m unittest test_submission16 test_submission16_eval -v
python build_submission16.py
python build_submission16_notebook.py
python verify_submission16_runtime.py
```

The first three commands require Python and PyYAML. The runtime command needs the competition packages:
adk-submission 0.2.13, adk-eval-core 0.1.0, google-adk 1.36.1, google-genai 2.11.0,
swegemma 0.2.11, and their CPU runtime dependencies. It uses a **scripted model**, deliberately emits
a missing-action call, recovers, writes a wrong candidate, executes a failed check, repairs in round two,
re-verifies and submits once through the real compiler, ADK Runner and sandbox skill executor.
It also checks the compiler's actual non-thinking request configuration. It is not a Gemma benchmark.

Build **`../artifacts/submission_16.zip`** with the command above, or download it from the
PR's `submission-16-12b-unscored` Actions artifact. The ZIP is generated rather than checked in;
its SHA-256 and every packaged file hash are recorded in the committed manifest.
Upload that ZIP and use
**`../../tests_on_tasks/submission_16_testing.ipynb`**:

1. Attach the ZIP as the `submission-16-zip` Kaggle dataset, competition data/runtime wheels and the
   same 12B weights. The notebook verifies every packaged file hash before model startup.
2. With Accelerator=None, run default `RUN_MODE='preflight'`. Keep the same first ten tasks; resolve
   genuine dependency or fixture failures before using GPU. No evaluator tests or scoring are rewritten.
3. Use `RUN_MODE='smoke'` on GPU to run a synthetic real-model failed-check/repair exercise. It must
   execute both rounds, make a nonempty patch, run independent verification, audit and submit once.
   Repetitive channel markers or missing evidence fail the gate and stop the server. Inspect
   `/kaggle/working/submission16_real_model_smoke/{events.json,smoke.json,synthetic.patch}`.
4. `RUN_MODE='evaluate'` repeats that smoke on the same serving stack, then evaluates the same ten tasks
   only if it passes. The acceptance run inherits **4 minutes / 40 calls / 48 turns**, with compaction
   disabled. Read `acceptance.json` and `real_model_smoke.json` alongside raw traces.
5. `RUN_MODE='replay'` grades saved patches without inference, retaining the matched baseline denominator.

The real-model gate is development-only and uses its own synthetic fixture, never competition test
patches or gold solutions as agent input. It does not modify the submitted agent's runtime behavior.
Local subprocess baseline/import fixes are inherited from v8; official Docker results may differ.
The 2 newly resolved / same 10 tasks target remains unmeasured. No new GPU run was available during PR preparation.
