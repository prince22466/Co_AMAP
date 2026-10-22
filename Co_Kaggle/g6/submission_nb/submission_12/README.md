# Submission 12B: Integrated graph-guided repair controller

**Scored baseline:** Submission 10, public Kaggle score **0.15**. This candidate is unscored. Submission 10 task-level traces are not available and Kaggle GPU credits are scarce.

This PR was revised from 12A's advisory AST tracer into a **coupled code-dependency graph + evidence-driven repair loop**, while retaining Submission 10's exact allowed Gemma 4 31B model, tool/agent topology, sampling and 4-minute/40-call task budget.

## Pipeline

1. For an applicable *multi-layer Python named-option change only*, the coder discovers up to 6 candidate files through normal bounded inspection. With >=150 s and >=14 calls remaining, it invokes `contract_trace.py` once with the named public entry, 1–4 stable requirement IDs and `store=true`. The script parses the named files (does not import/execute target code), records bounded API nodes, static call candidates, forwarding modes, and file hashes under per-task Git metadata, **outside the implementation diff**.
2. The coder implements the change. After editing, `graph_repair_gate.py` reads the saved graph, executes the same isolated assertion checker as Submission 10, and records task evidence using the same bounded repair loop and Git metadata. If the graph is missing or does not match the requirements, it falls back to the Submission 10 gate unchanged.
3. A passing **internal-only** check does **not** mark a public API requirement verified. The script checks that the public entry is actually called by the executable check and that at least one assertion exists per claimed requirement, otherwise returns `verify_public_api`. This is a conservative syntactic safeguard, not proof of semantic coverage.
4. For failures it reports candidate graph focus (missing keyword signature or potentially unforwarded parameter) as an **investigation hypothesis**, not an implementation verdict. After source edits, stale graph hints are downgraded. The controller retains the maximum two actual patch-changing repair cycles, one inconclusive alternative and budget guard; on insufficient budget it stops optional verification and directs audit/submit.
5. All non-Python tasks, simple bugs, low-budget tasks or unsupported graphs use Submission 10's existing repair gate. A model must invoke the scripts and respect the recommendations; **this is not an ADK LoopAgent or host-enforced orchestration**.

## Known limitations

Static graph candidates omit unresolved dynamic dispatch, aliases, decorators, positional forwarding, deep consumers and runtime dependency injection. Caller inference is deliberately conservative. Graph labels and model-supplied tests do not prove hidden test coverage, and a model could still fail to invoke the graph tools. File signatures are hashed to flag stale structural information after changes. The graph is stored outside submitted source, and neither graph scripts nor check scripts edit task implementation files.

## Reproduce with CPU-only GitHub CI

```bash
python -m pip install 'PyYAML>=6,<7'
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission12.py -v
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission12.py --validate-only
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission12.py
```

The deterministic builder enforces baseline preservation and explicitly allowlisted added scripts, source syntax, ADK skill-name/reference projection and ZIP integrity. GitHub Actions uploads `submission_12.zip` plus `submission_12.manifest.json` as an unscored candidate. Old ZIP checksums from 12A are not valid for this revised artifact.

For an optional official wheelhouse ADK compiler and skill runtime check, run `tests_on_tasks/tools/verify_submission12_runtime.py` in the competition environment against the exact new ZIP. That script uses synthetic source and test doubles, not Gemma inference. A true Kaggle resolution improvement versus the immutable 0.15 baseline **remains unverified**; do not use synthetic test results as Kaggle evidence.
