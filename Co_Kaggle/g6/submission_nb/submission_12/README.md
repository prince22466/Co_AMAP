# Submission 12A: selective public-contract tracing

**Baseline:** Submission 10, public Kaggle score **0.15**. No Submission 10 per-task traces or GPU-credit-heavy task evaluations are available. Submission 11 has no reported score in `submission_records.md`. **Submission 12 is unscored.**

## Hypothesis (not an established performance gain)

Some issue resolutions fail because agents verify only a private function while the requested change must flow through a public API, a wrapper and a consumer. Prior Submission 7 diagnostic evidence includes a FastAPI issue where a lower-level check missed rejection at the APIRouter boundary. That is *historical* evidence, not a measured Submission 10 failure rate. We test one intervention only.

## Controlled changes from Submission 10

- Keep the exact Gemma 4 31B competition model, coder, two tool-free helper agents, native tools, sampling, 4-minute/40-counted-call budget, code inspection/check/audit scripts and the Submission 10 repair gate **byte-identical**.
- Add one read-only, standard-library skill script `scripts/contract_trace.py` for explicitly named Python source files. It inspects bounded AST signatures and keyword forwarding candidates, never crawls the repository and never imports task code.
- Append conditional instructions to `prompts/system.md` and `skills/repo_navigation/SKILL.md`: invoke the trace **at most once**, only for multi-layer named parameter propagation with already observed candidate paths and sufficient remaining budget; skip otherwise.
- Prefer verification through the public interface, with a normal/default behavior check, using Submission 10's existing `repair_gate.py`.

**Not a real workflow graph:** The trace reports evidence candidates, not a verified dependency graph or execution. Dynamic dispatch, renames/aliases, `**kwargs`, decorators, positional forwarding, consumer behavior and non-Python code remain uncertain. Agent invocation itself is advisory, not host-enforced. The new tool may hurt simple tasks by wasting tokens and wall time, which is why it is strictly conditional.

## CPU-only verification and packaging

Run from the repository root (requires Python 3.11+ and PyYAML 6):

```bash
python -m pip install 'PyYAML>=6,<7'
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission12.py -v
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission12.py --validate-only
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission12.py
```

Outputs are `Co_Kaggle/g6/submission_nb/artifacts/submission_12.zip` and a SHA-256 manifest. The builder validates baseline preservation, source syntax, explicit file closure, allowed extensions, ADK skill name projection (`repo_navigation` source to `repo-navigation` in ZIP), references aliases and ZIP/source parity. The PR's GitHub Actions workflow uploads the unscored candidate ZIP without using Kaggle GPU credits.

Run the separately provided `verify_submission12_runtime.py` only in an environment with the exact official ADK competition wheelhouse. Its synthetic repository, model double and native tool doubles **do not** constitute real Gemma task tests. The actual Kaggle score, resolution rate and agent compliance with the optional trace remain **unmeasured**.

## Evaluation plan when GPU is available

Freeze the candidate ZIP and compare against immutable Submission 10, ideally on the same development tasks with the real four-minute/40-counted-call budget. Report resolved cases, nonempty patches, tool calls, elapsed time, `contract_trace.py` invocation, whether a real public-interface check ran, and regressions on simple issues. Do not use historical reference patches or hidden grading answers to guide the agent; do not label synthetic tests as benchmark results.

Do not update `submission_records.md` with a score until Kaggle actually evaluates the frozen ZIP.
