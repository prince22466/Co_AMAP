# Submission 10: budget-aware evidence-driven repair loop

Experimental successor to Submission 9 (**0.10** public score), with Submission 7 (**0.13**) retained as the strongest recorded baseline. **No Kaggle score or Gemma task-resolution improvement is claimed.**

The main coder and both helper agents, model, native tools, skill binding, sampling and evaluation budgets are retained from Submission 9. This experiment adds `skills/repo_navigation/scripts/repair_gate.py` and corresponding prompt/skill guidance.

After the first substantive patch, the coder declares 1–4 stable acceptance IDs, supplies an executable assertion check and lists only the requirements the check exercises. The gate invokes the existing `check.py`, records evidence in the workspace Git metadata (not the submitted diff), invalidates passing evidence after any edit, and returns one of four decisions:

- `verify_uncovered`: a declared requirement lacks passing executable evidence.
- `repair_if_code_defect`: failed check, with at most two repair requests; inspect cause before editing.
- `alternative_check`: one smaller check after an inconclusive result.
- `audit_then_submit`: acceptance checks covered, repeated unchanged failure, exhausted repair/alternative limit, or low budget.

The gate requires 45 seconds and six counted calls remaining to start a check when budget values are available; otherwise it directs final audit and capture. Unknown budget values are not treated as zero. A local check never establishes hidden-test success. This is an advisory, deterministic decision script inside an ADK skill, **not a native ADK LoopAgent or a host-enforced controller**. The model must invoke it and obey its result. The gate does not edit code or call `submit_patch`.

## Packaging and testing

Use the Submission 9 deterministic release builder as the starting point. It must be parameterized for Submission 10 (candidate path, preserved baseline, output archive and manifest), and package `skills/repo_navigation` as ADK-compatible `skills/repo-navigation` with `references/` aliases. Do not manually ZIP the source directory: the raw underscore skill path is not accepted by the documented ADK runtime. The ZIP, manifest, compiler smoke, real model evaluation and Kaggle submission are **not yet produced** by this PR.

The gate's initial nine synthetic tests covered all-declared passing checks, partial coverage, invalidation after edit, repeated failures, two-repair limit, low budgets, inconclusive checks, invalid IDs and immutable requirement IDs. Those tests used a local checker double and are not competition-resolution results. Before release, run the unchanged real checker integration, ADK runtime smoke and paired development tasks against Submissions 7 and 9. Do not promote this candidate based only on static or synthetic tests.

No change to `submission_records.md` is made until an actual Kaggle score exists.
