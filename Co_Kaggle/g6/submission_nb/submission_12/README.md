# Submission 10 (10A): bounded, evidence-driven repair loop

Experimental successor to Submission 9 (**0.10** public Kaggle score), compared against the strongest recorded Submission 7 baseline (**0.13**). **No task-resolution or Kaggle improvement is claimed.**

## What changes

Submission 10 retains Submission 9's competition model, coder, two tool-free helpers, native tools, sampling, skill and evaluation budgets. Its single experimental variable is the repair decision process:

1. The coder derives at most four stable acceptance IDs from the actual issue and implementation contract.
2. After a substantive patch, it invokes the repo-navigation skill's **self-contained** `scripts/repair_gate.py` with an executable assertion check, model-reported IDs exercised by that check, and observed remaining budgets.
3. The gate executes the same isolated assertion runner as Submission 9, records check results under the task workspace's Git metadata (not in the patch), and invalidates previous passes after changes to tracked, staged, renamed or untracked files.
4. It returns `verify_uncovered`, `repair_if_code_defect`, `alternative_check` or `audit_then_submit`. Two **actual patch-changing repair cycles** are allowed; a failed diagnostic without an edit does not consume a repair. Repeated identical failures and exhausted budgets stop the loop.
5. The coder must inspect the failure before changing code and audit the final diff before calling `submit_patch` exactly once.

The gate requires at least 45 seconds and six counted calls remaining **when these budgets are known**. A missing timer is unknown, not zero. It does not enforce the host's budgets or prevent a model from ignoring its recommendation. **Acceptance IDs and `covers` are supplied by the model and do not prove semantic test coverage; obviously vacuous assertions are rejected, but this is only a minimal safeguard.** Local checks never prove hidden-test success. This is an advisory ADK skill script, **not** a host-enforced `LoopAgent`.

## Reproducible candidate archive

From the repository root, with Python 3.11+ and PyYAML 6:

```bash
python -m pip install 'PyYAML>=6,<7'
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission10.py -v
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission10.py --validate-only
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission10.py
```

The builder enforces baseline preservation, syntax, allowed file types, skill resources, archive limits and ZIP/source parity. It projects source `skills/repo_navigation` to ADK-compatible `skills/repo-navigation`, with `references/` aliases. **Do not ZIP the raw source directory.** The output is `Co_Kaggle/g6/submission_nb/artifacts/submission_10.zip` and a SHA-256 manifest. The PR's [g6 Submission 10 GitHub Actions workflow](https://github.com/prince22466/Co_AMAP/actions/workflows/g6-submission10.yml) runs these offline tests and uploads an unscored candidate ZIP as an artifact when successful.

## Official runtime and evaluation gates

The official compiler/sandbox smoke is separate from offline unit tests. In the competition environment, install the exact wheelhouse dependencies noted in `tests_on_tasks/tools/verify_submission10_runtime.py`, then run:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/verify_submission10_runtime.py \
  --archive Co_Kaggle/g6/submission_nb/artifacts/submission_10.zip
```

The smoke uses the official ADK compiler and skill executor but synthetic source and native-tool/model doubles. It **does not** run Gemma or grade Kaggle tasks. Before treating Submission 10 as a validated candidate, check the smoke's reported SHA-256 matches the built ZIP, then run paired real-model development tasks against Submissions 7 and 9 under identical budgets. Kaggle score is recorded only after an actual submission and evaluation.

**Release checklist:** offline CI green; ZIP manifest and checksum; official compiler and `run_skill_script` smoke green; actual Gemma task-resolution testing; Kaggle submission/scoring. Missing steps must be reported as unverified, never as passed.
