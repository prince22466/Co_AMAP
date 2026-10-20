# Submission 9: submission 7 with an ADK repair skill

**Candidate, not yet scored.** Submission 7 remains the confirmed 0.13 public-score baseline; Submission 8 scored 0.12. See the [diagnostic review](../../tests_on_tasks/diagnostics/submission_9_review/review.md) for evidence and limits. No improved resolution rate is claimed.

The requested competition package layout is registered on the coder:

```text
agent.yaml
skills/
  repo_navigation/
    SKILL.md
    scripts/
      inspect_source.py
      check.py
      audit_patch.py
    resources/
      contracts.md
      tool_recovery.md
      verification.md
```

```yaml
skills:
  - skills/repo_navigation
```

**Build and use the ZIP, or extract that ZIP for directory-based evaluation.** The source layout above is retained as requested. The official ADK 1.36.1 loader rejects underscores by default and does not load `resources/`. The builder therefore packages the skill under `skills/repo-navigation`, adjusts the agent binding, and creates byte-identical `references/` aliases from the three canonical `resources/` files. The manifest name is `repo-navigation`, matching the packaged directory. Do not evaluate the raw source directory or ZIP it manually.

These are competition ADK skills, not a ChatGPT skill installation or a Python agent entry point. Scripts are Python because the supplied harness extension allowlist includes `.py`, not `.sh`. They execute through sandbox skill tools; no callbacks, custom host imports, extra model, network access or adapters are added.

## Changes and preservation

| Component | Behavior |
| --- | --- |
| `agent.yaml` | Adds the skill binding; original native tools and helper connections remain. |
| `prompts/system.md` | Adds one skill-routing paragraph; all submission-7 instructions remain in order. |
| `SKILL.md` | Contract, localization, recovery, verification and final-audit checkpoints. |
| `inspect_source.py` | Finds qualified Python definitions without importing code; returns bounded source and the next unread line. |
| `check.py` | Runs assertions in a fresh process using workspace imports; counts executed inline assertions, detects caught failures and disables inherited optimization. Uncalled checks cannot pass. |
| `audit_patch.py` | Reports protected edits, scratch candidates, syntax errors, conflict markers and whitespace problems (including new files). Excludes only the active ADK wrapper/materialization; never stages or modifies files. |
| `resources/` | On-demand guidance for API propagation, error conventions, tool recovery and reliable verification. |

All agents keep `gemma-4-31b-it-qat-w4a16-ct`. The other eight active files are byte-identical to submission 7: both helpers' YAML/prompts/sampling, coder sampling, and evaluation configuration. Budgets remain four minutes, 40 counted calls, 80 turns and 60 seconds per command. Helpers retain their roles and no tools. Submission 8's shell-only coder and new reviewer are not adopted.

The scripts assist the coder; they do not automatically generate fixes. The coder must interpret evidence, edit code, inspect the actual diff and submit. Skill activation and compliance remain model-driven. Extra skill interactions can hurt simple tasks, so the procedures replace equivalent native calls and load only relevant resources.

## Build and local verification

From repository root, using Python 3.10+ and PyYAML:

```bash
python -B -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission9.py -v
python -B Co_Kaggle/g6/tests_on_tasks/tools/build_submission9.py
python -B Co_Kaggle/g6/tests_on_tasks/tools/review_submission9_evidence.py --output Co_Kaggle/g6/tests_on_tasks/diagnostics/submission_9_review/evidence.json
```

The deterministic [ZIP](../artifacts/submission_9.zip) contains 20 active files with `agent.yaml` at its root: 17 source files plus three generated reference aliases. Its [manifest](../artifacts/submission_9.manifest.json) records the conversion, hashes and baseline preservation. README, tests, diagnostics and build tools stay outside the archive. The builder rejects symlinks, unsafe includes, missing resources, unsupported release extensions and task IDs in the active package.

Twenty-eight synthetic tests pass, including uncalled/caught/optimized assertions, source metadata overflow, untracked whitespace, non-Python conflict markers, dangling symlinks, import provenance, timeout, read-only audit, unchanged configuration and deterministic packaging. These are script/packaging tests, **not Gemma task-resolution tests**.

## Official runtime verification

The built ZIP passes eight CPU smoke checks using official `adk-submission 0.2.13`, `google-adk 1.36.1`, `adk-eval-core 0.1.0` and `google-genai 2.11.0` wheels from the [competition wheelhouse, version 31](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse/versions/31). The checks compile the candidate, activate the skill, load all three resources, execute every script through ADK's real runpy/subprocess bridge, verify workspace imports and caught failures, and audit both clean and edited repositories. The clean-repository check also exercises a workspace-local TMPDIR.

The [runtime report](../../tests_on_tasks/diagnostics/submission_9_review/runtime_smoke.json) identifies the exact archive hash, installed versions and test scope. Native tools and the model are test doubles; the compiler, skill tools and sandbox executor are real. Other dependencies are locally resolved and recorded, not a full frozen Kaggle image. No Gemma inference, benchmark grading, Docker execution or competition budget accounting has been tested. To reproduce in a Python 3.12 environment with those wheelhouse packages installed:

```bash
python -B Co_Kaggle/g6/tests_on_tasks/tools/verify_submission9_runtime.py
```

Use the existing [v5 testing notebook](../../tests_on_tasks/gemma-4-developer-agent-testing_v5.ipynb) with the release ZIP (or its extracted directory) and 31B model. Before a long run, check on one development task that the model activates the skill, passes the workspace observed through `pwd -P`, interprets JSON outcomes, respects counted-call/time limits and captures a nonempty patch. Recheck compatibility if the wheelhouse changes.

The routing paragraph falls back to native tools after one corrected skill failure. A run that always falls back is not evidence for skills. This submission makes no changes to the scorer or harness.

## Paired evaluation

Freeze baseline/candidate ZIP hashes. Compare unchanged submission 7 with 9 on identical tasks, runtime and settings. Set `EVAL_OVERRIDES['max_time_minutes'] = INHERIT` for release-budget runs; historical diagnostics disabled that timer. Track paired wins/losses, rejected calls, repeated inspections, skill use, import origins, protected edits, nonempty implementation patches and total generation/setup time, including the competition's 12-hour limit. Report grading environment failures separately.

Rerun the same 30 development tasks first, including previous passes as regression checks. Then use an unseen split excluding previously inspected task IDs and base-commit groups. Older holdout manifests referenced by baseline READMEs are absent at this source commit, so this PR does not falsely label a new split as historically untouched. Multiple paired runs help assess stochastic variation.

After the one-task model gate, a procedure-only skill versus the full scripts variant can isolate the contribution of scripts. This PR ships one candidate. Promote it only when paired results and an official score support doing so.
