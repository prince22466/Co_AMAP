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

The manifest name is `repo_navigation`, matching the directory. These are competition ADK skills, not a ChatGPT skill installation or a Python agent entry point. Scripts are Python because the supplied harness extension allowlist includes `.py`, not `.sh`. They execute through sandbox skill tools; no callbacks, custom host imports, extra model, network access or adapters are added.

## Changes and preservation

| Component | Behavior |
| --- | --- |
| `agent.yaml` | Adds the skill binding; original native tools and helper connections remain. |
| `prompts/system.md` | Adds one skill-routing paragraph; all submission-7 instructions remain in order. |
| `SKILL.md` | Contract, localization, recovery, verification and final-audit checkpoints. |
| `inspect_source.py` | Finds qualified Python definitions without importing code; returns bounded source and the next unread line. |
| `check.py` | Runs assertions in a fresh process using workspace imports; checks the named module's origin and distinguishes failure, timeout, early exit and print-only checks. |
| `audit_patch.py` | Reports protected edits, scratch candidates, syntax errors and whitespace problems without staging or modifying files. |
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

The deterministic [ZIP](../artifacts/submission_9.zip) contains 17 active files with `agent.yaml` at its root. Its [manifest](../artifacts/submission_9.manifest.json) records hashes and baseline preservation. README, tests, diagnostics and build tools stay outside the archive. The builder rejects symlinks, unsafe includes, missing resources, unsupported release extensions and task IDs in the active package.

Twenty synthetic tests pass: import provenance, assertion failure, print-only checks, early exit, runtime exceptions, timeout, output bounds, temporary working directories, source pagination, path confinement, protected edits, scratch files, read-only audit, unchanged configuration and deterministic packaging. These are script/packaging tests, **not Gemma task-resolution tests**.

## Official runtime gate

This machine lacks `adk-submission`, `swegemma`, the competition wheelhouse, weights and snapshots. Full official compilation and Gemma inference have not run here. Static validation is not official harness acceptance.

Use the existing [v5 testing notebook](../../tests_on_tasks/gemma-4-developer-agent-testing_v5.ipynb) with the mounted submission-9 directory/ZIP and 31B model. Before a long run, check on one development task:

1. The compiler accepts the `skills` binding and matching snake_case manifest name, and exposes the skill tools. Use their actual schemas; do not add skill tools manually to the closed native-tool registry.
2. The manifest and one `resources/` file load. The competition guide uses `resources/`, while upstream Google's generic skill loader uses `references/`/`assets/`. Verify the installed competition bridge; correct any mismatch before measuring this intervention.
3. Obtain `pwd -P` with `run_command` and pass it as `workspace`. Skill scripts can start in a temporary directory, and local subprocess sandboxes may map `/workspace` elsewhere. Scripts never search for another workspace themselves.
4. An inspection and assertion check see the same repository as native tools; the final audit sees the same diff. Read JSON `outcome`, not merely the outer tool status.
5. Skill calls/time/budget accounting appear in traces and a nonempty patch reaches grading.

The routing paragraph falls back to native tools after one corrected skill failure. A run that always falls back is not evidence for skills. This submission makes no changes to the scorer or harness.

## Paired evaluation

Freeze baseline/candidate ZIP hashes. Compare unchanged submission 7 with 9 on identical tasks, runtime and settings. Set `EVAL_OVERRIDES['max_time_minutes'] = INHERIT` for release-budget runs; historical diagnostics disabled that timer. Track paired wins/losses, rejected calls, repeated inspections, skill use, import origins, protected edits, nonempty implementation patches and total generation/setup time, including the competition's 12-hour limit. Report grading environment failures separately.

Rerun the same 30 development tasks first, including previous passes as regression checks. Then use an unseen split excluding previously inspected task IDs and base-commit groups. Older holdout manifests referenced by baseline READMEs are absent at this source commit, so this PR does not falsely label a new split as historically untouched. Multiple paired runs help assess stochastic variation.

After runtime compatibility is established, a procedure-only skill versus the full scripts variant can isolate the contribution of scripts. This PR ships one candidate. Promote it only when paired results and an official score support doing so.
