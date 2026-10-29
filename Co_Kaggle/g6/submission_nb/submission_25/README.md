# Submission 25: one repair agent with checked edits

**Candidate for real E4B testing; performance improvement is not yet measured.**
Uses `gemma-4-e4b-it-qat-w4a16-ct`, no LoRA, the official declarative compiler,
and sandboxed skill scripts. Upload [submission_25.zip](../artifacts/submission_25.zip)
to the same testing notebook used for S24. S24 and its historical ZIP are unchanged.

## Why change the architecture

[S24's S1 diagnostics](../../tests_on_tasks/diagnostics) show 1/15 solved, three empty
patches, five turn-limit errors, and 49.6 minutes of recorded evaluation time.
S21/S22 each solved 2/15. More reviews did not yield better fixes: a second editor
damaged an initially plausible repair, reviewers approved missing APIs and
docstring-only changes, and identical failed edits consumed whole runs.

S25 removes the locator/editor/reviewer/finalizer chain and two-round loop. One
agent retains its tool evidence, edits, checks, and submits. Thinking is disabled
and output is limited to 4,096 tokens to reduce the long malformed responses seen
in S24. This is a testable resource-use hypothesis, not an isolated ablation result.

| Observed failure | S25 change | Enforcement |
| --- | --- | --- |
| Repeated ambiguous or over-escaped old strings | Numbered whole-line edits from a fresh read; no `old_string` | Script checks read coverage and current file hash |
| Repeated guard insertion / bad indentation | Parse the entire candidate Python file before writing | Syntax errors leave source unchanged |
| Later edit breaks a passing repair | Rerun up to two saved passing checks during later edits | Restore the previous file if any check fails |
| Repeated identical failed edits | Reject duplicate failed arguments against the same source/read; 12 processed attempts maximum | Script; does not stop model turns spent elsewhere |
| Guessed test paths, zero tests, malformed shell probes | Existing pytest file/node validation; Python source probes; `check` reuses exact input | Exit code, JUnit cases, executed assertion and local activity recorded |
| Reviewer says a field exists but only docs changed | Show actual diff, substantive AST changes, and check freshness | No separate model-generated READY label |
| A failed proof gate leaves an empty submission | Accept structurally valid partial fixes and keep native `submit_patch` | No mandatory semantic publication gate |

Only `get_status`, `submit_patch`, and the standard ADK skill tools are exposed.
There is one skill name and one script path, with named scalar arguments in an
object. Unlike S20, there is no private candidate checkout, copy/publish step,
or mandatory fail-to-pass condition before source can reach the evaluator.

## Limits that still matter

- **Skill dispatch remains a risk.** S17 showed E4B can miscall wrappers. The
  canonical invocation is in the main prompt, and malformed-wrapper recovery is
  tested with a scripted model; real E4B dispatch must pass the pilot.
- Correct localization, complete API plumbing, relevant assertions, and stopping
  still depend on E4B. A passing probe must have an executed assertion linked
  to observed repository imports, but that heuristic cannot establish task correctness.
- Edits are limited to existing **Git-tracked** implementation `.py` files, at most 100 observed
  lines per edit and 1 MB per file. New-file and packaging changes are unsupported.
- Focused checks have a six-second limit; long/network-dependent tests may be
  blocked. That retains the source patch rather than producing an empty result.
- Checks restore changes to tracked files and new non-ignored files. Python still
  runs inside the existing sandbox: this is a cooperative reliability mechanism,
  not protection against malicious code or changes to Git metadata/ignored files.
- Interrupted edits have a recovery journal, recovered on the **next workbench
  call**. Native submission/automatic harness fallback does not invoke recovery.
  The two six-second guard checks normally fit inside the 30-second skill limit.
- Budgets remain 40 charged tools / 48 turns / four minutes. Historical S1 runs
  disabled the task timer; use the same override for the initial comparison.

## Validation and testing

From the repository root, after installing the official runtime:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/install_submission24_runtime.py --wheel-dir /tmp/g6-wheels
python -m pip install pytest==8.4.2
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p 'test_submission25*.py' -v
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission25.py
```

The deterministic ZIP contains only five agent/config/skill files. Test fixtures,
task IDs, diagnostics, reference fixes, evaluation tools and this README stay out.
See [validation scope](../../tests_on_tasks/reviews/submission_25/validation.md).

Use an existing competition E4B server and mounted task data for the first pilot:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/run_submission25_eval.py \
  --tasks /path/to/tasks.jsonl --snapshots /path/to/snapshots --wheels /path/to/wheels \
  --model-url http://localhost:8000/v1 --served-model gemma-4-e4b-it-qat-w4a16-ct \
  --stage pilot --output /path/to/s25-pilot
```

If `/v1/models` exposes the model as an absolute path, pass that exact ID as
`--served-model`. `LOCAL_API_KEY` is used when set. The runner uses the same
reviewed evaluator, cache/compaction and effective 40/48 budgets as S24, retains
setup failures, and checks empty-patch baselines before inference. It never starts
a GPU server or automatically launches a larger run.

Pilot: `rich_4077`, `rich_3905`, `rich_3518`, `rich_3454`, `rich_3882` (zero-based
notebook indices `[19, 75, 107, 113, 72]`). Promotion requires at least 3/5 solved,
retaining `rich_4077`, at most one empty patch, and no agent errors in every repeat.
This is a development gate. If dispatch/editing still fails, stop and inspect
those traces before spending GPU time on the whole set.

Then run `--stage s1` (prefer `--repeats 2`) with fresh output directories. Compare:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/compare_submission25.py \
  --s21 /path/to/s21.zip --s22 /path/to/s22.zip --s24 /path/to/s24.zip \
  --s25 /path/to/s25-s1/candidate_1
```

The predeclared practical target is **at least 6/15**, five net gains over S24,
retaining its win and at least three S21/S22 union wins, at most two empty patches,
and no agent errors. Repeat matched controls and inspect time/tokens as well as
score. This target is neither statistical significance nor a hidden-score claim;
use additional tasks outside this diagnostic set before claiming generalization.
