# Submission 13: Gemma 4 12B with repair-controller fixes

Based on the packaged Submission 12, with `gemma-4-12b-it-qat-w4a16-ct`
for the coder and both helper agents. Preserves its prompts, tools, sampling
settings and evaluation budgets.

The graph repair controller now instruments completed public API calls in the
isolated checker, requires assertions linked to a public call or its result,
and requires enough assertions to have actually passed for the claimed IDs.
It rejects unreachable public calls and unrelated passing assertions, stops
identical internal-only checks on an unchanged patch, preserves earlier valid
public evidence across later failed checks, and labels pre-edit graph hints as
historical candidates. These checks remain conservative heuristics and do not
prove semantic coverage or hidden-test success.

The controller incorporates the corrected version already present in the
Submission 12 source tree but absent from its saved ZIP. Submission 12 itself
is unchanged. Compared with that ZIP, only the three model declarations and
`skills/repo-navigation/scripts/graph_repair_gate.py` change.

Build and verify from the `g6` directory with Python and PyYAML installed:

```powershell
python -X utf8 -m unittest discover -s tests_on_tasks/tools -p test_submission13.py -v
python -X utf8 tests_on_tasks/tools/build_submission13.py --validate-only
python -X utf8 tests_on_tasks/tools/build_submission13.py
```

Upload `submission_nb/artifacts/submission_13.zip`. The builder enforces the
approved model changes and pins the fixed controller hash, preserves every
other archive entry, and checks ZIP/source parity and integrity. The adjacent
manifest records hashes and validation results. Tests cover controller behavior,
archive reproducibility and rejection of unapproved source changes.

No official ADK runtime check, Gemma inference, Kaggle evaluation or public
score is claimed. Official runtime dependencies are not installed locally.
See Submission 12's README for the inherited pipeline and limitations.
