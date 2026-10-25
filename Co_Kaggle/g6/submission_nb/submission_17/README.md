# Submission 17: Submission 16 with Gemma 4 E4B

Uses `gemma-4-e4b-it-qat-w4a16-ct` for coder, verifier and finalizer.
Only the three model declarations differ from the packaged Submission 16.
Prompts, workflow, skills, tools, sampling and evaluation budgets are unchanged.
See [Submission 16](../submission_16/README.md) for the inherited design and limitations.

Build from the `g6` directory with Python and PyYAML installed:

```powershell
python -X utf8 tests_on_tasks/tools/build_submission17.py --validate-only
python -X utf8 tests_on_tasks/tools/build_submission17.py
```

Upload `submission_nb/artifacts/submission_17.zip`. The builder checks exact
source and archive parity with Submission 16 except for the three model changes.
The adjacent manifest records the model, checksums and changed files.

No model inference, Kaggle evaluation or public score has been measured for this version.
Submission 16's testing notebook pins its 12B model and release hashes; it cannot
be used unchanged with this E4B package.
