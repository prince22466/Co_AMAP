# Submission 14: Submission 13 with Gemma 4 E4B

Uses `gemma-4-e4b-it-qat-w4a16-ct` for the coder and both helper agents.
Compared with the fixed, packaged Submission 13, only the three model
declarations change. Prompts, repair-controller fixes, skills, tools,
sampling settings and evaluation budgets are preserved.

Build from the `g6` directory with Python and PyYAML installed:

```powershell
python -X utf8 tests_on_tasks/tools/build_submission14.py --validate-only
python -X utf8 tests_on_tasks/tools/build_submission14.py
```

Upload `submission_nb/artifacts/submission_14.zip`. The builder checks source
preservation and exact archive parity with Submission 13 except for the three
model substitutions. The adjacent manifest records checksums and validation.
See Submission 13's README for the inherited controller fixes and limitations.

No official ADK runtime check, model inference, Kaggle evaluation or public
score is claimed.
