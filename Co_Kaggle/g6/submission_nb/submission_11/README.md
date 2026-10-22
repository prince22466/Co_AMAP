# Submission 11: Submission 10 with Gemma 4 E2B

This candidate preserves Submission 10's prompts, repair gate, skills, tools,
sampling settings and evaluation budgets. The only active-file changes are the
model declarations in `agent.yaml` and both helper YAML files, from
`gemma-4-31b-it-qat-w4a16-ct` to `gemma-4-e2b-it-qat-w4a16-ct`.
E2B is the smallest supported variant listed in
`competition_material/Update_new_models.md`; all three agents use it.

From the `g6` directory, with Python 3.11+ and PyYAML installed:

```powershell
python -X utf8 tests_on_tasks/tools/build_submission11.py --validate-only
python -X utf8 tests_on_tasks/tools/build_submission11.py
```

The builder validates the inherited Submission 10 architecture and enforces
byte-for-byte preservation except for the three model substitutions. It keeps
the same ADK-compatible skill directory, LF archive line endings and generated
reference aliases as Submission 10. Upload `submission_nb/artifacts/submission_11.zip`; do not ZIP
the raw source directory. The adjacent manifest records hashes and checks.

The optional compiler/executor smoke test is available with the exact official
dependencies documented in `tests_on_tasks/tools/verify_submission11_runtime.py`:

```powershell
python -X utf8 tests_on_tasks/tools/verify_submission11_runtime.py
```

Static structure, baseline preservation and ZIP/source parity were checked.
The official runtime smoke was not run because its ADK dependencies are not
installed locally. No Gemma inference, Kaggle evaluation or public score is claimed.
