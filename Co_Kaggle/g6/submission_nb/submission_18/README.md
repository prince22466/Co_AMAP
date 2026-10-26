# Submission 18 — native tools, E4B

Addresses the execution failures observed in Submission 17 (0/30, zero successful workbench
edits, 393 missing-wrapper argument errors, 133 missing-script errors, 25 turn-budget stops).
Uses **gemma-4-e4b-it-qat-w4a16-ct**, the same model variant as Submission 17.

The submitted ZIP contains only agent.yaml, instruction.md and eval_config.yaml.
There is one agent with the official read_file, edit_file, write_file, run_command,
get_status and submit_patch tools. There is no skill dispatcher, nested action schema,
mandatory assertion attached to an edit, separate finalizer, or multi-agent LoopAgent.
The agent owns the inspect/edit/test/repair/submit cycle and retains conversation history.
Thinking remains disabled. The declared budget remains 4 minutes / 40 calls / 48 turns.

This deliberately trades the earlier workbench's guarded edits for a simpler native interface.
Tests and source changes are now separate calls. Protected-file and deadline behavior comes
from the official harness; instructions additionally prohibit modifying evaluation files.
Failed native reads count against the harness budget. Malformed calls rejected before tool
execution can still consume turns without counting as tool calls. The hard 48-turn cap remains.
Early submission, useful assertions and stopping after repeated failures remain model behaviors,
not guaranteed host controls. A single agent can still waste its budget or submit empty work.
There is no claim of automatic independent verification or enforced per-stage handoff.

Graph and embedding tools are intentionally not added in this experiment: the measured failure
was inability to edit and submit, and this candidate isolates the direct-tool change.
No improvement in E4B task-solving has yet been measured. CPU tests are not model inference.

## Build and validation

From repository root:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission18.py
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission18.py -v
# With the pinned competition runtime wheels installed:
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission18_runtime.py -v
python Co_Kaggle/g6/tests_on_tasks/tools/smoke_submission18.py --output-dir /tmp/submission18-smoke
```

The runtime tests use adk-submission 0.2.13, adk-eval-core 0.1.0, google-adk 1.36.1,
google-genai 2.11.0 and swegemma 0.2.11. They exercise the actual native tools, compiler,
Runner and subprocess sandbox. The CLI smoke uses a scripted model and labels its output.
Archived Submission 17 failure excerpts live only in tests_on_tasks/reviews/submission_18;
they and the synthetic oracle never enter the submission ZIP.

Download the `submission-18-e4b-unscored` Actions artifact, extract it, and upload the inner
`submission_18.zip`. The committed manifest records its hash. A smaller model does not by
itself fix a Kaggle upload-format error; no upload diagnosis is claimed here.

## Real-model gate before task evaluation

On Kaggle, after starting the E4B server and constructing the standard ModelRegistry called
`models`, run the helper against the EXTRACTED submission directory. Copy build_submission18.py
and smoke_submission18.py from tools into an importable directory; importing the builder does
not build or overwrite anything. No existing notebook needs to be edited by this PR.

```python
from smoke_submission18 import run_smoke
report = await run_smoke('/kaggle/working/submission18_smoke',
                         models=models, submission_dir=AGENT_DIR)
assert report['passed'], report
```

The gate requires a baseline behavioral failure, an intentionally wrong first edit, a failed
candidate check, repair, the same passing output assertion, a fresh patch and one nonempty
submission. It independently checks final behavior, so a model replacing assertions with
`assert True` cannot pass. This is a development-only synthetic fault-injection exercise.
It does not change the official evaluator or enforce arbitrary benchmark behavior.

If the gate fails, inspect events.json and stop the GPU session rather than running 30 tasks.
If it passes, start with a small fixed task set. Record the effective four-minute deadline,
40 calls / 48 turns and disable compaction for a controlled comparison. Retain an unmodified
baseline run to distinguish test-environment failures. This PR does not change
submission_records.md or gemma-4-developer-agent-testing_v9.ipynb.
