# Submission 37 — deterministic repetition-guard research experiment

**Status: research-only S37 A/B; NOT YET a new Kaggle submission package.**
This experiment starts with the S36 *primary* config (public score 0.06)
and asks whether blocking identical previously failed native edits helps E4B.

**Do not upload a purported S37 ZIP yet.** The competition uses a
declarative YAML/approved-tool/skill/weight compiler. Arbitrary Python ADK
callbacks in the research harness cannot be added to a competitor ZIP.
A ZIP built from S36 remains S36 behavior; it does not inherit this guard.
We must validate the headroom and an approved deployment mechanism before
creating a new competitive submission.

## Experiment A/B

| Dimension | Control | Guard |
| --- | --- | --- |
| E4B model, prompt | S36 primary | Same |
| Native tools | S36 original six | Same |
| Temperature/top-p | 0 / 0.95 | Same |
| Max output / thinking tokens | 4096 / 512 | Same |
| Packaged per-task limits | 4 min / 40 calls / 48 turns | Same |
| ZIP bytes | Exact S36 primary | **Exact same ZIP** |
| Research harness | Unmodified official evaluator | Official evaluator with temporary ADK hook |
| Kaggle submit-ready | Existing S36 | **No** |

The controller is not an agent, wrapper tool, LoRA, prompt change, or skill.
Only the research runner installs it. Skipping actual native tool invocations
can change the *effective* charged-call budget, although configured limits
are identical; always disclose this confound.

### Exactly what is prevented

1. After a native edit_file returns a **non-mutating source-matching**
   FileEditError (missing or ambiguous old_string, empty source/anchor),
   store a SHA-256 fingerprint of its exact tool arguments, session and
   workspace-change epoch. Other FileEditError results (e.g. copy/transfer
   failures) may occur after mutation and are NOT cached. Raw source
   arguments are never logged.
2. If E4B retries the exact same edit in the same epoch, the before-tool
   hook returns a structured RepeatedFailedEdit message instead of
   dispatching another native call.
3. read_file/get_status do not invalidate the failed-anchor memory.
   Every run_command/write_file invalidates it **before dispatch**, even
   when the tool later raises an exception; any different edit_file
   arguments invalidate it for the same reason. Successful or uncertain
   editing outcomes also invalidate memory. This fail-open design
   prevents stale retries from being incorrectly blocked.
   A maximum of TWO synthetic denials per exact failed call protects
   against an infinite loop of zero-charge rejections; subsequent repeats
   execute natively and consume the ordinary tool-call budget.
4. Different arguments, a different session, or a new workspace epoch are
   never blocked. The guard does not prevent malformed tool calls rejected
   before dispatch, incomplete generations, or repeated internal reasoning.

This is a per-task deterministic state machine, not a model-held mental
ledger like the unsuccessful S27 prompt-only approach.

The runner temporarily patches the google.adk.apps.App initializer to
register a standard BasePlugin, only while executing the guarded arm. The
evaluation tools and grader are not modified. It checks ADK hook signatures
and requires an App to be intercepted. **Full compatibility with the actual
installed Kaggle wheelhouse is not proven by CPU tests**.

## Run with the existing vLLM E4B server

From Co_Kaggle/g6/tests_on_tasks/tools; use identical competition tasks,
snapshots, wheelhouse, model server, and sandbox settings in both arms.

~~~bash
python build_submission36.py

python run_submission37_eval.py --variant control --stage pilot \
    --tasks /data/tasks.jsonl --snapshots /data/snapshots \
    --wheels /data/wheels --repeats 2 --time-limit 4 \
    --output /work/s37_control

python run_submission37_eval.py --variant guard --stage pilot \
    --tasks /data/tasks.jsonl --snapshots /data/snapshots \
    --wheels /data/wheels --repeats 2 --time-limit 4 \
    --output /work/s37_guard

python compare_submission37.py \
    --control /work/s37_control --guard /work/s37_guard \
    --stage pilot --output /work/s37_pilot_comparison.json
~~~

For full historic S1 tasks use --stage s1 in all commands. For a
historical unlimited-time comparison, use --time-limit 0 in *both* arms,
but never compare that result with four-minute Kaggle-style runs.

Each arm builds/verifies the same S36 primary archive and runs the
original baseline preflight and phase-2 verifier. Evidence includes
submission37_research_telemetry.json (actual guard intervention),
submission37_research_summary.json, the official run_manifest.json,
and candidate_N/notebook_task_results.jsonl with full traces.

The comparator requires identical task IDs, package bytes, task data,
models, wheel versions, sampling/compaction configuration, budgets,
and repeats. It reports resolution gains, losses, retained successes,
empty patches, errors and runtime. Fewer rejected calls alone is NOT
an improvement in task-solving performance.

## Stop/go criteria

- Begin with a live one-task smoke confirming that the callback actually
  sees and (on a repeated deterministic failure) blocks native edits.
  CPU unit tests do not prove a live hook.
- Then run two matched pilot and S1 repeats, inspecting raw ATIF traces,
  final patches and real task resolution.
- Evaluate a preselected cross-repository holdout including Requests
  and FastAPI, rather than repeatedly tuning on Rich.
- If a strong benefit appears, identify a way to implement it within the
  competition's *allowed closed registries* before making a new
  Kaggle-submit-ready S37 archive. Do NOT smuggle Python host callbacks
  into the ZIP or alter the official scorer.

## CPU tests

~~~bash
python -m pip install 'pyyaml>=6,<7'
python -m unittest -v test_submission36.py \
    test_submission37_guard.py test_submission37_runner.py
python -m py_compile submission37_repetition_guard.py \
    run_submission37_eval.py compare_submission37.py
~~~

All S36 prompt, model, native tools and budgets remain unchanged.
