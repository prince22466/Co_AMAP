# Submission 27 — Evidence & Progress Manager (S26-controlled experiment)

**Unscored candidate**. This version is a single E4B LlmAgent with precisely the
same six native tools, generation settings and global limits as S26. No LoRA,
custom tool schema, mandatory skill wrapper, additional model agent, or stage
handoff was introduced. The ONLY runtime change from S26 is `instruction.md`.

## Why this experiment

S25: 0/15 solved and 14/15 empty, dominated by invalid custom-tool calls.
S26: 2/15 solved (`rich_3905`, `rich_3454`), restoring native tool calls, but
repeated failed `edit_file` anchors, unfocused `read_file` calls and 40-call
exhaustion remain visible. The S26 printed trace does not carry complete tool
arguments and should not be used to claim exact duplicate argument identities.

The runtime design is a **cooperative, bounded in-context decision protocol**:
- **Evidence Ledger**: four small slots (`R`, `L`, `H`, `F`) with verified
  requirements/locations, hypothesis and most recent failed approaches.
- **Repetition Detector**: model checks same read range, failed anchor, and
  unproductive queries; after two failed edit anchors it must switch strategy.
- **Progress Controller**: decision order prioritizes verification/submission
  after a patch, bounded recovery after edit failure, and early implementation.

**Enforcement boundary:** The competition YAML compiler supplies neither a
submission-defined host tool interceptor nor a callback registry in the pinned
harness. Therefore S27 cannot truthfully claim to prevent repeated tool calls
mechanically. The ledger is not durable state or automatic context compaction.
It depends on E4B following these concise instructions. Unlike S25, no wrapper
blocks direct reads/edits. A separate deterministic diagnostic analyzer outside
the ZIP measures what the model actually does; it cannot change runtime behavior.
The single changed axis makes a matched S26/S27 comparison interpretable.

## Build and tests

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission27.py
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission27.py -v
python Co_Kaggle/g6/tests_on_tasks/tools/analyze_submission27_traces.py \
  Co_Kaggle/g6/tests_on_tasks/diagnostics/submission-26-zip_s1TasksPrintouts.txt.txt
```

Upload `submission_nb/artifacts/submission_27.zip` as `submission.zip`.
The analyzer labels same-path repeated reads and same-path failed edits as
*proxies*, not confirmed identical call arguments; it parses text printouts,
which can omit tasks/details. Use complete structured tool-call traces for
precise duplicate-argument and progress findings.

## Official evaluator pilot command

Use the existing locally served E4B QAT model. The wrapper runs the official
baseline preflight before evaluating any candidate; it writes raw outcomes to
a new output directory and does not overwrite prior traces.

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/run_submission27_eval.py \
  --tasks /kaggle/input/competitions/gemma-4-developer-agent/tasks.jsonl \
  --snapshots /kaggle/input/competitions/gemma-4-developer-agent/snapshots \
  --wheels /kaggle/input/competitions/gemma-4-developer-agent/wheels \
  --served-model gemma-4-e4b-it-qat-w4a16-ct \
  --stage pilot --repeats 2 --time-limit 4 \
  --output /kaggle/working/s27-pilot
```

## Real-model gate (not replaced by CPU unit tests)

Run the same five S25 regression IDs (`rich_4077`, `rich_3905`, `rich_3518`,
`rich_3454`, `rich_3882`) with S26 and S27 on the *same* E4B server, model,
budget, task snapshots, baseline preflight and verification harness. Require
>=3/5 solved, retention of `rich_4077`, <=1 empty patch, zero agent errors
in each repeat. Inspect every model-emitted tool call including failed edits,
patch relevance, semantic assertions, and submit_patch. Stop if the gate fails.
Then run matched full S1, at least two repeats, and report solve rate, empties,
invalid edits, repeated calls, tool/time costs and regression IDs. No Kaggle
score or task-resolution improvement is claimed by packaging/unit checks.

**Model rule:** E4B testing must match the actual permitted model in the
competition environment; verify the current authoritative whitelist before
uploading. This configuration intentionally matches S26 to isolate the prompt.
