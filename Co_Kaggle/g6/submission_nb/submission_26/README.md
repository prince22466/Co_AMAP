# Submission 26 — native repair, bounded evidence, lower context pressure

**Unscored E4B candidate.** It is NOT an observed performance improvement until
real E4B testing passes the release gate below. The ZIP is independently usable
and contains only the official declarative configuration and instructions.

## Architectural decision

S25 regressed to 0/15 S1 tasks (14 empty patches) after removing native file and
command tools in favor of `run_skill_script`; this is a confirmed model-tool
interface regression. S21 solved 2/15 with native tools; S22 solved a different
2/15 but did not call optional skills; S24's multi-stage review loop solved 1/15,
with turn-limit and repeated-work failures. See the two diagnostics reviews.

S26 deliberately makes **one main change in architecture relative to S25**:
restore the working native tools, remove the mandatory workbench and all skills,
and return to a single agent. Unlike S24 there is no context-handoff ceremony,
second reviewer cycle or submit-only finalizer that can starve the actual edit.
It is intentionally closer to S21 so a matched ablation can isolate the effect.

Differences vs S21 are smaller and explicitly hypothetical: a shorter prompt,
a hard preference for early edit over exhaustive discovery, output-limited narrow
searches to reduce attention diffusion, native submission in the SAME context,
and 512 thinking tokens with deterministic sampling. `include_contents: default`
retains observations needed for reliable exact-string edits. The prompt encourages
small tool responses rather than pretending to provide an unsupported compaction
hook or an enforced per-stage budget. Only the global official limits are hard.

No custom script path, hidden edit restrictions, model weights, LoRA, reference
patches or evaluation data are present. Native tools can modify legitimate
metadata/config/docs/new files as well as implementation. Do not attribute any
future improvement to one factor without a matched prompt/config ablation.

## Build and local checks

From the repository root:

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/build_submission26.py
python -m unittest discover -s Co_Kaggle/g6/tests_on_tasks/tools -p test_submission26.py -v
```

Upload `Co_Kaggle/g6/submission_nb/artifacts/submission_26.zip` as `submission.zip`.
The pinned official compiler and native read/edit/submit scripted smoke should
also be run in Kaggle's existing testing environment. CPU checks establish
packaging and tool availability, **not** E4B tool-use reliability or task score.

## Mandatory real-E4B promotion gate

Reuse the existing S25/S24 official evaluator and its fixed pilot IDs:
`rich_4077`, `rich_3905`, `rich_3518`, `rich_3454`, `rich_3882`.

```bash
python Co_Kaggle/g6/tests_on_tasks/tools/run_submission26_eval.py \
  --tasks /kaggle/input/competitions/gemma-4-developer-agent/tasks.jsonl \
  --snapshots /kaggle/input/competitions/gemma-4-developer-agent/snapshots \
  --wheels /kaggle/input/competitions/gemma-4-developer-agent/wheels \
  --served-model gemma-4-e4b-it-qat-w4a16-ct \
  --stage pilot --repeats 2 --time-limit 4 \
  --output /kaggle/working/s26-pilot
```

If the server uses a model path as ID, supply the exact `/v1/models` ID. Check
baseline environment readiness first. Inspect all *model-emitted* calls and
results, not only whether tests eventually passed. Gate EACH repeat on >=3/5
resolved, retention of `rich_4077`, <=1 empty patch, zero agent errors; additionally
require no sustained malformed calls, real reads + native edits + nonempty
explicit submit_patch, and relevant tests. The runner only computes the objective
counts, NOT the qualitative inspection. A green summary is **not automatic
permission** for full S1 until every trace is reviewed.

Only after a passing pilot, run matched S1 on the SAME 15 tasks and the SAME model,
server, settings and budget as S21/S22/S24 controls, with repeats. Record all
resolved IDs, empty patches, dispatch errors, repeated calls, relevance of diffs,
measured time and aggregate tokens. Target >=6/15, >=3 retained prior wins,
<=2 empties, no errors in each repeat. Keep unseen public tasks for a holdout.
The `rich_4075` task-title/trace discrepancy must be reconciled with original
problem statements before treating its output as correctness evidence.

**No successful E4B pilot, official-runtime smoke, or S1 uplift is claimed here.**
