# Submission 5: longer grading timeout and thinking disabled

Prepared on 2026-10-13 from Submission 4, after its reported public score of 0.0. The score's cause remains unknown. This candidate applies the two configuration considerations in [the zero-score investigation](../zero_score_investigation.md).

## Changes from Submission 4

| Setting | Submission 4 | Submission 5 | Intended effect |
| --- | --- | --- | --- |
| `evaluation.timeout_seconds` | 60 | 300 | Allow longer final grading checks in the inspected harness. |
| Coder thinking configuration | `thinking_budget: 4096`, `include_thoughts: true` | Only `include_thoughts: false` | Request thinking disabled rather than relying on a numeric budget that may not be enforced. |
| Both helpers' thinking configuration | `thinking_budget: 1024`, `include_thoughts: true` | Only `include_thoughts: false` | Reduce helper reasoning overhead within the shared task allowance. |

Only four packaged files change: `eval_config.yaml` and the three sampling files. The root/agent declarations, all three instruction files, tools, helper routing and prompt thresholds are byte-identical to Submission 4. No adapters or custom callbacks are introduced.

The inspected compiler bridge uses `include_thoughts: false` to set the model's thinking toggle off. Numeric thinking budgets are omitted, avoiding a misleading positive budget alongside the off request and avoiding `thinking_budget: 0`, which some compiler schemas reject. Confirm the effective model request and observed behavior in the actual Kaggle runtime; local configuration validation does not establish that thinking is disabled during inference.

## Retained architecture and limits

- Coder: six direct harness tools plus both optional helpers; owns source edits, repairs and submission.
- Repository locator: shell searches and source reads to resolve one location uncertainty.
- Test verifier: one focused behavioral check or failure diagnosis using shell/read tools.
- All agents use `gemma-4-31b-it-qat-w4a16-ct`, with no adapters.
- Task allowance remains four minutes, 40 counted calls and 80 turns.
- Temperature remains 0.2 and top-p 0.95 for every agent.
- Output ceilings remain 16,384 for the coder and 2,048 for each helper.
- Helpers retain no full history, no editing/submission tools, and the same prompt restrictions on shell use.

The command timeout and agent session allowance are separate settings. The inspected harness applies the command timeout to the final grading pytest command too. Raising it to 300 does not raise the four-minute agent allowance. Agent commands are still subject to remaining session time; the unchanged prompts request quick checks using `timeout 20s` when available. Final verification can take longer under the new setting and its time is excluded from the competition's generation allowance. Confirm the scoring runtime's actual behavior and overall run time.

## Build and evaluate

From the project root:

```bat
python -B submission_nb\tools\submission_tools.py validate --submission-dir submission_nb\submission_5
python -B submission_nb\tools\submission_tools.py build --submission-dir submission_nb\submission_5 --output submission_nb\artifacts\submission_5.zip
python -B submission_nb\prepare_evaluation_notebook.py --submission-dir submission_nb\submission_5 --runtime-baseline-dir submission_nb\submission_4 --output submission_nb\evaluate_submission_5.ipynb
```

The default notebook comparison retains all three agents for every variant and embeds the actual Submission 4 and Submission 5 files:

| Variant | Command/grading timeout | Thinking requested | Purpose |
| --- | ---: | --- | --- |
| `baseline` | 60 seconds | On | Original Submission 4. |
| `timeout_only` | 300 seconds | On | Isolate the timeout change. |
| `thinking_off_only` | 60 seconds | Off | Isolate the thinking change. |
| `candidate` | 300 seconds | Off | Full Submission 5; default release. |

All variants receive the same selected tasks, prompts, agents, direct tools, generation ceilings and agent allowance. Preflight rejects other configuration changes, incomplete thinking changes or mismatched active package paths. A mounted candidate must satisfy the same restrictions. The intermediate controls are generated in fresh run directories; earlier submission archives are preserved.

Start with the default two-task smoke check in Kaggle. Preserve raw task errors, traces, patches and verification output, then pin a larger development set and separate validation tasks. Rotate/reverse variant order for timing comparisons and repeat promising results. The two isolated variants can show which setting matters; the combined candidate's score alone cannot establish that.

After evaluation, `RELEASE_VARIANT` selects the package written to `/kaggle/working/submission.zip`. The local release is `submission_nb/artifacts/submission_5.zip`; only ten active agent files are packaged. Documentation, manifests, reports and notebooks stay outside the ZIP. Existing diagnostic reports retain task outcomes, available tokens/timings and locator/verifier calls. This candidate does not add a mechanism for forcing private scoring logs into Kaggle's visible log.

For helper-availability comparisons using Submission 5's settings, an additional notebook can be generated with `--verification-control` instead of `--runtime-baseline-dir`. Keep that architecture experiment separate from the default runtime comparison.

## Evidence status

Submission 3 and Submission 4 have reported public scores of 0.0; Submission 2 previously scored 0.10. No task-level scoring artifacts explain the zeros. The source findings behind the timeout/thinking changes come from a published copy of organizer harness code, not a verified inspection of the user's installed Kaggle wheels.

Submission 5 is unmeasured. Local checks validate dependency closure, three-agent declarations, exact scope of configuration changes, runtime-control isolation and archive/notebook consistency. Real-model inference, official runtime compilation, grading and full-run timing still require the competition environment.
