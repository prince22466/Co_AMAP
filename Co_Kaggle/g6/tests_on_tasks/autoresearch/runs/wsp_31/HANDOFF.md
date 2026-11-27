# WSP research handoff · 25 November 2026

**Keep `iter_04` as champion. Its recorded best is 5/15 on QUICK_15.** The same adapter scored 4/30 in an earlier QUICK_30 run. This Colab experiment produced four candidate adapters and zero new benchmark scores. The private L4 session `wsp7dev` was released after its research outputs were saved.

## What the runs established

| Evidence | Result | Implication |
|---|---|---|
| Frozen agent, fresh run on five published Rich development snapshots outside QUICK_30 | 1/5 solved (`rich_3067`), 16 file-edit errors, two premature submissions, one syntax leak | The 1/5 is a separate development result. |
| `rich_3506`, champion’s returned patch | The agent submitted without testing. Native replay reproduced that decision on the exact 6,437-token input. The grader reported 14 failed and 38 passed. A short assertion on the two widths returned by `Segment.split_cells(3)` failed on the actual submitted patch and passed on the published fix. | A generated, executable behavior check could have exposed this defect before submission. A subsequent agent repair remains unobserved. |
| Two other returned patches | A padding behavior check on `rich_3296` caught a syntax error at import. A URL-span check passed on the champion’s successful `rich_3067` patch. | These cases distinguish an edit that needs repair from a valid result worth preserving. |
| Three positive calibration histories | Each public-API check failed on the original snapshot and passed after applying the published development fix. The corrected chat renderer also reproduced native token IDs across all 71 saved requests. | Use these constructed histories to test whether the adapter preserves valid submissions. The agent must still demonstrate that it submits on the verified state. |
| Four sparse down-projection edits at gain 0.25 | Layers 0 and 5, one or 16 columns each. The intended activation was positive on 0/3 discovery branches for three edits and 1/3 for layer 5 with 16 columns. | None met the reference’s greater-than-80% sign target on these branches. They are weak candidates for a full run. |

The ranking used three snapshots with reproduced actions, including two where the observed action was a test command. The contrast used teacher-forced alternative continuations. The three signs above therefore describe discovery branches. They do not constitute three independently observed premature submissions or a held-out selectivity measurement.

## What is saved

- [Discovery run](discovery02/latest.zip) contains metrics, returned patches and grader output inside the nested diagnostics archive, HTTP inputs, and exact native token IDs.
- [Final Colab checkpoint](wsp31_handoff.zip) contains the native champion replays, per-snapshot activation summaries, rankings, four factor files, compiled candidate/zero/random adapters, three exactly formatted verified histories, and the failed screen log. The 44-member ZIP passed integrity checks. Its SHA-256 is `bd2159d3a09d3581c7a3a11bc23cdbb101262d1ba5f5b8da3200e88a7f162bd9`. The most useful members are `harvest03/native04/columns/candidates/ranking.json`, `harvest03/native04/columns/summaries.pt`, `harvest03/native04/original/champion.json`, and `preservation/exact/verified_checks.json`.
- [Champion archive](../../../../submission_nb/artifacts/submission_40.zip) contains the frozen agent configuration and prompt. The champion **adapter** SHA-256 is `8a018d2df7d8bc367925533ce0488ee1bbadeecd7ac813fdd3a97d5b9d3c9815`. The [score ledger](../../results.tsv) records the historical benchmark runs.

## Tool guide

| Tool | Purpose |
|---|---|
| [Public data staging](stage_public_data.py), [runtime recovery](recover_colab.py), [evaluator environment](prepare_evaluator.py) | Stage selected published Kaggle objects, restore the authorized files to a private Colab runtime, and install the pinned competition wheels. A fresh VM also needs working Python 3.12 venv support for task sandboxes. |
| [Frozen runner](run_evaluator.py) and [canonical notebook](../../kaggle_eval_kernel/gemma-4-developer-agent-testing_v12_wsp.ipynb) | Run the agent and grade patches. Development runs capture the local HTTP inputs and serving tokenizer output. The runner checks the staged adapter digest before grading. |
| [Decision extraction](decision_cases.py), [native replay](native_replays.py), [verified states](verified_states.py) | Recover actual post-edit decisions, reproduce them in vLLM on the original full input, and construct positive calibration states by executing checks on published snapshots. |
| [Activation harvest](harvest_decisions.py) and [column ranking](rank_columns.py) | Collect input/output means across 42 FFNs and rank sparse edits by signed contribution with one snapshot omitted at a time. Full inputs drive action replay. Left-truncated windows of up to 4,096 tokens drive activation harvesting. |
| [Adapter compiler](../../recipe.py) and [compiler tests](test_explicit.py) | Preserve the champion in slot zero and write explicit BF16 factors into slot one. Produce zero-padded and norm-matched random controls, including for layers outside the champion’s original five. |
| [Candidate screen](run_screen_stage.py) and [serving-logit check](serving_logits.py) | Compare generated actions under candidate and control adapters, and separately compare raw next-token logits at the actual serving rank settings. The first screen exhausted GPU memory while warming 256 dummy requests. The revised replay limits concurrency to eight. No completed candidate comparison was recovered. |
| [Runtime guard](guard_runtime.py) | Checkpoint results, verify the download, and stop a backed-up idle runtime. Configure the session and instance marker when reusing it. |

## Resume in this order

1. **Start with the saved evidence.** Inspect the four signed activation vectors and the champion replay before renting another GPU. The strongest behavioral example is `rich_3506`. Gather more reproduced unchecked submissions from distinct public development snapshots. Use real successful checks and edit-argument tokens to measure what a new edit must preserve.
2. **Validate the serving control.** The local compiler preserved the champion factors. A Transformers BF16 check still showed a maximum raw-logit difference of 0.1875 between rank one and zero-padded rank two. The competition vLLM maps these to different maximum ranks (1 and 8). On the same saved cases, run `serving_logits.py champion` followed by `serving_logits.py paired`, both with `--cases /content/wsp/harvest03/native04/columns/cases.json` under `eval_env/bin/python`. If the serving logits differ, evaluate a mathematically equivalent rank-one export for an edit on a previously unused layer and recheck the zero control.
3. **Test a more selective edit before QUICK_15.** Measure its sign on genuine unchecked decisions and its magnitude on verified submissions and edit arguments. Compare the frozen candidate with the champion, zero, and norm-matched random controls on identical native inputs. Execute the emitted command on the actual returned patch. The relevant outcome is a command that catches the defect, followed by a repair that passes grading. The three reserved fresh episode IDs are `rich_3043`, `rich_3064`, and `rich_2725`.
4. **Promote only on a measured gain.** Use the frozen runner on QUICK_15 and list newly solved and lost tasks. At 7/15 or higher with all five champion solves retained, archive the clean submission and confirm that exact adapter SHA on QUICK_30. Report the original and additional 15 separately. Follow the [continuation reference](../../WSP_CONTINUATION.md) and [archive procedure](../../program.md).

For a new VM, restore the final checkpoint under `/content/wsp/` preserving its directory structure. Restore the contents of `discovery02/latest.zip` under `/content/wsp/experiments/discovery02/` and the champion adapter under that run’s `input/adapters/wsp_r1/`. Stage the base model, published snapshots, wheels, canonical notebook, compiler and current experiment scripts separately. The checkpoint stores research outputs. The large base model and wheel environment must be provisioned again.

All task-specific checks and published reference patches used here stayed in development tooling outside the submission. This run used published development data and the private Colab account. No leaderboard submission was made.
