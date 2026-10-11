# Submission 3: optional patch reviewer

Submission 3 extends Submission 2 with a third configured role: an optional read-only patch reviewer. The coder owns implementation, tests, repairs and submission. The analyzer answers a focused localization or root-cause question; the reviewer checks an existing patch against the issue and observed verification results. Both helpers are `agent_tool` connections with `skip_summarization: true`, `include_contents: none`, and separate output keys. They receive self-contained requests, not the full coder history.

| Role | Tools | Use |
| --- | --- | --- |
| Coder | Nine harness built-ins plus both helpers | Own the entire task and capture the patch. |
| Analyzer | `read_file` and three graph/retrieval tools | Optional focused uncertainty before or during implementation. |
| Reviewer | Only `read_file` | Optional correctness check after a patch and targeted verification exist. |

## Review policy

The coder checks `get_status` before review. Its prompt skips review when less than 90 seconds or fewer than six counted calls remain, when the patch is empty or clearly incomplete, or when a known fix has priority. At most one review is requested. The reviewer receives the issue, acceptance conditions, current diff, changed/new paths, and verification commands with actual outcomes or explicit unrun status. Omitted diff portions must be identified.

The reviewer normally uses zero to two bounded reads and returns at most 180 words with supported correctness defects, minimal corrections and verification gaps. It cannot edit, run commands, test, delegate or submit. The coder checks remaining resources, repairs evidence-backed defects and performs focused verification when feasible. Reviewer findings do not prove test success. Failed or unhelpful delegation falls back to the coder's own final inspection.

Time thresholds, call/read targets and word limits are prompt instructions, not enforced helper quotas. Helper inference and reads share the task allowance; the harness's total task budgets remain the hard limits. The final 30 seconds are reserved by prompt convention for repair, review and submission.

## Fixed configuration

- All three agents use `gemma-4-31b-it-qat-w4a16-ct`; no adapters.
- Task allowance: four minutes, 40 counted tool calls, 80 turns and a 60-second command cap.
- Coder sampling: temperature 0.2, top-p 0.95, 16,384 output ceiling and 4,096 thinking budget.
- Both helpers: temperature 0.2, top-p 0.95, 2,048 output ceiling and 1,024 thinking budget.
- Submission 2's evaluation config, coder sampling, analyzer config and analyzer resources are preserved byte-for-byte.
- Packaged changes: coder instructions, the added reviewer connection in `agent.yaml`, and three reviewer configuration/resource files.

## Build and compare

From the project root:

```bat
python submission_nb\tools\submission_tools.py validate --submission-dir submission_nb\submission_3
python submission_nb\tools\submission_tools.py build --submission-dir submission_nb\submission_3 --output submission_nb\artifacts\submission_3.zip
python submission_nb\prepare_evaluation_notebook.py --submission-dir submission_nb\submission_3 --reviewer-control --output submission_nb\evaluate_submission_3.ipynb
```

The companion notebook embeds the candidate and current diagnostic tools. Its default variants are:

| Variant | Configured agents | Change from the candidate |
| --- | ---: | --- |
| `coder_only` | 1 | Remove both helper connections and unreachable resources. |
| `optional_analyzer` | 2 | Remove only the reviewer connection and its resources. |
| `optional_reviewer` | 3 | Full Submission 3; default release. |

All variants use the same coder prompt, shared sampling and total budget. Helper-specific instructions are conditional on tool availability. This comparison isolates helper availability under Submission 3's instructions; its two-agent control is not the original Submission 2 prompt. A count of three configured agents does not mean all three are invoked on every task.

Run the default two tasks as a smoke check, then pin a development set spanning repositories (approximately 24 tasks is a starting proposal). Repeat/reverse variant order when assessing generation variation or timing, and retain a separate validation set. Use the full matching competition data and model in Kaggle. Review task outcomes, patches, failure categories, analyzer/reviewer calls and runtime before promotion. Call counts alone do not establish reviewer benefit; inspect the review findings and subsequent repairs in available traces.

The notebook writes per-variant JSON, CSV and Markdown diagnostics and packages the selected `RELEASE_VARIANT` to `/kaggle/working/submission.zip`. Local notebooks, tests and documentation stay outside the submission ZIP. Scoring Docker parity and total inference-time checks remain necessary.

## Evidence status

The user's [submission records](../../submission_records.md) report starter 0.03, Submission 1 0.06 and 0.10, and Submission 2 0.10. These aggregate scores do not establish that an analyzer helps, that prompt revisions improved performance, or that review will help.

Submission 3 has no reported public score or GPU benchmark. Local validation, packaging and offline integration tests check configuration and workflow behavior; the competition compiler and runtime remain authoritative. The inspected local Python environment lacks `swegemma` and `adk_submission`, and the downloaded task resources currently do not form a complete matching task. Runtime compilation, inference and verification must be performed in the competition environment.
