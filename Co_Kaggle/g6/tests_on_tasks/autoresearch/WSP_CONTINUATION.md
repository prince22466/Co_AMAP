# Adapter weight-space programming toward 7/15

Start from `iter_04` and spend the next GPU session finding a small FFN edit that makes Gemma verify the behavior it just changed. The first end-to-end milestone is a previously unseen development issue where the serialized adapter makes the agent run a relevant check, use the failure to repair the patch, and pass grading. Promote the candidate to QUICK_15 after that behavior transfers across issues and preserves good submissions.

For this campaign, use the competition's quantized Gemma E4B checkpoint and the frozen `submission_36` agent, prompts, tools, sampling and evaluator. The submission change is a single PEFT LoRA adapter programmed from forward-pass measurements and closed-form calculations. Use the existing four-minute, 40-tool-call and 48-turn budgets. Start with a uniform rank of 2 and `lora_alpha=2`. Rank 2 has scored in the frozen evaluation, and the organizer documentation gives a server limit of 128 and an unpacked submission limit under 3 GiB. [Original program](program.md) · [Rank-2 result](runs/iter_08/neon_metrics.json) · [Organizer harness](../../competition_material/competition_data/HARNESS_README.md#3-model-registry-single-model-rule--lora-adapters-on-4x-l4-gpus) · [Organizer model announcement](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287)

## What to recover

`iter_04` reached 5/15 once. All 30 QUICK_15 runs together solved just five distinct tasks. On QUICK_30, the same adapter solved three of the original 15 and one of the 15 added tasks. Track those original five individually when comparing candidates. `iter_26` reduced edit errors from 68 to 26 while scoring 4/15. Error counts help explain trajectories. Pass/fail and per-task wins and losses decide promotion. [Run ledger](results.tsv) · [Best QUICK_15](runs/iter_04/neon_metrics.json) · [Same-adapter QUICK_30](runs/iter_04_m30/neon_metrics.json)

| Observed miss | What an adapter would need to change |
| --- | --- |
| `rich_3052` changed the constructor and choice checker, then submitted. The public `Prompt.ask(...)` call still rejected the new keyword. In 27/30 runs, only the case-insensitive test failed. | Exercise the public API named in the issue after a successful edit. Use the resulting exception to trace the entry point and repair the missing change. |
| `rich_3718` changed the title style and submitted without rendering a panel. Six runs had only the panel-background test failing. | Reproduce the requested rendered behavior and check the default case before submitting. |
| `rich_4075` is a backup target. In 24/30 runs, only the custom `end` case for an empty `Console.print(...)` failed. | Turn a short issue into a direct behavior check and inspect the actual output. |

[Champion traces and grading output](runs/iter_04/diagnostic_runs/g6-autoresearch-subx2a_aggu/) · [Public task catalog](../task_inventory/task_catalog.md)

Use these examples to define a behavior, then derive and select weights on other tasks. On `requests_7315`, 12 runs removed the same two lines as the published reference and still failed grading. An older trace showed the agent sandbox importing installed Requests. Verification-sandbox import origin remains open. Record this as a grader investigation and include the official result in every score. A reference-patch replay in a clean verification environment can test the import hypothesis independently of WSP. [Matching source change and failing test](runs/iter_02/diagnostic_runs/g6-autoresearch-sub1ij_r4o3/raw_results/requests_7315.json) · [Earlier import diagnosis](../reviews/submission_9/review.md)

## Environment and replay

| Environment | Job and checks |
| --- | --- |
| Local CPU | Build labels and task splits from archives. Inspect Safetensors and fit or factorize cached activation summaries in FP32. Check shapes, hashes, package entries and low-rank arithmetic. |
| Single L4 or another validated GPU | Run the activation lab in one persistent session. Use the exact QAT model revision, tokenizer and champion adapter. Measure model-load and first-forward memory, then verify a representative long replay. The existing Colab workflow reports first-forward expansion and single-T4 OOM with its Transformers path. Reuse its setup after checking versions and mounted paths. |
| Frozen Kaggle L4×4 evaluator | Run official-format comparisons with the same model, parser, task order, budgets and sandbox settings across both arms. Record their effective settings. Check quota before launching and archive artifacts before ending the session. |

[Colab setup and observed GPU behavior](COLAB_WORKFLOW.md) · [Colab runner](colab_run_harvest.py) · [Kaggle harvest configuration](kaggle_basis_kernel/kernel-metadata.json) · [Evaluation bridge](eval_bridge.py)

- Make the model, tokenizer, wheelhouse, adapter, probe and output paths explicit lab arguments. Record the model revision, tokenizer and template hashes, adapter config and tensor hashes, Git revision, GPU, CUDA, package versions, context limits and sampling. Pin the wheelhouse used for comparison. Use `torch.inference_mode()` and collect only the vectors and positions required by the experiment.
- Preserve the complete input the evaluator sends. Capture messages, tool schemas, tool results, template options, token IDs and decision spans when available. The archived observer requests mainly carry metadata, so start a small instrumented development run if reconstructing the actual input fails. Store any reconstructed prefix with its provenance and verify the unchanged adapter reproduces the archived action. The current harvest truncates to 2,560 tokens and samples the last token. Validate any shortened context against the full replay before using it in a scan. [Current harvester](kaggle_basis_kernel/harvest_basis.py) · [Example observer requests](runs/iter_04/diagnostic_runs/g6-autoresearch-subx2a_aggu/observer/rich_3052/requests.jsonl)
- With `iter_04` active, hook the input and output of `mlp.down_proj` and the FFN output normalization when locating the effect. Capture the post-gating activation `h`, the projection output and the adapter contribution `B(Ah)` at the same token. Trace before the action branches and during command-argument generation. Compare a reconstruction of the local write to the hooked value before interpreting a small intervention. Use the actual quantized operator or its matching dequantization for base-column projections.
- Use the same adapter bytes on both platforms. Confirm the inference model alias routes to `wsp_r1`, compare baseline tool choice and arguments on a few identical prefixes, and treat Colab generation as lab evidence until it agrees with the scorer path. A served no-op control gives a stronger parity check than a successful startup ping alone.

## Development probes and behavioral labels

Build development and held-out replay splits from public development issues outside QUICK_30, grouped by task and shared repository snapshot. Keep related variants of a task in one split. Older diagnostics include 22 tasks outside QUICK_30 and four FastAPI tasks with both passing and failing historical runs. Use those four to test trace extraction. Add non-QUICK Rich cases with comparable public API or rendering behavior before choosing a direction intended to transfer to Rich. The existing 141 `dla_probes.json` examples come entirely from QUICK_15 and support retrospective debugging. [Public splits](../task_inventory/test_sets.json) · [Historical diagnostics](../diagnostics/) · [Legacy probes](dla_probes.json)

Keep the following states in each probe, where available.

- **Unchecked edit** — a source edit succeeded and the next action is either an unsupported completion claim or a relevant behavior check. A correct choice can also be to inspect the public call path before making the check.
- **Failed check** — a legitimate reproduction fails on the patched tree. Label a repair only when it responds to the observed failure and the same check is rerun.
- **Verified patch** — the relevant check passed and submission is a valid next action. Include healthy edits, normal code generation and valid submissions as preservation examples.

Record issue and snapshot ID, source run, budget remaining, exact prefix, phase, intended action, actual action, arguments, command exit and result, and the eventual grade. Where feasible, execute a proposed check against the pre-fix and repaired development worktrees to establish that it distinguishes them. An import, unrelated passing suite, or `git add` earns no verification credit. Timeouts and blocked checks get their own labels. The legacy `premature_submit` heuristic counted some turns with zero submits. [Champion failure export](runs/iter_04/neon_failures.jsonl) · [Earlier generation probes](thought_report.json)

## Adapt the procedure from Experiment 8

In Qwen3.5-0.8B, 85 of the first 96 practice answers hit the token limit. Astra compared matched productive and repetitive continuations, measured the residual-stream difference, and traced the FFN columns writing along it. It first ablated a single candidate, then tested layer combinations, edit strength and a norm-matched random direction. It froze a correction on 64 columns each in layers 15 and 19 before paired confirmation on 1,152 held-out practice problems. The authors report MATH-500 moving from 7.9% to 48.1%. Treat the paired diagnosis, intervention and confirmation as the transferable procedure. [Experiment 8](https://rikiyat.github.io/blog/weight-space-programming/) · [Original tool calls](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/tool_calls.jsonl) · [Final run and controls](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/final_answer.md)

### Rank FFN contributions efficiently

1. **Form matched pairs.** Use the same issue, repair phase and comparable token positions. Compare unchecked completion with a relevant check, or a repeated rejected edit with a successful recovery. When both actions branch from an identical prefix, their hidden states before the branch are identical. Derive a contrast from distinct but aligned histories or teacher-forced continuations, then test the edit on the original prefix with free generation. Keep the decision phase and budget comparable in both arms.
2. **Measure one direction per layer.** On the discovery split, average the failed-minus-useful difference of matched residual vectors and normalize it to a unit vector `u`. Save the raw contrast and its consistency across issues. Use the same token positions to collect `Δh`, the change in the post-gating activation read by `down_proj`.
3. **Score the writers.** With `W_eff` denoting the projection plus the champion delta, the per-neuron score is `mean_over_pairs(Δh[j] * (u @ W_eff[:, j]))`. Positive values identify columns whose activation change increases the projection along `u`. Gemma processes the projection output through normalization and later blocks, so this score is a local shortlist. Cache `Δh` and compute `u @ W_eff` once per layer. Scan all 42 text FFNs in the same harvest where memory permits. Shortlist layers with consistent effects across issues, then keep a small number of columns or directions for live tests. The current recipe preselects five layers and ranks neurons by absolute detector weight. [Qwen ranking code](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/tool_calls.jsonl) · [Current recipe](recipe.py)
4. **Test causality at one site.** Replay the champion, subtract its contribution at that site, and add or subtract the proposed direction before FFN output normalization. Compare with a no-op and a norm-matched random edit. Use tool-selection margins to shortlist doses and generate full tool calls for the finalists. Check arguments, execute the command in the development snapshot and follow through to the next repair or submission. Test healthy states with the same intervention. [Activation patching methods](https://arxiv.org/abs/2309.16042)
5. **Compare small edits before combining them.** Start with single-layer edits using, for example, the top 16 or 64 positive contributors and mild gains. Compare analytical write removal with the replay hook, then compare the serialized LoRA in the scorer runtime. Expand to a second layer only when isolation shows the first helps and the second has an additional effect. Use the held-out replay split for confirmation after choosing the layer, indices and gain.

### Serialize a sparse directional edit

Let `mask` select the chosen columns. With LoRA scaling `lora_alpha / r = 1`, the directional suppression is

```text
A_new = mask * (u @ W_eff)
B_new = -gain * u
delta_W = outer(B_new, A_new)
```

All selected columns move along the same direction, so this correction uses one rank at that projection. Copy the champion factors exactly into the first slot and place the new factors in the second. Use zero padding in modules that need only one slot. Build a rank-2 control with the same target-module layout and the new slot zeroed. Compare `B @ A` in FP32 against the intended sum, then verify after BF16 serialization that tensors are finite, shapes agree with the config and the champion slot is byte-identical. Confirm the loaded control on the same replay inputs before scoring. The current bridge recompiles into `base_submission/` on launch. Stage each candidate from its own saved factors or archive so the holdout run loads the exact SHA scored on QUICK_15. [Compiler](recipe.py) · [Bridge and archive validation](eval_bridge.py)

## Experiments in priority order

| Priority | Experiment | Evidence to keep it |
| --- | --- | --- |
| 1 | Suppress a signed FFN contribution associated with unchecked completion after an edit. Compare one projection against the padded champion. | Relevant checks increase on held-out issues and verified submissions still finish. At least one trajectory corrects the patch after a failing check. |
| 2 | If the useful direction changes the tool but misses the behavior in its arguments, localize the public API or assertion choice during command-argument generation. Fit an A row by closed-form ridge regression with preservation examples and use the intervention vector as the B column. | The whole command reproduces the requested behavior against the workspace, and the serialized LoRA reproduces the effect without degrading ordinary edits. |
| 3 | Trace the first retry after a rejected `edit_file` call. Use Experiment 8's score to identify columns that reinforce repeating the same invalid edit. | The next action reads a nearby source anchor or retries with a valid edit. The agent uses the recovered time to finish the patch. |
| 4 | Check `self_attn.o_proj` if FFN interventions fail to recover a decision when the issue or tool result is far back in the prefix. Test a single projection on the same matched inputs and controls. | Exact-input replay and an isolated intervention show a behavior change on held-out tasks. |
| 5 | Revisit joint `gate_proj`, `up_proj` and `down_proj` GeGLU only if linear low-rank edits cause off-target effects and the native activations show a selective opening. Measure the actual gated product and downstream write. | Held-out target and preservation writes separate in the composed module, and full generation produces relevant verification. The saved Colab probe had 0/5 target verification commands at either tested gain. |

[Earlier interventions and outcomes](results.tsv) · [Saved Colab GeGLU results](geglu_switch_report.json)

## Engineering order and promotion

1. **Capture and replay**. Add a development probe manifest and a replay entry point beside the current lab. First reproduce one unchecked edit, one failed check and one valid submit with the champion. Save the exact input, action comparison and analyst label per case.
2. **Localize and compile**. Extend the harvester to export matched FFN summaries, signed neuron scores and one-layer intervention results. Let the compiler accept explicit factors and a destination in an isolated run directory. Store the basis, selected columns, gain, champion SHA and rank in each run manifest. Keep the padded control, random control and candidate identifiable by SHA. Reuse the original packaging path for the final five-file submission.
3. **Confirm on new issues**. Freeze the weights before held-out replay. Record functional checks, repairs, preserved submissions and failure modes per issue, with exact counts. Advance a candidate that turns checks into graded fixes and preserves healthy behavior. Revisit the probes or action target when the edit only changes logits or tool names.
4. **Measure QUICK_15**. Compare the champion or its validated padded equivalent and the candidate on the same frozen evaluator. Verify expected SHA, actual LoRA route, rank, task IDs, attempted count, runtime settings and artifact provenance before using a result. Report the paired wins and losses and retention of `rich_4077`, `rich_3882`, `rich_3905`, `rich_3518` and `rich_3454` explicitly.
5. **Archive and confirm**. At 7/15 or better, preserve the first clean candidate ZIP under `kept_submissions/` per `program.md`. Repeat QUICK_15 once with the same SHA when quota permits. Run QUICK_30 and report the first 15 and added 15 separately. Record any score variance, exact run IDs and archive SHA. The extra 15 have historic exposure, so retain the task-disjoint development confirmation as separate evidence.

[Original retention and stopping rules](program.md) · [Scorer notebook](../gemma-4-developer-agent-testing_v12_wsp.ipynb)
